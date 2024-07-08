import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.random_utils import philox_cuda_seed_offset, uint_to_uniform_float
from flag_gems.utils.shape_utils import volume


@triton.heuristics(
    values={
        "BLOCK": lambda args: 512 if args["N"] <= 512 else 1024,
        "num_warps": lambda args: 4 if args["N"] <= 512 else 8 if args["N"] <= 1024 else 16,  # fmt: skip
    }
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def rand_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    dtype,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + off_0, r0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, r1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, r2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, r3, mask=off_3 < N, eviction_policy="evict_first")


def rand(size, *, dtype=None):
    logging.debug("GEMS RAND")
    out = torch.empty(size, dtype=dtype, device=torch.device("cuda"))
    N = volume(size)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    philox_seed, philox_offset = philox_cuda_seed_offset(N)
    rand_kernel[grid_fn](out, N, philox_seed, philox_offset, dtype)
    return out


if __name__ == "__main__":
    a = rand(size=(10, 2))
    print(a)
