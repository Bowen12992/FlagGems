import torch
import triton

from .utils import TOTAL_CORE_NUM


def argmax_heur_block_m(args):
    return 4 if args["M"] < 4096 else 8


def argmax_heur_block_n(args):
    return min(4096, triton.next_power_of_2(args["N"]))


def argmin_heur_block_m(args):
    return 4 if args["M"] < 4096 else 8


def argmin_heur_block_n(args):
    return min(4096, triton.next_power_of_2(args["N"]))


def bmm_heur_divisible_m(args):
    return args["M"] % args["TILE_M"] == 0


def bmm_heur_divisible_n(args):
    return args["N"] % args["TILE_N"] == 0


def bmm_heur_divisible_k(args):
    return args["K"] % args["TILE_K"] == 0


def dropout_heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 1024


def exponential_heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 1024


def exponential_heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16


def gather_heur_block_m(args):
    return min(4, triton.next_power_of_2(triton.cdiv(args["N"], 2048)))


def gather_heur_block_n(args):
    return min(2048, triton.next_power_of_2(args["N"]))


def index_select_heur_block_m(args):
    return min(4, triton.next_power_of_2(triton.cdiv(256, args["N"])))


def index_select_heur_block_n(args):
    m = min(triton.next_power_of_2(triton.cdiv(args["N"], 16)), 512)
    return max(m, 16)


def mm_heur_even_k(args):
    return args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0


def rand_heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 1024


def randn_heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 1024


def softmax_heur_tile_k(args):
    MAX_TILE_K = 8192
    NUM_SMS = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    tile_k = 1
    upper_bound = min(args["K"], MAX_TILE_K)
    while tile_k <= upper_bound:
        num_blocks = args["M"] * triton.cdiv(args["K"], tile_k)
        num_waves = num_blocks / NUM_SMS
        if (num_waves > 1) and (tile_k * 2 <= upper_bound):
            tile_k *= 2
        else:
            break
    return tile_k


def softmax_heur_tile_mode_non_inner(args):
    M, N, K, TILE_N, TILE_K = (
        args["M"],
        args["N"],
        args["K"],
        args["TILE_N"],
        args["TILE_K"],
    )
    one_tile_k = TILE_K * max(TOTAL_CORE_NUM // M, 1) >= K
    one_tile_n = TILE_N >= N
    if one_tile_n and one_tile_k:
        return 0
    elif one_tile_n and not one_tile_k:
        return 1
    else:
        return 2


def softmax_heur_tile_mode_inner(args):
    one_tile_m = args["BLOCK_M"] * TOTAL_CORE_NUM >= args["M"]
    one_tile_n = args["BLOCK_N"] >= args["N"]
    if one_tile_n and one_tile_m:
        return 0
    elif one_tile_n and not one_tile_m:
        return 1
    else:
        return 2


def uniform_heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 1024


def uniform_heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16


def upsample_nearest2d_SAME_H(args):
    return args["OH"] == args["IH"]


def upsample_nearest2d_SAME_W(args):
    return args["OW"] == args["IW"]


def vdot_heur_block_size(args):
    n = args["n_elements"]
    if n < 1024:
        return 32
    elif n < 8192:
        return 256
    else:
        return 1024


def linspace_heur_inner_block_size(args):
    n = args["BLOCK_SIZE"]
    if n < 1024:
        return 64
    elif n < 8192:
        return 1024
    else:
        return 8192


def simple_elementwise_blocksize_heur(args):
    return 1024


HEURISTICS_CONFIGS = {
    "argmax": {
        "BLOCK_M": argmax_heur_block_m,
        "BLOCK_N": argmax_heur_block_n,
    },
    "argmin": {
        "BLOCK_M": argmin_heur_block_m,
        "BLOCK_N": argmin_heur_block_n,
    },
    "bmm": {
        "DIVISIBLE_M": bmm_heur_divisible_m,
        "DIVISIBLE_N": bmm_heur_divisible_n,
        "DIVISIBLE_K": bmm_heur_divisible_k,
    },
    "dropout": {
        "BLOCK": dropout_heur_block,
    },
    "exponential_": {
        "BLOCK": exponential_heur_block,
    },
    "gather": {
        "BLOCK_M": gather_heur_block_m,
        "BLOCK_N": gather_heur_block_n,
    },
    "index_select": {
        "BLOCK_M": index_select_heur_block_m,
        "BLOCK_N": index_select_heur_block_n,
    },
    "mm": {
        "EVEN_K": mm_heur_even_k,
    },
    "rand": {
        "BLOCK": rand_heur_block,
    },
    "randn": {
        "BLOCK": randn_heur_block,
    },
    "softmax_non_inner": {
        "TILE_MODE": softmax_heur_tile_mode_non_inner,
    },
    "softmax_inner": {
        "TILE_MODE": softmax_heur_tile_mode_inner,
    },
    "softmax_backward_non_inner": {
        "TILE_MODE": softmax_heur_tile_mode_non_inner,
    },
    "softmax_backward_inner": {
        "TILE_MODE": softmax_heur_tile_mode_inner,
    },
    "uniform": {
        "BLOCK": uniform_heur_block,
    },
    "upsample_nearest2d": {
        "SAME_H": upsample_nearest2d_SAME_H,
        "SAME_W": upsample_nearest2d_SAME_W,
    },
    "var_mean": {},
    "batch_norm": {},
    "vdot": {
        "BLOCK_SIZE": vdot_heur_block_size,
    },
    "linspace": {
        "INNER_BLOCK_SIZE": linspace_heur_inner_block_size,
    },
    "elementwise_generic": {
        "BLOCK_SIZE": simple_elementwise_blocksize_heur,
        "num_warps": lambda args: 8,
    },
}
