import torch

from . import testing  # noqa: F401
from .fused import *  # noqa: F403
from .ops import *  # noqa: F403

__version__ = "2.1"

aten_lib = torch.library.Library("aten", "IMPL")


def enable(lib=aten_lib):
    lib.impl("divide.Tensor", true_divide, "PrivateUse1")  # divide, an alias for div
    lib.impl("divide.Scalar", true_divide, "PrivateUse1")
    lib.impl("div.Tensor", true_divide, "PrivateUse1")
    lib.impl("div.Scalar", true_divide, "PrivateUse1")
    lib.impl(
        "true_divide.Tensor", true_divide, "PrivateUse1"
    )  # true_divide, an alias for div
    lib.impl("true_divide.Scalar", true_divide, "PrivateUse1")
    lib.impl("gt.Tensor", gt, "PrivateUse1")
    lib.impl("gt.Scalar", gt_scalar, "PrivateUse1")
    lib.impl("ge.Tensor", ge, "PrivateUse1")
    lib.impl("ge.Scalar", ge_scalar, "PrivateUse1")
    lib.impl("bitwise_or.Tensor", bitwise_or_tensor, "PrivateUse1")
    lib.impl("bitwise_or.Scalar", bitwise_or_scalar, "PrivateUse1")
    lib.impl("bitwise_or.Scalar_Tensor", bitwise_or_scalar_tensor, "PrivateUse1")
    # lib.impl("mul.Tensor", mul, "PrivateUse1")
    # lib.impl("neg", neg, "PrivateUse1")
    # lib.impl("sub.Tensor", sub, "PrivateUse1")
    lib.impl("ne.Tensor", ne, "PrivateUse1")
    # lib.impl("ne.Scalar", ne_scalar, "PrivateUse1")
    # lib.impl("isnan", isnan, "PrivateUse1")
    lib.impl("full_like", full_like, "PrivateUse1")
    lib.impl("resolve_neg", resolve_neg, "PrivateUse1")
    lib.impl("ones_like", ones_like, "PrivateUse1")
    lib.impl("full", full, "PrivateUse1")
    lib.impl("zeros", zeros, "PrivateUse1")
    lib.impl("cos", cos, "PrivateUse1")
    lib.impl("pow.Scalar", pow_scalar, "PrivateUse1")
    lib.impl("pow.Tensor_Scalar", pow_tensor_scalar, "PrivateUse1")
    lib.impl("pow.Tensor_Tensor", pow_tensor_tensor, "PrivateUse1")
    lib.impl("lt.Tensor", lt, "PrivateUse1")
    lib.impl("lt.Scalar", lt_scalar, "PrivateUse1")
    lib.impl("sin", sin, "PrivateUse1")
    lib.impl("eq.Tensor", eq, "PrivateUse1")
    lib.impl("eq.Scalar", eq_scalar, "PrivateUse1")
    lib.impl("add.Tensor", add, "PrivateUse1")
    lib.impl("sum", sum, "PrivateUse1")
    lib.impl("prod", prod, "PrivateUse1")
    lib.impl("linalg_vector_norm", vector_norm, "PrivateUse1")
    lib.impl("stack", stack, "PrivateUse1")
    lib.impl("arange", arange, "PrivateUse1")
    lib.impl("cumsum", cumsum, "PrivateUse1")


class use_gems:
    def __init__(self):
        self.lib = torch.library.Library("aten", "IMPL")

    def __enter__(self):
        enable(self.lib)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.lib


__all__ = [
    "enable",
    "use_gems",
]
