import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic, tl_extra_shim

_pow = tl_extra_shim.pow


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func(x, exponent):
    return _pow(x.to(tl.float32), exponent)


def pow_tensor_tensor(A, exponent):
    logging.debug("GEMS POW_TENSOR_TENSOR")
    return pow_func(A, exponent)


def pow_tensor_tensor_(A, exponent):
    logging.debug("GEMS POW_TENSOR_TENSOR_")
    return pow_func(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    return _pow(x.to(tl.float32), exponent)


def pow_tensor_scalar(A, exponent):
    logging.debug("GEMS POW_TENSOR_SCALAR")
    return pow_func_tensor_scalar(A, exponent)


def pow_tensor_scalar_(A, exponent):
    logging.debug("GEMS POW_TENSOR_SCALAR_")
    return pow_func_tensor_scalar(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return _pow(x.to(tl.float32), exponent)


def pow_scalar(A, exponent):
    logging.debug("GEMS POW_SCALAR")
    return pow_func_scalar_tensor(A, exponent)
