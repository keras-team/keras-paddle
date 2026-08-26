import contextlib
import weakref

import numpy as np
import paddle

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import (
    standardize_dtype as _keras_standardize_dtype,
)
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.config import floatx

DEFAULT_DEVICE = "cpu"

PADDLE_DTYPES = {
    "float16": paddle.float16,
    "float32": paddle.float32,
    "float64": paddle.float64,
    "uint8": paddle.uint8,
    "uint16": paddle.int32,
    "uint32": paddle.int64,
    "int8": paddle.int8,
    "int16": paddle.int16,
    "int32": paddle.int32,
    "int64": paddle.int64,
    "bfloat16": paddle.bfloat16,
    "bool": paddle.bool,
    "float8_e4m3fn": paddle.float8_e4m3fn,
    "float8_e5m2": paddle.float8_e5m2,
    "complex64": paddle.complex64,
    "complex128": paddle.complex128,
}

_weak_tensors = weakref.WeakSet()


def standardize_dtype(dtype):
    if hasattr(dtype, "name"):
        dtype = dtype.name.lower()
    elif isinstance(dtype, str):
        dtype = dtype.lower()
    return _keras_standardize_dtype(dtype)


def to_paddle_dtype(dtype):
    if isinstance(dtype, paddle.dtype):
        return dtype
    standardized_dtype = PADDLE_DTYPES.get(standardize_dtype(dtype), None)
    if standardized_dtype is None:
        raise ValueError(f"Unsupported dtype for Paddle: {dtype}")
    return standardized_dtype


def _parse_device_input(device_name):
    if isinstance(device_name, str):
        device_name = device_name.lower()
        if device_name.startswith("cpu"):
            return "cpu"
        return device_name
    raise ValueError(
        "Invalid value for argument `device_name`. "
        "Expected a string like 'gpu:0' or 'cpu'. "
        f"Received: device_name='{device_name}'"
    )


@contextlib.contextmanager
def device_scope(device_name):
    previous_device = paddle.get_device()
    current_device = _parse_device_input(device_name)
    paddle.set_device(current_device)
    global_state.set_global_attribute("paddle_device", current_device)
    try:
        yield current_device
    finally:
        paddle.set_device(previous_device)
        global_state.set_global_attribute("paddle_device", previous_device)


def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if sparse:
        raise ValueError("`sparse=True` is not supported with paddle backend")
    if ragged:
        raise ValueError("`ragged=True` is not supported with paddle backend")
    if isinstance(x, KerasVariable) or is_tensor(x):
        if isinstance(x, KerasVariable):
            x = x.value
        if dtype is not None:
            x = x.cast(to_paddle_dtype(dtype))
        return x
    if isinstance(x, (bool, int, float, complex)):
        if dtype is not None:
            dt = to_paddle_dtype(dtype)
        elif isinstance(x, bool):
            dt = paddle.bool
        elif isinstance(x, int):
            dt = paddle.int64 if x < -(2**31) or x >= 2**31 else paddle.int32
        elif isinstance(x, float):
            dt = to_paddle_dtype(floatx())
        else:
            dt = paddle.complex64
        t = paddle.to_tensor(x, dtype=dt)
        if dtype is None:
            _weak_tensors.add(t)
        return t

    if isinstance(x, (list, tuple)):
        if len(x) > 0 and any(
            is_tensor(item) or isinstance(item, KerasVariable)
            for item in tree.flatten(x)
        ):
            return paddle.stack(
                [convert_to_tensor(x1, dtype=dtype) for x1 in x]
            )
    elif not isinstance(x, (bool, int, float)):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        dtype = dtype or x.dtype
    if dtype is None:
        dtype = result_type(
            *[getattr(item, "dtype", type(item)) for item in tree.flatten(x)]
        )
    dtype = to_paddle_dtype(dtype)
    return paddle.to_tensor(x, dtype=dtype)


def convert_to_numpy(x):
    def transform(x):
        if is_tensor(x):
            if not x.stop_gradient:
                x = x.detach()
        return np.array(x)

    if isinstance(x, (list, tuple)):
        return np.array([transform(e) for e in x])
    return transform(x)


def is_tensor(x):
    return isinstance(x, paddle.Tensor)


def shape(x):
    return tuple(d if d >= 0 else None for d in x.shape)


def cast(x, dtype):
    dtype = to_paddle_dtype(dtype)
    if isinstance(x, KerasVariable):
        x = x.value
    if is_tensor(x):
        if x.dtype == dtype:
            return x
        return x.cast(dtype)
    return convert_to_tensor(x, dtype)


def compute_output_spec(fn, *args, **kwargs):
    raise NotImplementedError(
        "`compute_output_spec` is not yet implemented in keras-paddle."
    )


def cond(pred, true_fn, false_fn):
    if bool(pred):
        return true_fn()
    return false_fn()


def stop_gradient(variable):
    if isinstance(variable, KerasVariable):
        variable = variable.value
    return variable.detach()
