import functools
import itertools
import operator

import numpy as np
import paddle
import paddle.nn.functional as F

from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.variables import standardize_dtype
from keras.src.backend.config import standardize_data_format
from keras_paddle.src.ops.core import convert_to_tensor
from keras_paddle.src.ops.core import to_paddle_dtype

FLOAT_DTYPES = (
    "float8_e4m3fn",
    "float8_e5m2",
    "float16",
    "bfloat16",
    "float32",
    "float64",
)


def _is_float_dtype(dtype):
    return standardize_dtype(dtype) in FLOAT_DTYPES


RESIZE_INTERPOLATIONS = ("bilinear", "nearest", "bicubic")
UNSUPPORTED_INTERPOLATIONS = (
    "lanczos3",
    "lanczos5",
)

AFFINE_TRANSFORM_INTERPOLATIONS = {  # map to order
    "nearest": 0,
    "bilinear": 1,
}
MAP_COORDINATES_FILL_MODES = {
    "constant",
    "nearest",
    "wrap",
    "mirror",
    "reflect",
}
_INDEX_FIXERS = {
    # Out-of-bound indices are handled after the fixer for `constant`
    # and `nearest`, so both just clip here.
    "constant": lambda index, size: paddle.clip(index, 0, size - 1),
    "nearest": lambda index, size: paddle.clip(index, 0, size - 1),
    "wrap": lambda index, size: index % size,
    # `mirror` folds at `size - 1` with half-wavelength `size - 1`.
    "mirror": lambda index, size: paddle.abs(
        (index + size - 1) % (2 * size - 2) - (size - 1)
    ),
    # `reflect` folds at the edge itself with half-wavelength `size`.
    "reflect": lambda index, size: paddle.abs(
        (index + size - 1) % (2 * size) - (size - 1)
    ),
}


def _is_integer(dtype):
    return "int" in standardize_dtype(dtype) or dtype == "bool"


def _nearest_indices_and_weights(coordinate):
    coordinate = paddle.round(coordinate)
    index = coordinate.cast("int64")
    return [(index, 1)]


def _linear_indices_and_weights(coordinate):
    lower = paddle.floor(coordinate)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = lower.cast("int64")
    return [(index, lower_weight), (index + 1, upper_weight)]


def map_coordinates(
    inputs, coordinates, order, fill_mode="constant", fill_value=0.0
):
    input_arr = convert_to_tensor(inputs)
    coordinate_arrs = [convert_to_tensor(c) for c in coordinates]

    if len(coordinate_arrs) != len(input_arr.shape):
        raise ValueError(
            "First dim of `coordinates` must be the same as the rank of "
            "`inputs`. "
            f"Received inputs with shape: {input_arr.shape} and coordinate "
            f"leading dim of {len(coordinate_arrs)}"
        )
    if len(coordinate_arrs[0].shape) < 1:
        shape = (len(coordinate_arrs),) + coordinate_arrs[0].shape
        raise ValueError(
            "Invalid coordinates rank: expected at least rank 2."
            f" Received input with shape: {shape}"
        )

    input_dtype = standardize_dtype(input_arr.dtype)
    # Paddle's `where` cannot mix float and integer operands, and has no
    # uint8 kernel for it, so any input that could reach `where` with a
    # float `fill_value` has to be computed in float32.
    if not _is_float_dtype(input_dtype) or input_dtype in (
        "float16",
        "bfloat16",
        "float8_e4m3fn",
        "float8_e5m2",
    ):
        input_arr = input_arr.cast("float32")
        coordinate_arrs = [c.cast("float32") for c in coordinate_arrs]
    fill_value = paddle.to_tensor(fill_value, dtype=input_arr.dtype)

    index_fixer = _INDEX_FIXERS.get(fill_mode)
    if index_fixer is None:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected one of "
            f"{set(_INDEX_FIXERS.keys())}. Received: fill_mode={fill_mode}"
        )

    if order == 0:
        interp_fun = _nearest_indices_and_weights
    elif order == 1:
        interp_fun = _linear_indices_and_weights
    else:
        raise NotImplementedError("map_coordinates currently requires order<=1")

    if fill_mode == "constant":

        def is_valid(index, size):
            return (0 <= index) & (index < size)

    else:

        def is_valid(index, size):
            return True

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinate_arrs, input_arr.shape):
        interp_nodes = interp_fun(coordinate)
        valid_interp = []
        for index, weight in interp_nodes:
            fixed_index = index_fixer(index, size)
            valid = is_valid(index, size)
            valid_interp.append((fixed_index, valid, weight))
        valid_1d_interpolations.append(valid_interp)

    outputs = []
    for items in itertools.product(*valid_1d_interpolations):
        indices, validities, weights = zip(*items)
        if all(valid is True for valid in validities):
            contribution = input_arr[indices]
        else:
            all_valid = functools.reduce(operator.and_, validities)
            contribution = paddle.where(
                all_valid, input_arr[indices], fill_value
            )
        outputs.append(functools.reduce(operator.mul, weights) * contribution)
    result = functools.reduce(operator.add, outputs)
    if _is_integer(input_dtype):
        result = paddle.round(result)
    return result.cast(to_paddle_dtype(input_dtype))


def _dtype_limits(dtype):
    if dtype == "bool":
        return 0, 1
    info = np.iinfo(dtype)
    return int(info.min), int(info.max)


def _resize_nearest(image, size):
    """Resize with nearest neighbors sampled at the output pixel centers.

    `F.interpolate(mode="nearest")` samples at the top-left corner of every
    output pixel, while the reference implementations sample at its center.
    """
    src_height, src_width = image.shape[-2], image.shape[-1]
    for axis, (src, dst) in enumerate(
        ((src_height, size[0]), (src_width, size[1])), start=2
    ):
        indices = paddle.arange(dst, dtype="float64")
        indices = paddle.floor((indices + 0.5) * (src / dst))
        indices = paddle.clip(indices, 0, src - 1).cast("int64")
        image = paddle.index_select(image, indices, axis=axis)
    return image


def rgb_to_grayscale(images, data_format=None):
    data_format = standardize_data_format(data_format)
    images = convert_to_tensor(images)
    if images.ndim not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    channel_axis = -1 if data_format == "channels_last" else -3
    if images.shape[channel_axis] not in (1, 3):
        raise ValueError(
            "Invalid channel size: expected 3 (RGB) or 1 (Grayscale). "
            f"Received input with shape: images.shape={images.shape}"
        )
    if images.shape[channel_axis] == 3:
        # `multiply` / `add` have no float16 or bfloat16 CPU kernel, and
        # integer inputs have to be weighted in floating point anyway.
        orig_dtype = images.dtype
        r, g, b = paddle.unbind(images.cast("float32"), axis=channel_axis)
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.unsqueeze(channel_axis).cast(orig_dtype)
    return images.clone()


def resize(
    image,
    size,
    interpolation="bilinear",
    antialias=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    fill_mode="constant",
    fill_value=0.0,
    data_format="channels_last",
):
    data_format = standardize_data_format(data_format)
    if interpolation in UNSUPPORTED_INTERPOLATIONS:
        raise ValueError(
            "Resizing with Lanczos interpolation is "
            "not supported by the paddle backend. "
            f"Received: interpolation={interpolation}."
        )
    if interpolation not in RESIZE_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{RESIZE_INTERPOLATIONS}. Received: interpolation={interpolation}"
        )
    if fill_mode != "constant":
        raise ValueError(
            "Invalid value for argument `fill_mode`. Only `'constant'` "
            f"is supported. Received: fill_mode={fill_mode}"
        )
    if pad_to_aspect_ratio and crop_to_aspect_ratio:
        raise ValueError(
            "Only one of `pad_to_aspect_ratio` & `crop_to_aspect_ratio` "
            "can be `True`."
        )
    if not len(size) == 2:
        raise ValueError(
            "Argument `size` must be a tuple of two elements "
            f"(height, width). Received: size={size}"
        )
    size = tuple(size)
    image = convert_to_tensor(image)
    out_dtype = standardize_dtype(image.dtype)
    if out_dtype not in ("float32", "float64"):
        image = image.cast("float32")
    if image.ndim not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={image.shape}"
        )
    has_batch = image.ndim == 4
    if not has_batch:
        image = paddle.unsqueeze(image, axis=0)

    if data_format == "channels_last":
        image = paddle.transpose(image, [0, 3, 1, 2])

    if crop_to_aspect_ratio:
        shape = image.shape
        height, width = shape[-2], shape[-1]
        target_height, target_width = size
        crop_height = int(float(width * target_height) / target_width)
        crop_height = max(min(height, crop_height), 1)
        crop_width = int(float(height * target_width) / target_height)
        crop_width = max(min(width, crop_width), 1)
        crop_box_hstart = int(float(height - crop_height) / 2)
        crop_box_wstart = int(float(width - crop_width) / 2)
        image = image[
            :,
            :,
            crop_box_hstart : crop_box_hstart + crop_height,
            crop_box_wstart : crop_box_wstart + crop_width,
        ]
    elif pad_to_aspect_ratio:
        shape = image.shape
        height, width = shape[-2], shape[-1]
        target_height, target_width = size
        pad_height = int(float(width * target_height) / target_width)
        pad_height = max(height, pad_height)
        pad_width = int(float(height * target_width) / target_height)
        pad_width = max(width, pad_width)
        img_box_hstart = int(float(pad_height - height) / 2)
        img_box_wstart = int(float(pad_width - width) / 2)

        batch_size = image.shape[0]
        channels = image.shape[1]
        if img_box_hstart > 0:
            padded_img = paddle.concat(
                [
                    paddle.full(
                        [batch_size, channels, img_box_hstart, width],
                        fill_value,
                        dtype=image.dtype,
                    ),
                    image,
                    paddle.full(
                        [batch_size, channels, img_box_hstart, width],
                        fill_value,
                        dtype=image.dtype,
                    ),
                ],
                axis=2,
            )
        else:
            padded_img = image
        if img_box_wstart > 0:
            padded_img = paddle.concat(
                [
                    paddle.full(
                        [batch_size, channels, height, img_box_wstart],
                        fill_value,
                        dtype=image.dtype,
                    ),
                    padded_img,
                    paddle.full(
                        [batch_size, channels, height, img_box_wstart],
                        fill_value,
                        dtype=image.dtype,
                    ),
                ],
                axis=3,
            )
        image = padded_img

    if antialias and interpolation not in ("bilinear", "bicubic"):
        # Paddle only supports antialiasing for bilinear and bicubic modes.
        # The parameter is irrelevant for the other modes.
        antialias = False
    if interpolation == "nearest":
        out = _resize_nearest(image, size)
    else:
        out = F.interpolate(
            image,
            size=size,
            mode=interpolation,
            align_corners=False,
            antialias=antialias,
        )

    if data_format == "channels_last":
        out = paddle.transpose(out, [0, 2, 3, 1])

    if not has_batch:
        out = paddle.squeeze(out, axis=0)

    if standardize_dtype(out.dtype) != out_dtype:
        if "int" in out_dtype or out_dtype == "bool":
            # Rounding before the cast avoids truncating e.g. 0.999 to 0.
            out = paddle.round(out)
            out = paddle.clip(out, *_dtype_limits(out_dtype))
        out = out.cast(to_paddle_dtype(out_dtype))
    return out


def affine_transform(
    images,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format=None,
):
    raise NotImplementedError(
        "`affine_transform` is not supported with paddle backend"
    )


def rgb_to_hsv(images, data_format=None):
    # Ref: dm_pix
    data_format = standardize_data_format(data_format)
    images = convert_to_tensor(images)
    dtype = standardize_dtype(images.dtype)
    channels_axis = -1 if data_format == "channels_last" else -3
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if not _is_float_dtype(dtype):
        raise ValueError(
            "Invalid images dtype: expected float dtype. "
            f"Received: images.dtype={dtype}"
        )
    # Paddle CPU has incomplete kernel coverage for these dtypes, so
    # compute in float32 and cast back to the original dtype at the end.
    compute_dtype = dtype if dtype not in ("float16", "bfloat16") else "float32"
    if dtype != compute_dtype:
        images = images.cast("float32")
    if dtype.startswith("float8"):
        eps = np.finfo("float32").eps
    else:
        eps = paddle.finfo(to_paddle_dtype(dtype)).eps
    images = paddle.where(paddle.abs(images) < eps, 0.0, images)
    red, green, blue = paddle.split(images, 3, channels_axis)
    red = paddle.squeeze(red, channels_axis)
    green = paddle.squeeze(green, channels_axis)
    blue = paddle.squeeze(blue, channels_axis)

    def rgb_planes_to_hsv_planes(r, g, b):
        value = paddle.maximum(paddle.maximum(r, g), b)
        minimum = paddle.minimum(paddle.minimum(r, g), b)
        range_ = value - minimum

        safe_value = paddle.where(value > 0, value, 1.0)
        safe_range = paddle.where(range_ > 0, range_, 1.0)

        saturation = paddle.where(value > 0, range_ / safe_value, 0.0)
        norm = 1.0 / (6.0 * safe_range)

        hue = paddle.where(
            value == g,
            norm * (b - r) + 2.0 / 6.0,
            norm * (r - g) + 4.0 / 6.0,
        )
        hue = paddle.where(value == r, norm * (g - b), hue)
        hue = paddle.where(range_ > 0, hue, 0.0) + (hue < 0.0).cast(hue.dtype)
        return hue, saturation, value

    hue, saturation, value = rgb_planes_to_hsv_planes(red, green, blue)
    images = paddle.stack([hue, saturation, value], axis=channels_axis)
    return images.cast(to_paddle_dtype(dtype))


def hsv_to_rgb(images, data_format=None):
    # Ref: dm_pix
    data_format = standardize_data_format(data_format)
    images = convert_to_tensor(images)
    dtype = standardize_dtype(images.dtype)
    channels_axis = -1 if data_format == "channels_last" else -3
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if not _is_float_dtype(dtype):
        raise ValueError(
            "Invalid images dtype: expected float dtype. "
            f"Received: images.dtype={dtype}"
        )
    # Paddle CPU has incomplete kernel coverage for these dtypes, so
    # compute in float32 and cast back to the original dtype at the end.
    compute_dtype = dtype if dtype not in ("float16", "bfloat16") else "float32"
    if dtype != compute_dtype:
        images = images.cast("float32")
    hue, saturation, value = paddle.split(images, 3, channels_axis)
    hue = paddle.squeeze(hue, channels_axis)
    saturation = paddle.squeeze(saturation, channels_axis)
    value = paddle.squeeze(value, channels_axis)

    def hsv_planes_to_rgb_planes(hue, saturation, value):
        dh = (hue % 1.0) * 6.0
        dr = paddle.clip(paddle.abs(dh - 3.0) - 1.0, 0.0, 1.0)
        dg = paddle.clip(2.0 - paddle.abs(dh - 2.0), 0.0, 1.0)
        db = paddle.clip(2.0 - paddle.abs(dh - 4.0), 0.0, 1.0)
        one_minus_s = 1.0 - saturation

        red = value * (one_minus_s + saturation * dr)
        green = value * (one_minus_s + saturation * dg)
        blue = value * (one_minus_s + saturation * db)
        return red, green, blue

    red, green, blue = hsv_planes_to_rgb_planes(hue, saturation, value)
    images = paddle.stack([red, green, blue], axis=channels_axis)
    return images.cast(to_paddle_dtype(dtype))


def perspective_transform(
    images,
    start_points,
    end_points,
    interpolation="bilinear",
    fill_value=0,
    data_format=None,
):
    data_format = standardize_data_format(data_format)

    images = convert_to_tensor(images)
    dtype = standardize_dtype(images.dtype)
    # Paddle's CPU `arange`/`empty` and the linear-algebra ops have no
    # float16/bfloat16 kernels, so the transform math runs in float32.
    if dtype in ("float16", "bfloat16"):
        compute_dtype = "float32"
    else:
        compute_dtype = dtype
    images = images.cast(to_paddle_dtype(compute_dtype))
    start_points = convert_to_tensor(start_points, dtype=compute_dtype)
    end_points = convert_to_tensor(end_points, dtype=compute_dtype)

    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{set(AFFINE_TRANSFORM_INTERPOLATIONS)}. Received: "
            f"interpolation={interpolation}"
        )

    if images.ndim not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    if start_points.ndim not in (2, 3) or start_points.shape[-2:] != (4, 2):
        raise ValueError(
            "Invalid start_points shape: expected (4,2) for a single image"
            f" or (N,4,2) for a batch. Received shape: {start_points.shape}"
        )
    if end_points.ndim not in (2, 3) or end_points.shape[-2:] != (4, 2):
        raise ValueError(
            "Invalid end_points shape: expected (4,2) for a single image"
            f" or (N,4,2) for a batch. Received shape: {end_points.shape}"
        )
    if start_points.shape != end_points.shape:
        raise ValueError(
            "start_points and end_points must have the same shape."
            f" Received start_points.shape={start_points.shape}, "
            f"end_points.shape={end_points.shape}"
        )

    need_squeeze = False
    if images.ndim == 3:
        images = images.unsqueeze(0)
        need_squeeze = True

    if start_points.ndim == 2:
        start_points = start_points.unsqueeze(0)
    if end_points.ndim == 2:
        end_points = end_points.unsqueeze(0)

    if data_format == "channels_first":
        images = paddle.transpose(images, [0, 2, 3, 1])

    batch_size, height, width, channels = images.shape

    transforms = compute_homography_matrix(start_points, end_points)

    if transforms.ndim == 1:
        transforms = transforms.unsqueeze(0)
    if transforms.shape[0] == 1 and batch_size > 1:
        transforms = transforms.tile([batch_size, 1])

    grid_x, grid_y = paddle.meshgrid(
        paddle.arange(width, dtype=to_paddle_dtype(compute_dtype)),
        paddle.arange(height, dtype=to_paddle_dtype(compute_dtype)),
        indexing="xy",
    )

    output = paddle.empty(
        [batch_size, height, width, channels],
        dtype=to_paddle_dtype(dtype),
    )

    for i in range(batch_size):
        a0, a1, a2, a3, a4, a5, a6, a7 = transforms[i]
        denom = a6 * grid_x + a7 * grid_y + 1.0
        x_in = (a0 * grid_x + a1 * grid_y + a2) / denom
        y_in = (a3 * grid_x + a4 * grid_y + a5) / denom

        coords = paddle.stack([y_in.flatten(), x_in.flatten()], axis=0)
        mapped_channels = []
        for channel in range(channels):
            channel_img = images[i, :, :, channel]
            mapped_channel = map_coordinates(
                channel_img,
                coords,
                order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                fill_mode="constant",
                fill_value=fill_value,
            )
            mapped_channels.append(mapped_channel.reshape([height, width]))
        output[i] = paddle.stack(mapped_channels, axis=-1)

    if data_format == "channels_first":
        output = paddle.transpose(output, [0, 3, 1, 2])
    if need_squeeze:
        output = output.squeeze(0)

    return output.cast(to_paddle_dtype(dtype))


def compute_homography_matrix(start_points, end_points):
    start_points = convert_to_tensor(start_points)
    end_points = convert_to_tensor(end_points)
    dtype = result_type(
        start_points.dtype,
        end_points.dtype,
        paddle.float32,
    )
    start_points = start_points.cast(to_paddle_dtype(dtype))
    end_points = end_points.cast(to_paddle_dtype(dtype))

    start_x1, start_y1 = start_points[:, 0, 0], start_points[:, 0, 1]
    start_x2, start_y2 = start_points[:, 1, 0], start_points[:, 1, 1]
    start_x3, start_y3 = start_points[:, 2, 0], start_points[:, 2, 1]
    start_x4, start_y4 = start_points[:, 3, 0], start_points[:, 3, 1]

    end_x1, end_y1 = end_points[:, 0, 0], end_points[:, 0, 1]
    end_x2, end_y2 = end_points[:, 1, 0], end_points[:, 1, 1]
    end_x3, end_y3 = end_points[:, 2, 0], end_points[:, 2, 1]
    end_x4, end_y4 = end_points[:, 3, 0], end_points[:, 3, 1]

    coefficient_matrix = paddle.stack(
        [
            paddle.stack(
                [
                    end_x1,
                    end_y1,
                    paddle.ones_like(end_x1),
                    paddle.zeros_like(end_x1),
                    paddle.zeros_like(end_x1),
                    paddle.zeros_like(end_x1),
                    -start_x1 * end_x1,
                    -start_x1 * end_y1,
                ],
                axis=-1,
            ),
            paddle.stack(
                [
                    paddle.zeros_like(end_x1),
                    paddle.zeros_like(end_x1),
                    paddle.zeros_like(end_x1),
                    end_x1,
                    end_y1,
                    paddle.ones_like(end_x1),
                    -start_y1 * end_x1,
                    -start_y1 * end_y1,
                ],
                axis=-1,
            ),
            paddle.stack(
                [
                    end_x2,
                    end_y2,
                    paddle.ones_like(end_x2),
                    paddle.zeros_like(end_x2),
                    paddle.zeros_like(end_x2),
                    paddle.zeros_like(end_x2),
                    -start_x2 * end_x2,
                    -start_x2 * end_y2,
                ],
                axis=-1,
            ),
            paddle.stack(
                [
                    paddle.zeros_like(end_x2),
                    paddle.zeros_like(end_x2),
                    paddle.zeros_like(end_x2),
                    end_x2,
                    end_y2,
                    paddle.ones_like(end_x2),
                    -start_y2 * end_x2,
                    -start_y2 * end_y2,
                ],
                axis=-1,
            ),
            paddle.stack(
                [
                    end_x3,
                    end_y3,
                    paddle.ones_like(end_x3),
                    paddle.zeros_like(end_x3),
                    paddle.zeros_like(end_x3),
                    paddle.zeros_like(end_x3),
                    -start_x3 * end_x3,
                    -start_x3 * end_y3,
                ],
                axis=-1,
            ),
            paddle.stack(
                [
                    paddle.zeros_like(end_x3),
                    paddle.zeros_like(end_x3),
                    paddle.zeros_like(end_x3),
                    end_x3,
                    end_y3,
                    paddle.ones_like(end_x3),
                    -start_y3 * end_x3,
                    -start_y3 * end_y3,
                ],
                axis=-1,
            ),
            paddle.stack(
                [
                    end_x4,
                    end_y4,
                    paddle.ones_like(end_x4),
                    paddle.zeros_like(end_x4),
                    paddle.zeros_like(end_x4),
                    paddle.zeros_like(end_x4),
                    -start_x4 * end_x4,
                    -start_x4 * end_y4,
                ],
                axis=-1,
            ),
            paddle.stack(
                [
                    paddle.zeros_like(end_x4),
                    paddle.zeros_like(end_x4),
                    paddle.zeros_like(end_x4),
                    end_x4,
                    end_y4,
                    paddle.ones_like(end_x4),
                    -start_y4 * end_x4,
                    -start_y4 * end_y4,
                ],
                axis=-1,
            ),
        ],
        axis=1,
    )

    target_vector = paddle.stack(
        [
            start_x1,
            start_y1,
            start_x2,
            start_y2,
            start_x3,
            start_y3,
            start_x4,
            start_y4,
        ],
        axis=-1,
    ).unsqueeze(-1)

    homography_matrix = paddle.linalg.solve(coefficient_matrix, target_vector)
    homography_matrix = homography_matrix.reshape([-1, 8])
    homography_matrix = homography_matrix.cast(to_paddle_dtype(dtype))
    return homography_matrix


def gaussian_blur(
    images, kernel_size=(3, 3), sigma=(1.0, 1.0), data_format=None
):
    def _get_gaussian_kernel1d(size, sigma):
        x = paddle.arange(size, dtype=compute_dtype) - (size - 1) / 2
        kernel1d = paddle.exp(-0.5 * (x / sigma) ** 2)
        return kernel1d / paddle.sum(kernel1d)

    data_format = standardize_data_format(data_format)
    images = convert_to_tensor(images)
    input_dtype = standardize_dtype(images.dtype)
    # The Paddle CPU depthwise_conv2d kernel is only registered for
    # float32 and float64.
    compute_dtype = input_dtype
    if input_dtype in ("float16", "bfloat16"):
        compute_dtype = "float32"

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    kernel_size = convert_to_tensor(kernel_size)
    sigma = convert_to_tensor(sigma, dtype=compute_dtype)

    # Sizes can be tensors; resolve them to ints before indexing.
    kernel_height = int(kernel_size[0])
    kernel_width = int(kernel_size[1])
    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise NotImplementedError(
            "gaussian_blur with an even kernel size is not supported by "
            "the paddle backend."
        )

    need_squeeze = False
    if images.ndim == 3:
        images = images.unsqueeze(0)
        need_squeeze = True

    if data_format == "channels_last":
        images = paddle.transpose(images, [0, 3, 1, 2])
    num_channels = images.shape[1]

    if input_dtype != compute_dtype:
        images = images.cast("float32")

    kernel1d_x = _get_gaussian_kernel1d(kernel_height, sigma[0])
    kernel1d_y = _get_gaussian_kernel1d(kernel_width, sigma[1])
    kernel = paddle.outer(kernel1d_y, kernel1d_x)
    kernel = kernel.reshape([1, 1, kernel_height, kernel_width])
    kernel = kernel.tile([num_channels, 1, 1, 1]).astype(compute_dtype)

    # Odd kernel: Paddle's SAME pads with (k - 1) // 2 on each side,
    # which is the same as what the reference implementations use.
    blurred_images = F.conv2d(
        images,
        kernel,
        stride=1,
        padding=kernel_height // 2,
        groups=num_channels,
    )

    if data_format == "channels_last":
        blurred_images = paddle.transpose(blurred_images, [0, 2, 3, 1])

    if need_squeeze:
        blurred_images = blurred_images.squeeze(0)

    if input_dtype != compute_dtype:
        blurred_images = blurred_images.cast(to_paddle_dtype(input_dtype))
    return blurred_images


def elastic_transform(
    images,
    alpha=20.0,
    sigma=5.0,
    interpolation="bilinear",
    fill_mode="reflect",
    fill_value=0.0,
    seed=None,
    data_format=None,
):
    raise NotImplementedError(
        "`elastic_transform` is not supported with paddle backend"
    )


def scale_and_translate(
    images,
    output_shape,
    scale,
    translation,
    spatial_dims,
    method,
    antialias=True,
):
    raise NotImplementedError(
        "`scale_and_translate` is not supported with paddle backend"
    )


def sobel_edges(images, data_format=None):
    data_format = standardize_data_format(data_format)
    images = convert_to_tensor(images)
    dtype = standardize_dtype(images.dtype)

    # Ensure images are in NCHW format for convolution
    if data_format == "channels_last":
        images_nchw = paddle.transpose(images, [0, 3, 1, 2])
    else:
        images_nchw = images

    channels = images_nchw.shape[1]

    # Paddle CPU only registers the `depthwise_conv2d` kernel for
    # float32 and float64, so compute in float32 and cast back.
    compute_dtype = dtype if dtype not in ("float16", "bfloat16") else "float32"
    if dtype != compute_dtype:
        images_nchw = images_nchw.cast("float32")

    # Sobel kernels
    sobel_x = paddle.to_tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=compute_dtype,
    )
    sobel_y = paddle.to_tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=compute_dtype,
    )

    # Reshape for depthwise conv: (out_channels, in_channels/groups, H, W)
    kernel_x = sobel_x.reshape([1, 1, 3, 3]).tile([channels, 1, 1, 1])
    kernel_y = sobel_y.reshape([1, 1, 3, 3]).tile([channels, 1, 1, 1])

    # Apply depthwise convolutions
    edges_x = F.conv2d(images_nchw, kernel_x, padding=1, groups=channels)
    edges_y = F.conv2d(images_nchw, kernel_y, padding=1, groups=channels)

    # Stack to get (N, C, H, W, 2)
    edges = paddle.stack([edges_y, edges_x], axis=-1)

    if data_format == "channels_last":
        # Convert to NHWC format: (N, C, H, W, 2) -> (N, H, W, C, 2)
        edges = paddle.transpose(edges, [0, 2, 3, 1, 4])

    if dtype != compute_dtype:
        edges = edges.cast(to_paddle_dtype(dtype))
    return edges
