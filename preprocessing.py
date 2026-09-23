import cv2
import numpy as np
import torch
from PIL import Image


IMAGE_SIZE = 256

SRAD_ITERATIONS = 30
SRAD_STEP = 0.20
SRAD_DECAY = 0.125

CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID = (8, 8)


def robust_unit_scale(image_u8):
    image = np.asarray(image_u8, dtype=np.float32)

    low, high = np.percentile(image, (0.5, 99.5))

    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    return np.clip(
        (image - low) / (high - low),
        0.0,
        1.0,
    )


def srad(
    image_u8,
    iterations=SRAD_ITERATIONS,
    step=SRAD_STEP,
    decay=SRAD_DECAY,
    eps=1e-8,
):
    if not 0 < step <= 0.25:
        raise ValueError(
            "SRAD step must satisfy 0 < step <= 0.25."
        )

    image = robust_unit_scale(image_u8)

    if iterations == 0 or image.max() == image.min():
        return np.round(image * 255).astype(np.uint8)

    height, width = image.shape
    result = np.maximum(image, eps).astype(np.float32)

    for iteration in range(iterations):
        central = result[
            height // 10 : max(height // 10 + 1, 9 * height // 10),
            width // 10 : max(width // 10 + 1, 9 * width // 10),
        ]

        roi = central[
            central > np.percentile(central, 5)
        ]

        if roi.size < 32:
            roi = central.ravel()

        mean = float(roi.mean())
        variance = float(roi.var())

        q0_squared = max(
            variance / (mean * mean + eps),
            eps,
        ) * np.exp(-decay * iteration)

        padded = np.pad(
            result,
            1,
            mode="edge",
        )

        north = padded[:-2, 1:-1] - result
        south = padded[2:, 1:-1] - result
        west = padded[1:-1, :-2] - result
        east = padded[1:-1, 2:] - result

        gradient_squared = (
            north**2
            + south**2
            + west**2
            + east**2
        ) / (result**2 + eps)

        laplacian = (
            north + south + west + east
        ) / (result + eps)

        q_squared = np.maximum(
            (
                0.5 * gradient_squared
                - 0.0625 * laplacian**2
            )
            / (
                (1 + 0.25 * laplacian) ** 2
                + eps
            ),
            0,
        )

        denominator = (
            q_squared - q0_squared
        ) / (
            q0_squared * (1 + q0_squared)
            + eps
        )

        coefficient = np.clip(
            1 / (1 + denominator),
            0,
            1,
        )

        coefficient_padded = np.pad(
            coefficient,
            1,
            mode="edge",
        )

        divergence = (
            coefficient * north
            + coefficient_padded[2:, 1:-1] * south
            + coefficient * west
            + coefficient_padded[1:-1, 2:] * east
        )

        result = np.clip(
            result + 0.25 * step * divergence,
            eps,
            1,
        )

    if not np.isfinite(result).all():
        raise FloatingPointError(
            "SRAD produced non-finite values."
        )

    return np.round(result * 255).astype(np.uint8)


def apply_clahe(
    image_u8,
    clip_limit=CLAHE_CLIP_LIMIT,
    grid=CLAHE_GRID,
):
    if image_u8.ndim != 2:
        raise ValueError(
            "CLAHE expects a two-dimensional image."
        )

    if image_u8.dtype != np.uint8:
        image_u8 = np.clip(
            image_u8,
            0,
            255,
        ).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(map(int, grid)),
    )

    return clahe.apply(image_u8)


def preprocess_ultrasound(image):
    if image is None:
        raise ValueError(
            "Please upload an ultrasound image."
        )

    if not isinstance(image, Image.Image):
        image = Image.fromarray(
            np.asarray(image)
        )

    original_gray = np.asarray(
        image.convert("L"),
        dtype=np.uint8,
    )

    despeckled = srad(original_gray)
    enhanced = apply_clahe(despeckled)

    resized = cv2.resize(
        enhanced,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    normalized = (
        resized.astype(np.float32) / 255.0
    )

    # Same normalization used during model training.
    normalized = (
        normalized - 0.5
    ) / 0.5

    tensor = torch.from_numpy(
        np.ascontiguousarray(normalized)
    )

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return {
        "original": original_gray,
        "despeckled": despeckled,
        "enhanced": enhanced,
        "tensor": tensor,
        "original_size": original_gray.shape,
    }