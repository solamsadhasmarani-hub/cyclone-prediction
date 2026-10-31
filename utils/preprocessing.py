from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_SIZE = (224, 224)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PREPROCESSING_CONFIG = {
    "image_size": [224, 224],
    "color_mode": "RGB",
    "input_dtype": "float32",
    "resize": "tf.image.resize bilinear with antialias=True",
    "normalization": "tf.keras.applications.mobilenet_v2.preprocess_input",
    "normalization_range": [-1.0, 1.0],
}


def load_rgb_image(source: bytes | bytearray | str | Path | Image.Image) -> Image.Image:
    """Decode and validate an image, returning an independent RGB image."""
    try:
        if isinstance(source, Image.Image):
            source.load()
            image = source.copy()
        elif isinstance(source, (bytes, bytearray)):
            with Image.open(BytesIO(source)) as opened:
                opened.load()
                image = opened.copy()
        else:
            with Image.open(source) as opened:
                opened.load()
                image = opened.copy()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("The uploaded file is not a valid or readable image.") from error

    return image.convert("RGB")


def preprocess_image(source: bytes | bytearray | str | Path | Image.Image) -> np.ndarray:
    """Decode and resize an image for previews and basic input validation."""
    image = load_rgb_image(source).resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def mobilenet_input(image: np.ndarray) -> np.ndarray:
    """Apply the same TensorFlow resize and MobileNetV2 normalization as training."""
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for model inference. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from error
    tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    if tensor.shape.rank == 3:
        tensor = tf.expand_dims(tensor, axis=0)
    tensor = tf.image.resize(tensor, IMAGE_SIZE, method="bilinear", antialias=True)
    return preprocess_input(tensor).numpy()


def image_tensor(source: bytes | bytearray | str | Path | Image.Image) -> np.ndarray:
    """Build a batched tensor using the same normalization as model training."""
    image = np.asarray(load_rgb_image(source), dtype=np.float32)
    return mobilenet_input(image)
