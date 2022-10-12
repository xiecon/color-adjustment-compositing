import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
    """Read image and rescale the output from integer data types to the range [0, 1].

    Args:
        path: Path to the image.

    Returns:
        Image array in the range [0, 1].
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return image


def save_image(image: np.ndarray, save_path: str) -> None:
    """Save image and clip the input to the range [0, 1] before converting to integer data types.

    Args:
        image: Image array.
        save_path: Path to save the image.
    """
    image = np.clip(image, 0, 1)
    image = image * 255.0
    image = np.uint8(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image)
