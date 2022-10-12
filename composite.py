import cv2
import numpy as np

from src.color.color_correction import color_correction
from src.conv_pyr.convolution_pyramid import convlution_pyramid
from utils.image import read_image, save_image


def composite(
    src_path: str, trg_path: str, alpha_path: str, w: list, save_path: str
) -> None:
    """Composite src image to trg image with alpha mask.

    Args:
        src_path: Path to source image.
        trg_path: Path to target image.
        alpha_path: Path to alpha mask.
        w: Weights of color correction.
        save_path: Path to save result.
    """
    src = read_image(src_path)
    trg = read_image(trg_path)
    alpha = read_image(alpha_path)
    alpha = alpha[:, :, 0]

    mask = cv2.dilate(
        np.uint8(np.where(alpha > 0, 1, 0) * 255),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    mask = np.uint8(mask / 255)

    membrane = convlution_pyramid(src, trg, mask)

    output = color_correction(src, trg, mask, alpha, membrane, w)
    save_image(output, save_path)
