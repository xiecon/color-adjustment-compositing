import numpy as np


def compute_dominant_color(
    target: np.ndarray, mask: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Compute dominant color.

    Args:
        target: Background image.
        mask: A rough cut of the image source.
        alpha: Alpha mask of the image source.

    Returns:
        dominant_color

    """

    transition_zone = mask - alpha

    r = np.squeeze(target[:, :, 0])
    g = np.squeeze(target[:, :, 1])
    b = np.squeeze(target[:, :, 2])

    dominant_color = np.zeros((3,))
    dominant_color[0] = np.mean(np.mean(r[transition_zone > 0.6]))
    dominant_color[1] = np.mean(np.mean(g[transition_zone > 0.6]))
    dominant_color[2] = np.mean(np.mean(b[transition_zone > 0.6]))

    return dominant_color
