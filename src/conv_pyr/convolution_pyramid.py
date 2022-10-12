import cv2
import numpy as np
from src.conv_pyr.evalf import evalf


def convlution_pyramid(
    source: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Convolution pyramid to compute the Laplacian membranes.

    Args:
        source: Source image.
        target: Target image.
        mask: Mask of source image.

    Returns:
        membrane: Laplacian membranes.

    """

    # Characteristic function: 1 on the boundary, 0 otherwise
    h = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    chi = cv2.filter2D(mask.astype(np.float32), -1, h)
    chi[chi < 0] = 0
    chi[chi > 0] = 1

    erf = target - source

    # Filter set: h1, h2 and g
    w = np.array([0.1507, 0.6836, 1.0334, 0.0270, 0.0312, 0.7753])

    h1 = w[:3]
    h1 = np.hstack((h1, h1[-2::-1]))
    h1 = np.expand_dims(h1, axis=0)
    h1 = h1.T * h1

    h2 = h1 * w[3]

    g = w[4:]
    g = np.hstack((g, g[-2::-1]))
    g = np.expand_dims(g, axis=0)
    g = g.T * g

    membrane = np.zeros_like(source, dtype=np.float32)

    chi_evalf = evalf(chi, h1, h2, g)
    for i in range(3):
        erf_i = erf[:, :, i]
        erf_i[chi == 0] = 0

        # Convolution Pyramid
        erf_i_evalf = evalf(erf_i, h1, h2, g)

        membrane[:, :, i] = np.divide(
            erf_i_evalf, chi_evalf, out=np.zeros_like(erf_i_evalf), where=chi_evalf != 0
        )

    return membrane
