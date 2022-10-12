import math

import numpy as np
from scipy.signal import convolve2d


def evalf(a: np.ndarray, h1: np.ndarray, h2: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Evaluate convolution pyramid on input a using filter set specified by: h1, h2, g."""

    h, w = a.shape
    max_level = math.ceil(math.log2(max(h, w)))
    fs = h1.shape[0]

    # Forward transform (analysis)
    pyr = [0] * max_level
    pyr[0] = np.pad(a, ((fs, fs), (fs, fs)), mode="constant", constant_values=0)
    for i in range(1, max_level):
        down = convolve2d(pyr[i - 1], h1, boundary="symm", mode="same")
        down = down[::2, ::2]

        down = np.pad(down, ((fs, fs), (fs, fs)), mode="constant", constant_values=0)
        pyr[i] = down

    # Backward transform (synthesis)
    fpyr = [0] * max_level
    fpyr[max_level - 1] = convolve2d(
        pyr[max_level - 1], g, boundary="symm", mode="same"
    )
    for i in range(max_level - 2, -1, -1):
        rd = fpyr[i + 1]
        rd = rd[fs:-fs, fs:-fs]

        up = np.zeros_like(pyr[i])
        up[::2, ::2] = rd

        fpyr[i] = convolve2d(up, h2, boundary="symm", mode="same") + convolve2d(
            pyr[i], g, boundary="symm", mode="same"
        )

    ahat = fpyr[0]
    ahat = ahat[fs:-fs, fs:-fs]
    return ahat
