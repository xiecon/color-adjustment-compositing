import numpy as np
from src.color.change_basis_matrix import change_basis_matrix
from src.color.compute_dominant_color import compute_dominant_color


def color_correction(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    alpha: np.ndarray,
    membrane: np.ndarray,
    w: list,
) -> np.ndarray:
    """Creates a composition according to three parameters that
    weigh the dominant color present in the surroundings of
    the image source in the image target and their 2 orthonormal vectors.

    Args:
        source: Image that will be pasted in the target image.
        target: Background image.
        mask: A rough cut of the image source.
        alpha: Alpha mask of the image source.
        membrane: Membrane.
        w: A list of the weight for each vector space.

    Returns:
        image composite result

    """

    dominant_color = compute_dominant_color(target, mask, alpha)
    basis_matrix = change_basis_matrix(dominant_color)

    width, height, channels = source.shape
    tmp = np.hstack(
        [
            source[:, :, 0].reshape(-1, 1),
            source[:, :, 1].reshape(-1, 1),
            source[:, :, 2].reshape(-1, 1),
        ]
    )
    tmp = np.dot(tmp, basis_matrix)
    new_source = np.zeros((width, height, channels)).astype(np.float32)
    new_source[:, :, 0] = tmp[:, 0].reshape(width, height)
    new_source[:, :, 1] = tmp[:, 1].reshape(width, height)
    new_source[:, :, 2] = tmp[:, 2].reshape(width, height)

    tmp = np.hstack(
        [
            target[:, :, 0].reshape(-1, 1),
            target[:, :, 1].reshape(-1, 1),
            target[:, :, 2].reshape(-1, 1),
        ]
    )
    tmp = np.dot(tmp, basis_matrix)
    new_target = np.zeros((width, height, channels)).astype(np.float32)
    new_target[:, :, 0] = tmp[:, 0].reshape(width, height)
    new_target[:, :, 1] = tmp[:, 1].reshape(width, height)
    new_target[:, :, 2] = tmp[:, 2].reshape(width, height)

    tmp = np.hstack(
        [
            membrane[:, :, 0].reshape(-1, 1),
            membrane[:, :, 1].reshape(-1, 1),
            membrane[:, :, 2].reshape(-1, 1),
        ]
    )
    tmp = np.dot(tmp, basis_matrix)
    new_membrane = np.zeros((width, height, channels)).astype(np.float32)
    new_membrane[:, :, 0] = tmp[:, 0].reshape(width, height)
    new_membrane[:, :, 1] = tmp[:, 1].reshape(width, height)
    new_membrane[:, :, 2] = tmp[:, 2].reshape(width, height)

    result_new = np.zeros((width, height, channels)).astype(np.float32)
    for i in range(3):
        tar = new_target[:, :, i]
        temp = (1 - w[i] * alpha) * new_membrane[:, :, i] + new_source[:, :, i]
        tar[mask == 1] = temp[mask == 1]
        result_new[:, :, i] = tar

    tmp = np.hstack(
        [
            result_new[:, :, 0].reshape(-1, 1),
            result_new[:, :, 1].reshape(-1, 1),
            result_new[:, :, 2].reshape(-1, 1),
        ]
    )
    tmp = np.dot(tmp, basis_matrix.T)
    result = np.zeros((width, height, channels))
    result[:, :, 0] = tmp[:, 0].reshape(width, height)
    result[:, :, 1] = tmp[:, 1].reshape(width, height)
    result[:, :, 2] = tmp[:, 2].reshape(width, height)

    return result
