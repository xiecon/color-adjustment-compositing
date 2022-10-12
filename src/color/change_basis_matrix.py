import numpy as np
from src.color.rot_vec_around_axis import rot_vec_around_axis


def change_basis_matrix(dominant_color):
    """Change basis matrix."""

    r = np.array([1, 0, 0])
    g = np.array([0, 1, 0])
    b = np.array([0, 0, 1])

    dominant_norm = np.linalg.norm(dominant_color)
    dominant_color = dominant_color / dominant_norm

    rotate_axis = np.cross(r, dominant_color)
    rotate_axis = rotate_axis / np.linalg.norm(rotate_axis)
    angle_rot = np.arccos(np.dot(r, dominant_color))

    rnew = rot_vec_around_axis(r, rotate_axis, angle_rot)
    gnew = rot_vec_around_axis(g, rotate_axis, angle_rot)
    bnew = rot_vec_around_axis(b, rotate_axis, angle_rot)

    matrix = np.vstack(
        [np.expand_dims(rnew, 0), np.expand_dims(gnew, 0), np.expand_dims(bnew, 0)]
    )

    return matrix
