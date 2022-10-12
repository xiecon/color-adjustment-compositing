import numpy as np


def rot_vec_around_axis(v, axis, angle_radians):
    c = round(np.cos(angle_radians), 4)
    s = round(np.sin(angle_radians), 4)
    C = round(1 - c, 4)

    rot_matrix = np.array(
        [
            [
                axis[0] * axis[0] * C + c,
                axis[0] * axis[1] * C - axis[2] * s,
                axis[0] * axis[2] * C + axis[1] * s,
            ],
            [
                axis[1] * axis[0] * C + axis[2] * s,
                axis[1] * axis[1] * C + c,
                axis[1] * axis[2] * C - axis[0] * s,
            ],
            [
                axis[2] * axis[0] * C - axis[1] * s,
                axis[2] * axis[1] * C + axis[0] * s,
                axis[2] * axis[2] * C + c,
            ],
        ]
    )
    out_vector = np.dot(rot_matrix, np.expand_dims(v, axis=-1))
    out_vector = np.squeeze(out_vector)
    return out_vector
