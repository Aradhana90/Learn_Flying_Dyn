import numpy as np

from metrics import get_rmse


def get_zrot(angle):
    """
    :param angle:   Angle around which to rotate the frame
    :param o:
    :return:        Rotation matrix of shape (3, 3) and transformation matrix of shape (4, 4)
    """

    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]], dtype=float).reshape(3, 3)

    return R


print(np.random.rand(10, 5))
