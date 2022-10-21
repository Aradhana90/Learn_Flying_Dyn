import numpy as np


def quat2mat(q):
    """
    :param q:   Quaternion of shape (4,)
    :return:    3 x 4 matrix Q that can be multiplied with dq to obtain omega. See Challis, 2020, eq. (58)
    """
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    Q = np.array([[-q1, -q0, -q3, q2],
                  [-q2, q3, q0, -q1],
                  [-q3, -q2, q1, q0]])
    return Q


def quatmul(q0, q1):
    """
    :param q0:  Quaternion q0 = [w0, x0, y0, z0]
    :param q1:  Quaternion q1 = [w1, x1, y1, z1]
    :return:    Quaternion product of q0 and q1
    """
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1

    return np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                     w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                     w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                     w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1], dtype=np.float64)


def quatconj(q):
    """
    :param q:   Quaternion q=[w,x,y,z]
    :return:    Conjugate q*=[w,-x,-y,-z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def get_ang_vel(quat, delta_t=0.01):
    """
        :param quat:    Matrix of quaternions of shape (4, n_samples)
        :param delta_t: Sampling time
        :return:        Matrix of euler angles of shape (3, n_samples)
    """
    n_samples = quat.shape[1]
    omega = np.zeros((3, n_samples - 1))
    dquat = np.diff(quat) / delta_t
    for ii in range(n_samples - 1):
        # omega[:, ii] = 2 * quat2mat(quat[:, ii]) @ dquat[:, ii]
        tmp = 2 * quatmul(dquat[:, ii], quatconj(quat[:, ii]))
        omega[:, ii] = tmp[1:4]

    return omega


def quat2eul(quat):
    """
    :param quat:    Matrix of quaternions of shape (4, n_samples)
    :return:        Matrix of angular velocities of shape (3, n_samples)
    """
    n_samples = quat.shape[1]
    eul = np.empty((3, n_samples))
    for ii in range(n_samples):
        q0, q1, q2, q3 = quat[0, ii], quat[1, ii], quat[2, ii], quat[3, ii]
        phi = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * q1 ** 2 + q2 ** 2)
        theta = np.arcsin(2 * (q0 * q2 - q1 * q3))
        psi = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
        eul[:, ii] = np.array([phi, theta, psi]) * 180 / np.pi

    return eul
