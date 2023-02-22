import math

import numpy as np


def sample_random_states(cont=True, n_samples=1000):
    """
    :param cont:
    :param n_samples:
    :return:            randomly samples states of shape (n_samples, 10) for cont=True or (n_samples, 13) for cont=False
    """
    o_rand_x = np.random.uniform(0, 2, size=(n_samples, 1))
    o_rand_y = np.random.uniform(-0.1, 0.1, size=(n_samples, 1))
    o_rand_z = np.random.uniform(0, 2, size=(n_samples, 1))
    o_rand = np.concatenate((o_rand_x, o_rand_y, o_rand_z), axis=1)

    q_rand = np.random.uniform(-1, 1, size=(n_samples, 4))
    for ii in range(n_samples):
        q_rand[ii] = q_rand[ii] / np.linalg.norm(q_rand[ii])

    v_rand_x = np.random.uniform(1, 3, size=(n_samples, 1))
    v_rand_y = np.random.uniform(-0.5, 0.5, size=(n_samples, 1))
    v_rand_z = np.random.uniform(-5, 1, size=(n_samples, 1))
    v_rand = np.concatenate((v_rand_x, v_rand_y, v_rand_z), axis=1)

    omega_rand = np.random.uniform(-10, 10, size=(n_samples, 3))

    x_rand = np.concatenate((o_rand, q_rand, v_rand, omega_rand), axis=1)

    return x_rand


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
                     w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1], dtype=np.float64).reshape(4)


def quatconj(q):
    """
    :param q:   Quaternion q=[w,x,y,z]
    :return:    Conjugate q*=[w,-x,-y,-z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


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


def get_zrot(angle, o=np.zeros(3)):
    """
    :param angle:   Angle around which to rotate the frame
    :param o:
    :return:        Rotation matrix of shape (3, 3) and transformation matrix of shape (4, 4)
    """

    T = np.array([[np.cos(angle), -np.sin(angle), 0, o[0]],
                  [np.sin(angle), np.cos(angle), 0, o[1]],
                  [0, 0, 1, o[2]],
                  [0, 0, 0, 1]], dtype=float).reshape(4, 4)
    R = T[0:3, 0:3]

    return R, T


def rotate_trajectory(X, Y):
    """
    :param X:   Input data of shape (6, n_samples), (13, n_samples) or (14, n_samples)
    :param Y:   Output data of shape (3, n_samples), (6, n_samples) or (7, n_samples)
    :return:    Trajectory represented in a different frame rotated by a random angle around the z-axis
    """

    Xt, Yt = np.empty_like(X), np.empty_like(Y)

    x_dim, n_samples = X.shape
    y_dim, _ = Y.shape

    # New frame centered at o = \in [-1, 1]^3 and rotated about a random angle around the z-axis
    angle = np.random.rand(1) * 2 * np.pi
    # angle = np.pi / 4
    o = np.random.rand(3) * 2 - 1

    # Compute rotation and transformation matrix and quaternion
    R, T = get_zrot(angle, o)
    q = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)], dtype=float)

    for n in range(n_samples):
        tmp = np.append(X[0:3, n], [1]).reshape(4)
        Xt[0:3, n] = (T @ tmp)[0:3]  # position
        if x_dim == 6:
            Xt[3:6, n] = R @ X[3:6, n]  # linear velocity
            Yt[0:3, n] = R @ Y[0:3, n]  # linear acceleration
        elif x_dim == 13:
            Xt[3:7, n] = quatmul(quatconj(q), quatmul(X[3:7, n], q))  # orientation
            Xt[7:10, n] = R @ X[7:10, n]  # linear velocity
            Xt[10:13, n] = R @ X[10:13, n]  # angular velocity
            Yt[0:3, n] = R @ Y[0:3, n]  # linear acceleration
            Yt[3:6, n] = R @ Y[3:6, n]  # angular acceleration
        elif x_dim == 14:
            Xt[3:7, n] = quatmul(quatconj(q), quatmul(X[3:7, n], q))  # orientation
            Xt[7:10, n] = R @ X[7:10, n]  # linear velocity
            Xt[10:14, n] = quatmul(quatconj(q), quatmul(X[10:14, n], q))  # angular velocity
            Yt[0:3, n] = R @ Y[0:3, n]  # linear acceleration
            Yt[3:7, n] = quatmul(quatconj(q, quatmul(Y[3:7, n], q)))  # angular acceleration

    return Xt, Yt


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
        if math.isnan(eul[0, ii]) or math.isnan(eul[1, ii]) or math.isnan(eul[2, ii]):
            print('Euler angle is NaN!')

    return eul


def eul2quat(eul):
    """
        :param eul:    Matrix of Euler angles of shape (4, n_samples)
        :return:       Matrix of quaternions of shape (3, n_samples)
        """
    n_samples = eul.shape[1]
    quat = np.empty((4, n_samples))

    for ii in range(n_samples):
        roll, pitch, yaw = eul[0, ii], eul[1, ii], eul[2, ii]

        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        quat[0, ii] = cr * cp * cy + sr * sp * sy
        quat[1, ii] = sr * cp * cy - cr * sp * sy
        quat[2, ii] = cr * sp * cy + sr * cp * sy
        quat[3, ii] = cr * cp * sy - sr * sp * cy

    return quat


def projectile_motion_disc(x):
    """
    :param x:   Current state of shape (13,)
    :return:    Next state when applying the projectile motion model
    """
    dt = 0.01
    x_next = np.empty_like(x)

    # Position
    x_next[0] = x[0] + x[7] * dt
    x_next[1] = x[1] + x[8] * dt
    x_next[2] = x[2] + x[9] * dt - 0.5 * 9.81 * dt ** 2

    # Orientation: TODO
    x_next[3:7] = x[3:7]

    # Linear velocity
    x_next[7:9] = x[7:9]
    x_next[9] = x[9] - 9.81 * dt

    # Angular velocity
    x_next[10:12] = x[10:12]

    return x_next


def sub_prior(Xt, Yt, sub=True, sys_rep='cont'):
    """
    :param Xt:  Training data of shape (10, N_samples) or (13, N_samples)
    :param Yt:  Training targets of shape (6, N_samples) or (13, N_samples)
    :param sub: True if the mean should be subtracted, False if the mean should be added back
    :return:    Mean-subtracted/added targets of shape (6, N_samples) or (13, N_samples)
    """

    if Xt.ndim == 1:
        X = Xt[:, np.newaxis]
        Y = Yt[:, np.newaxis]
    else:
        X = np.copy(Xt)
        Y = np.copy(Yt)

    n_samples = X.shape[1]
    dt = 0.01

    if sub:
        sign = -1
    else:
        sign = 1

    if sys_rep == 'cont':
        Y_new = np.copy(Y)
        Y_new[2] -= sign * 9.81
    else:
        # Position: o_next = o_cur + v_cur * dt + 0.5 * [0, 0, -9.81] * dt**2
        Y_new = np.copy(Y)
        Y_new[0] += sign * X[0] + sign * X[7] * dt
        Y_new[1] += sign * X[1] + sign * X[8] * dt
        Y_new[2] += sign * X[2] + sign * X[9] * dt - sign * 0.5 * 9.81 * dt ** 2

        # Orientation: q_next = q_cur + 0.5 * omega_cur * q_cur
        tmp = np.empty((4, n_samples))
        for ii in range(n_samples):
            q = X[3:7, ii]
            omega = np.append([0], X[10:13, ii])
            tmp[:, ii] = 0.5 * quatmul(omega, q) * dt

        Y_new[3:7] += sign * X[3:7] + sign * tmp

        # Linear velocity: v_next = v_cur + [0, 0, -9.81 dt]
        Y_new[7] += sign * X[7]
        Y_new[8] += sign * X[8]
        Y_new[9] += sign * X[9] - sign * 9.81 * dt

        # Angular velocity:
        Y_new[10:13] += sign * X[10:13]

    return Y_new


def grad_proj(x, dt=0.01):
    """
    :param X:   Input of shape (13,)
    :return:    Gradient of the projectile motion model with respect to x
    """

    g = np.eye(13)
    g[0, 7] = dt
    g[1, 8] = dt
    g[2, 9] = dt
    g[3, 4] = -x[10] * dt / 2
    g[3, 5] = -x[11] * dt / 2
    g[3, 6] = -x[12] * dt / 2
    g[3, 10] = -x[4] * dt / 2
    g[3, 11] = -x[5] * dt / 2
    g[3, 12] = -x[6] * dt / 2
    g[4, 3] = x[10] * dt / 2
    g[4, 5] = -x[12] * dt / 2
    g[4, 6] = x[11] * dt / 2
    g[4, 10] = x[3] * dt / 2
    g[4, 11] = x[6] * dt / 2
    g[4, 12] = -x[5] * dt / 2
    g[5, 3] = x[11] * dt / 2
    g[5, 4] = x[12] * dt / 2
    g[5, 6] = -x[10] * dt / 2
    g[5, 10] = -x[6] * dt / 2
    g[5, 11] = x[3] * dt / 2
    g[5, 12] = x[4] * dt / 2
    g[6, 3] = x[12] * dt / 2
    g[6, 4] = -x[11] * dt / 2
    g[6, 5] = x[10] * dt / 2
    g[6, 10] = x[5] * dt / 2
    g[6, 11] = -x[4] * dt / 2
    g[6, 12] = x[3] * dt / 2

    return g


def rotate_trajectories_to_plane(x, T_vec, rand_angle=False, normalize_x=False):
    """
    :param x:       States of shape (19, n_samples) consisting of position, orientation, linear and angular velocity and acceleration
    :param T_vec:   Length of the trajectories. Entries must sum up to n_samples
    :return:        States of shape (19, n_samples) for the rotated trajectories such that each has an initial velocity in the x-z-plane
    """
    N_traj = len(T_vec)
    x_rot = np.copy(x)
    for ii in range(N_traj):
        x_tmp = x[:, sum(T_vec[0:ii]): sum(T_vec[0:ii]) + T_vec[ii]]  # States of trajectory ii
        # Initial velocity
        v_init = x_tmp[7:10, 0]
        # Angle between horizontal projection of the initial velocity and the x-z-plane
        if not rand_angle:
            phi = np.arctan2(v_init[1], v_init[0])
        else:
            phi = np.random.rand() * 2 * np.pi
        R, _ = get_zrot(-phi)
        q = np.array([np.cos(-phi / 2), 0, 0, np.sin(-phi / 2)], dtype=float)

        # Rotate positions, linear and angular velocities and linear and angular accelerations
        for kk in range(T_vec[ii]):
            x_rot[0:3, sum(T_vec[0:ii]) + kk] = R @ x_tmp[0:3, kk]  # o
            x_rot[3:7, sum(T_vec[0:ii]) + kk] = quatmul(q, quatmul(x_tmp[3:7, kk], quatconj(q)))
            x_rot[7:10, sum(T_vec[0:ii]) + kk] = R @ x_tmp[7:10, kk]  # v
            x_rot[10:13, sum(T_vec[0:ii]) + kk] = R @ x_tmp[10:13, kk]  # \omega
            x_rot[13:16, sum(T_vec[0:ii]) + kk] = R @ x_tmp[13:16, kk]  # \dot{v}
            x_rot[16:19, sum(T_vec[0:ii]) + kk] = R @ x_tmp[16:19, kk]  # \dot{\omega}

        # Shift trajectory such that the y = 0 for the initial position
        x_rot[1, sum(T_vec[0:ii]): sum(T_vec[0:ii]) + T_vec[ii]] -= x_rot[1, sum(T_vec[0:ii])]

        if normalize_x:
            x_rot[0, sum(T_vec[0:ii]): sum(T_vec[0:ii]) + T_vec[ii]] -= x_rot[0, sum(T_vec[0:ii])]

    return x_rot
