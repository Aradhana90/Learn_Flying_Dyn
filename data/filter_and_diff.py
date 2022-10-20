import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import interpolate

# Specify input directory
which_object = 'benchmark_box'
run = 'med_dist'
path = 'extracted/' + which_object + '/' + run + '/'
pos_only = False


def mean_filt(data, kernel_size=5):
    # Perform mean filtering with kernel size k on a 1D array
    kernel = np.ones(kernel_size) / kernel_size
    data = np.convolve(data, kernel, mode='same')
    return data


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
    # for ii in range(n_samples):


def get_trajectory(path_to_csv, delta_t=0.01, kernel_size=5, which_filter='mean', only_pos=True, ang_vel=False):
    df = pd.read_csv(path_to_csv)
    D_pos, D_vel = 3, 3
    if only_pos:
        D_ori, D_ang_vel = 0, 0
        tmp = ['Time', 'pose.position.x', 'pose.position.y', 'pose.position.z', 'pose.orientation.x',
               'pose.orientation.y', 'pose.orientation.z', 'pose.orientation.w']
        xi = np.array([df[tmp[1]], df[tmp[2]], df[tmp[3]]])
    else:
        tmp = ['Time', 'pose.position.x', 'pose.position.y', 'pose.position.z', 'pose.orientation.x',
               'pose.orientation.y', 'pose.orientation.z', 'pose.orientation.w']
        D_ori = 4
        if ang_vel:
            D_ang_vel = 3
        else:
            D_ang_vel = 4
        xi = np.array([df[tmp[1]], df[tmp[2]], df[tmp[3]], df[tmp[4]], df[tmp[5]], df[tmp[6]], df[tmp[7]]])
    # t = np.array([df[tmp[0]]]) # Timestamps sometimes do not match pose data

    # Clip data
    start_idx = np.sum(xi[1] > 0.3)  # y >= 0.3
    end_idx = np.sum(xi[2] > 0.3) - 1  # z >= 0.3
    xi = xi[:, start_idx:end_idx]
    t = np.linspace(0, (end_idx - start_idx - 1) * delta_t, end_idx - start_idx)

    # Start at t=0, p = 0 --> useless since we omit positions anyway
    # xi = xi - xi[:, 0].reshape(3, 1)

    # Filter position data
    for ii in range(D_pos + D_ori):
        if which_filter == 'mean':
            xi[ii, :] = mean_filt(xi[ii, :], kernel_size)
        elif which_filter == 'savgol':
            xi[ii, :] = savgol_filter(xi[ii, :], kernel_size, 3, mode='nearest')

    # Compute linear velocity (and quaternion derivative or angular velocity) by numerical differentiation
    if not ang_vel or only_pos:
        dxi = np.diff(xi) / delta_t
    else:
        dxi = np.zeros((6, xi.shape[1] - 1))
        # Linear velocity
        dxi[0:3] = np.diff(xi[0:3]) / delta_t
        # Angular velocity
        dxi[3:6] = get_ang_vel(xi[3:7], delta_t=delta_t)

    # Filter velocity data
    for ii in range(D_vel + D_ang_vel):
        if which_filter == 'mean':
            dxi[ii, :] = mean_filt(dxi[ii, :], kernel_size)
        elif which_filter == 'savgol':
            dxi[ii, :] = savgol_filter(dxi[ii, :], kernel_size, 3, mode='nearest')

    # Differentiate velocity data
    ddxi = np.diff(dxi) / delta_t

    # Filter acceleration data
    for ii in range(D_vel + D_ang_vel):
        if which_filter == 'mean':
            ddxi[ii, :] = mean_filt(ddxi[ii, :], kernel_size)
        elif which_filter == 'savgol':
            ddxi[ii, :] = savgol_filter(ddxi[ii, :], kernel_size, 3, mode='nearest')

    # Due to differentiation, dxi and ddxi have smaller length than xi
    t = t[:-2]
    xi = xi[:, :-2]
    dxi = dxi[:, :-1]

    # Due to filtering, the borders have to be cropped
    tmp = 3 * int((kernel_size - 1) / 2)
    t = t[0:-2 * tmp]
    xi = xi[:, tmp:-tmp]
    dxi = dxi[:, tmp:-tmp]
    ddxi = ddxi[:, tmp:-tmp]

    x = np.concatenate((xi, dxi), axis=0)

    return t, x, ddxi


def get_trajectories(path_to_dir, runs, delta_t=0.01, kernel_size=5, which_filter='mean', only_pos=True, ang_vel=False):
    N_tilde = len(runs)

    # Initialize X and Y
    path_to_file = path_to_dir + '/' + str(runs[0]) + '.csv'
    _, x_train, y_train = get_trajectory(path_to_file, delta_t=delta_t, kernel_size=kernel_size,
                                         which_filter=which_filter, only_pos=only_pos, ang_vel=ang_vel)
    for ii in range(1, N_tilde):
        training_path = path_to_dir + '/' + str(runs[ii]) + '.csv'
        _, x_train_tmp, y_train_tmp = get_trajectory(training_path, delta_t=delta_t, kernel_size=kernel_size,
                                                     which_filter=which_filter, only_pos=only_pos, ang_vel=ang_vel)
        x_train = np.concatenate((x_train, x_train_tmp), axis=1)
        y_train = np.concatenate((y_train, y_train_tmp), axis=1)

    return x_train, y_train


if __name__ == "__main__":
    # Read data
    idx = 0
    fig1, axes1 = plt.subplots(3, 3)
    fig1.suptitle('$\mathbf{p}, \mathbf{v}=\mathbf{\dot{p}}, \mathbf{a}=\mathbf{\ddot{p}}$')
    fig2, axes2 = plt.subplots(3, 4)
    comp_ang_vel = True
    for _ in range(10, 11):
        path_tmp = path + str(_ + 1) + '.csv'
        # Compare filtering and differentiation approaches
        t1, x1, ddxi1 = get_trajectory(path_tmp, kernel_size=5, which_filter='mean', only_pos=pos_only,
                                       ang_vel=comp_ang_vel)
        # t2, x2, ddxi2 = get_trajectory(path, which_filter='savgol')

        # Plot position, velocity and acceleration
        # fig, axes = plt.subplots(3, 3)
        if pos_only:
            for kk in range(3):
                axes1[0, kk].plot(t1, x1[kk, :], color='r')
                axes1[1, kk].plot(t1, x1[kk + 3, :], color='g')
                axes1[2, kk].plot(t1, ddxi1[kk, :], color='b')

        if not pos_only:
            for kk in range(3):
                axes1[0, kk].plot(t1, x1[kk, :], color='r')
                axes1[1, kk].plot(t1, x1[kk + 7, :], color='g')
                axes1[2, kk].plot(t1, ddxi1[kk, :], color='b')

            # Plot orientation as quaternions
            for kk in range(3, 7):
                axes2[0, kk - 3].plot(t1, x1[kk, :], color='r')

            if comp_ang_vel:
                D_ang_vel = 3
                fig2.suptitle('$\mathbf{q}, \mathbf{\omega}, \mathbf{\dot{\omega}}$')
                # fig2.suptitle('$q, \omega, \dot{\omega}$')
            else:
                D_ang_vel = 4
                fig2.suptitle('$\mathbf{q}, \mathbf{\dot{q}}, \mathbf{\ddot{q}}$')
                # fig2.suptitle('$q, \omega, \dot{\omega}$')

            # Plot angular velocity and acceleration as omega and domega or dquat and ddquat
            for kk in range(3, 3 + D_ang_vel):
                axes2[1, kk - 3].plot(t1, x1[kk + 7, :], color='g')
                axes2[2, kk - 3].plot(t1, ddxi1[kk, :], color='b')

    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    plt.show()
