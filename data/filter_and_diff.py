import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from .mechanics import get_ang_vel, quat2eul
from scipy import interpolate

# Specify input directory
which_object = 'benchmark_box'
run = 'med_dist'
path = 'extracted/' + which_object + '/' + run + '/'
pos_only = False


def mean_filt(data, filter_size=5):
    # Perform mean filtering with kernel size k on a 1D array
    kernel = np.ones(filter_size) / filter_size
    data = np.convolve(data, kernel, mode='same')
    return data


def get_trajectory(path_to_csv, delta_t=0.01, filter_size=5, which_filter='mean', only_pos=True, ang_vel=False):
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
            xi[ii, :] = mean_filt(xi[ii, :], filter_size)
        elif which_filter == 'savgol':
            xi[ii, :] = savgol_filter(xi[ii, :], filter_size, 3, mode='nearest')

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
            dxi[ii, :] = mean_filt(dxi[ii, :], filter_size)
        elif which_filter == 'savgol':
            dxi[ii, :] = savgol_filter(dxi[ii, :], filter_size, 3, mode='nearest')

    # Differentiate velocity data
    ddxi = np.diff(dxi) / delta_t

    # Filter acceleration data
    for ii in range(D_vel + D_ang_vel):
        if which_filter == 'mean':
            ddxi[ii, :] = mean_filt(ddxi[ii, :], filter_size)
        elif which_filter == 'savgol':
            ddxi[ii, :] = savgol_filter(ddxi[ii, :], filter_size, 3, mode='nearest')

    # Due to differentiation, dxi and ddxi have smaller length than xi
    t = t[:-2]
    xi = xi[:, :-2]
    dxi = dxi[:, :-1]

    # Due to filtering, the borders have to be cropped
    tmp = 3 * int((filter_size - 1) / 2)
    t = t[0:-2 * tmp]
    xi = xi[:, tmp:-tmp]
    dxi = dxi[:, tmp:-tmp]
    ddxi = ddxi[:, tmp:-tmp]

    x = np.concatenate((xi, dxi), axis=0)

    return t, x, ddxi


def get_trajectories(path_to_dir, runs, delta_t=0.01, filter_size=5, which_filter='mean', only_pos=True, ang_vel=False):
    """
    :param path_to_dir:
    :param runs:
    :param delta_t:
    :param filter_size:
    :param which_filter:
    :param only_pos:
    :param ang_vel:
    :return:                Filtered trajectories
    """
    # Number of trajectories to return
    N_tilde = len(runs)

    T_n = np.zeros(N_tilde)

    # Initialize X and Y
    path_to_file = path_to_dir + '/' + str(runs[0]) + '.csv'
    _, X, Y = get_trajectory(path_to_file, delta_t=delta_t, filter_size=filter_size,
                             which_filter=which_filter, only_pos=only_pos, ang_vel=ang_vel)
    T_n[0] = X.shape[1]
    for ii in range(1, N_tilde):
        training_path = path_to_dir + '/' + str(runs[ii]) + '.csv'
        _, X_tmp, Y_tmp = get_trajectory(training_path, delta_t=delta_t, filter_size=filter_size,
                                         which_filter=which_filter, only_pos=only_pos, ang_vel=ang_vel)
        X = np.concatenate((X, X_tmp), axis=1)
        Y = np.concatenate((Y, Y_tmp), axis=1)
        T_n[ii] = X_tmp.shape[1]

    # Also return euler angles for evaluation purpose
    eul = quat2eul(X[3:7])
    return X, Y, eul, T_n.astype(int)


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
        t1, x1, ddxi1 = get_trajectory(path_tmp, filter_size=5, which_filter='mean', only_pos=pos_only,
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
