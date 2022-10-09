import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import interpolate

# Specify input directory
which_object = 'white_box'
run = 'med_dist'
path = 'extracted/' + which_object + '/' + run + '/'


def mean_filt(data, kernel_size=5):
    # Perform mean filtering with kernel size k on a 1D array
    kernel = np.ones(kernel_size) / kernel_size
    data = np.convolve(data, kernel, mode='same')
    return data


def get_trajectory(path_to_csv, delta_t=0.01, kernel_size=5, which_filter='mean'):
    df = pd.read_csv(path_to_csv)
    tmp = ['Time', 'pose.position.x', 'pose.position.y', 'pose.position.z', 'pose.orientation.x']
    xi = np.array([df[tmp[1]], df[tmp[2]], df[tmp[3]]])
    # t = np.array([df[tmp[0]]])       # Timestamps sometimes do not match pose data

    # Clip data
    start_idx = np.sum(xi[1] > 0.3)  # y >= 0.3
    end_idx = np.sum(xi[2] > 0.3) - 1  # z >= 0.3
    xi = xi[:, start_idx:end_idx]
    t = np.linspace(0, (end_idx - start_idx - 1) * delta_t, end_idx - start_idx)

    # Start at t=0, p = 0
    xi = xi - xi[:, 0].reshape(3, 1)

    # Filter position data
    for ii in range(3):
        if which_filter == 'mean':
            xi[ii, :] = mean_filt(xi[ii, :], kernel_size)
        elif which_filter == 'savgol':
            xi[ii, :] = savgol_filter(xi[ii, :], kernel_size, 3, mode='nearest')

    # Differentiate position data
    dxi = np.diff(xi) / delta_t

    # Filter velocity data
    for ii in range(3):
        if which_filter == 'mean':
            dxi[ii, :] = mean_filt(dxi[ii, :], kernel_size)
        elif which_filter == 'savgol':
            dxi[ii, :] = savgol_filter(dxi[ii, :], kernel_size, 3, mode='nearest')

    # Differentiate velocity data
    ddxi = np.diff(dxi) / delta_t

    # Filter acceleration data
    for ii in range(3):
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


if __name__ == "__main__":
    # Read data
    idx = 0
    fig, axes = plt.subplots(3, 3)
    for _ in range(18):
        path_to_file = path + str(_ + 1) + '.csv'
        # Compare filtering and differentiation approaches
        t1, x1, ddxi1 = get_trajectory(path_to_file, which_filter='mean')
        # t2, x2, ddxi2 = get_trajectory(path, which_filter='savgol')

        # Plot position, velocity and acceleration
        # fig, axes = plt.subplots(3, 3)
        for kk in range(3):
            axes[0, kk].plot(t1, x1[kk, :], color='r')
            axes[1, kk].plot(t1, x1[kk + 3, :], color='g')
            axes[2, kk].plot(t1, ddxi1[kk, :], color='b')

            # axes[0, kk].plot(t2, x2[kk, :], color='r', linestyle='--')
            # axes[1, kk].plot(t2, x2[kk + 3, :], color='g', linestyle='--')
            # axes[2, kk].plot(t2, ddxi2[kk, :], color='b', linestyle='--')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
