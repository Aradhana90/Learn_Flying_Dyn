import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import interpolate

# Specify input directory
which_object = 'benchmark_box'
topic_name = 'Benchmark_Box_DARKO/pose'
run = 'small_dist'

path = 'extracted/' + which_object + '/' + run

delta_t = 0.01
kernel_size = 5


def mean_filt(x, k):
    # Perform mean filtering with kernel size k on a 1D array
    kernel = np.ones(k) / k
    x = np.convolve(x, kernel, mode='same')
    return x


if __name__ == "__main__":
    # Read data
    idx = 0
    for filename in os.listdir(path):
        df = pd.read_csv(path + '/' + filename)
        tmp = ['Time', 'pose.position.x', 'pose.position.y', 'pose.position.z', 'pose.orientation.x']
        xi = np.array([df[tmp[1]], df[tmp[2]], df[tmp[3]]])
        # timestamps = np.array([df[tmp[0]]])       # Timestamps sometimes do not match pose data!
        timestamps = np.linspace(0, (xi.shape[1] - 1) * delta_t, xi.shape[1])

        # Interpolate position data
        # xi_interp = interpolate.interp1d(timestamps, xi)

        # Clip data
        start_idx = np.sum(xi[1] > 0.3)
        end_idx = np.sum(xi[2] > 0.3) - 1
        xi = xi[:, start_idx:end_idx]
        # xi_interp = xi_interp[:, start_idx:end_idx]
        timestamps = np.linspace(0, (end_idx - start_idx - 1) * delta_t, end_idx - start_idx)

        # Remove datapoints occuring multiple times
        # duplicate_indices = np.array([])
        # for ii in range(1, timestamps.shape[1]-1):
        #     if abs(timestamps[0, ii] - timestamps[0, ii-1]) <= sampling_time/2:
        #         duplicate_indices = np.append(duplicate_indices, ii)
        #
        # timestamps = np.delete(timestamps, duplicate_indices.astype(int), axis=1)
        # xi = np.delete(xi, duplicate_indices.astype(int), axis=1)
        # print(duplicate_indices)

        # Start at t=0, p = 0
        xi = xi - xi[:, 0].reshape(3, 1)

        # Filter position data
        for ii in range(3):
            # xi[ii, :] = savgol_filter(xi[ii, :], 7, 3, mode='nearest')
            xi[ii, :] = mean_filt(xi[ii, :], kernel_size)

        # Differentiate position data
        diff_data = np.diff(xi)
        dxi = diff_data / delta_t

        # Filter velocity data
        for ii in range(3):
            # dxi[ii, :] = savgol_filter(dxi[ii, :], 7, 3, mode='nearest')
            dxi[ii] = mean_filt(dxi[ii], kernel_size)

        # Differentiate velocity data
        ddxi = np.diff(dxi) / delta_t

        idx = idx + 1

        # Cut one, two respectively three values from the beginning and end of the position, velocity and acceleration

        # Plot position, velocity and acceleration in x-direction
        if idx == 5:
            fig, axes = plt.subplots(3, 3)
            for ii in range(3):
                axes[0, ii].plot(timestamps[kernel_size:(-kernel_size - 2)], xi[ii, kernel_size:(-kernel_size - 2)])
                axes[1, ii].plot(timestamps[kernel_size:(-kernel_size - 2)], dxi[ii, kernel_size:(-kernel_size - 1)])
                axes[2, ii].plot(timestamps[kernel_size:(-kernel_size - 2)], ddxi[ii, kernel_size:-kernel_size])
            plt.show()
