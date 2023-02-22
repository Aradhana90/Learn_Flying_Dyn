from data.filter_and_diff import get_trajectories
import tikzplotlib
import matplotlib.pyplot as plt
import numpy as np

obj = 1

# Specify which trajectories to use for training and testing
if obj == 1:
    path1 = '../data/extracted/benchmark_box/med_dist'
    runs1 = np.arange(1, 10)
    path2 = '../data/extracted/benchmark_box/small_dist'
    runs2 = np.arange(1, 21)
else:
    path1 = '../data/extracted/white_box/med_dist'
    runs1 = np.arange(1, 22)
    path2 = '../data/extracted/white_box/small_dist'
    runs2 = np.arange(1, 19)

xi, ddxi, _, T_vec = get_trajectories(path1, runs1, only_pos=False, ang_vel=True, filter_size=7, cont_time=True, which_filter='mean')
xi2, ddxi2, _, T_vec2 = get_trajectories(path1, runs1, only_pos=False, ang_vel=True, filter_size=7, cont_time=True, which_filter='none')

fig, ax = plt.subplots(1, 4)
idx = 3
for ii in range(len(T_vec)):
    ax[0].plot(np.arange(T_vec[ii]), xi[idx + 0, int(np.sum(T_vec[:ii])):int(np.sum(T_vec[:ii])) + T_vec[ii]], color='r')
    ax[0].plot(np.arange(T_vec[ii]), xi2[idx + 0, int(np.sum(T_vec[:ii])):int(np.sum(T_vec[:ii])) + T_vec[ii]], color='b')
    ax[1].plot(np.arange(T_vec[ii]), xi[idx + 1, int(np.sum(T_vec[:ii])):int(np.sum(T_vec[:ii])) + T_vec[ii]], color='r')
    ax[1].plot(np.arange(T_vec[ii]), xi2[idx + 1, int(np.sum(T_vec[:ii])):int(np.sum(T_vec[:ii])) + T_vec[ii]], color='b')
    ax[2].plot(np.arange(T_vec[ii]), xi[idx + 2, int(np.sum(T_vec[:ii])):int(np.sum(T_vec[:ii])) + T_vec[ii]], color='r')
    ax[2].plot(np.arange(T_vec[ii]), xi2[idx + 2, int(np.sum(T_vec[:ii])):int(np.sum(T_vec[:ii])) + T_vec[ii]], color='b')
    ax[3].plot(np.arange(T_vec[ii]), xi[idx + 3, int(np.sum(T_vec[:ii])):int(np.sum(T_vec[:ii])) + T_vec[ii]], color='r')
    ax[3].plot(np.arange(T_vec[ii]), xi2[idx + 3, int(np.sum(T_vec[:ii])):int(np.sum(T_vec[:ii])) + T_vec[ii]], color='b')

tikzplotlib.save("./tex_files/filtering_q.tex")
plt.show()
