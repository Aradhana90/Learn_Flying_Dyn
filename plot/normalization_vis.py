import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt
from data.filter_and_diff import get_trajectories
from data.mechanics import rotate_trajectories_to_plane, quat2eul

obj = 1

# Specify which trajectories to use for training and testing
if obj == 1:
    path1 = '../data/extracted/benchmark_box/med_dist'
    runs1 = np.arange(1, 20)
    path2 = '../data/extracted/benchmark_box/small_dist'
    runs2 = np.arange(1, 21)
else:
    path1 = '../data/extracted/white_box/med_dist'
    runs1 = np.arange(1, 22)
    path2 = '../data/extracted/white_box/small_dist'
    runs2 = np.arange(1, 19)

# Get trajectories
xi, ddxi, _, T_vec = get_trajectories(path1, runs1, only_pos=False, ang_vel=True, filter_size=7, cont_time=True)
# xi1, ddxi1, _, T_vec1 = get_trajectories(path1, runs1, only_pos=False, ang_vel=True, filter_size=7, cont_time=True)
# xi2, ddxi2, _, T_vec2 = get_trajectories(path2, runs2, only_pos=False, ang_vel=True, filter_size=7, cont_time=True)
# xi = np.concatenate((xi1, xi2), axis=1)
# ddxi = np.concatenate((ddxi1, ddxi2), axis=1)
# T_vec = np.concatenate((T_vec1, T_vec2))

o = xi[0:3]
q = xi[3:7]
v = xi[7:10]
omega = xi[10:13]
dv = ddxi[0:3]
domega = ddxi[3:6]

# Rotate trajectories such that the initial velocity vector lies inside the x-z-plane
traj_rot = rotate_trajectories_to_plane(np.concatenate((xi, ddxi), axis=0), T_vec, normalize_x=True)
o_rot = traj_rot[0:3]
q_rot = traj_rot[3:7]
eul_rot = quat2eul(q_rot)
v_rot = traj_rot[7:10]
omega_rot = traj_rot[10:13]
dv_rot = traj_rot[13:16]
domega_rot = traj_rot[16:19]

# Rotate trajectories such that the initial velocity vector lies inside the x-z-plane
traj_rot_rand = rotate_trajectories_to_plane(np.concatenate((xi, ddxi), axis=0), T_vec, rand_angle=True)
o_rot_rand = traj_rot_rand[0:3]
q_rot_rand = traj_rot_rand[3:7]
eul_rot_rand = quat2eul(q_rot_rand)
v_rot_rand = traj_rot_rand[7:10]
omega_rot_rand = traj_rot_rand[10:13]
dv_rot_rand = traj_rot_rand[13:16]
domega_rot_rand = traj_rot_rand[16:19]

# Plot positions of rotated and non-rotated trajectories in the x-y-plane
fig, ax = plt.subplots(1, 2)
ax[0].scatter(o_rot_rand[0], o_rot_rand[1], color='b', label='$\\mathbf{o}$')
ax[0].scatter(o_rot[0], o_rot[1], color='r',
              label='$\\mathbf{o}_{rot}$')
ax[1].scatter(o_rot[0], o_rot[1], color='r',
              label='$\\mathbf{o}_{rot}$')
tikzplotlib.save("./tex_files/normalization_o.tex")

# Plot velocities of rotated and non-rotated trajectories
fig, ax = plt.subplots(1, 2)
ax[0].scatter(v_rot_rand[0], v_rot_rand[1], color='b', label='$\\mathbf{v}$')
ax[0].scatter(v_rot[0], v_rot[1], color='r',
              label='$\\mathbf{v}_{rot}$')
ax[1].scatter(v_rot[0], v_rot[1], color='r',
              label='$\\mathbf{v}_{rot}$')
tikzplotlib.save("./tex_files/normalization_v.tex")

# Plot accelerations of rotated and non-rotated trajectories
fig, ax = plt.subplots(1, 2)
ax[0].scatter(dv_rot_rand[0], dv_rot_rand[1], color='b', label='$\\mathbf{\\dot{v}}$')
ax[0].scatter(dv_rot[0], dv_rot[1], color='r',
              label='$\\mathbf{\\dot{v}}_{rot}$')
ax[1].scatter(dv_rot[0], dv_rot[1], color='r',
              label='$\\mathbf{\\dot{v}}_{rot}$')

plt.show()
