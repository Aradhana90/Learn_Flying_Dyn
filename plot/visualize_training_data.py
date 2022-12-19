import numpy as np
import matplotlib.pyplot as plt
from data.filter_and_diff import get_trajectories
from data.mechanics import rotate_trajectories_to_plane, quat2eul

obj = 0

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
xi1, ddxi1, _, T_vec1 = get_trajectories(path1, runs1, only_pos=False, ang_vel=True, filter_size=7, cont_time=True)
xi2, ddxi2, _, T_vec2 = get_trajectories(path2, runs2, only_pos=False, ang_vel=True, filter_size=7, cont_time=True)
xi = np.concatenate((xi1, xi2), axis=1)
ddxi = np.concatenate((ddxi1, ddxi2), axis=1)
T_vec = np.concatenate((T_vec1, T_vec2))

o = xi[0:3]
q = xi[3:7]
v = xi[7:10]
omega = xi[10:13]
dv = ddxi[0:3]
domega = ddxi[3:6]

# Rotate trajectories such that the initial velocity vector lies inside the x-z-plane
traj_rot = rotate_trajectories_to_plane(np.concatenate((xi, ddxi), axis=0), T_vec)
o_rot = traj_rot[0:3]
q_rot = traj_rot[3:7]
eul_rot = quat2eul(q_rot)
v_rot = traj_rot[7:10]
omega_rot = traj_rot[10:13]
dv_rot = traj_rot[13:16]
domega_rot = traj_rot[16:19]

# Plot positions of rotated and non-rotated trajectories
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(o[0], o[1], o[2], color='b', label='$\\mathbf{o}$')
ax.scatter(o_rot[0], o_rot[1], o_rot[2], color='r',
           label='$\\mathbf{o}_{rot}$')
ax.set_xlabel('$o_1$')
ax.set_ylabel('$o_2$')
ax.set_zlabel('$o_3$')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title('Positions of original and rotated flying trajectories')
ax.legend()

# Plot velocities of rotated and non-rotated trajectories
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(v[0], v[1], v[2], color='b', label='$\\mathbf{v}$')
ax.scatter(v_rot[0], v_rot[1], v_rot[2], color='r',
           label='$\\mathbf{v}_{rot}$')
ax.set_xlabel('$v_1$')
ax.set_ylabel('$v_2$')
ax.set_zlabel('$v_3$')
ax.set_title('Velocities of original and rotated flying trajectories')
ax.legend()

# Plot velocities of rotated and non-rotated trajectories
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(dv_rot[0], dv_rot[1], dv_rot[2], color='r',
           label='$\\mathbf{\\dot{v}}_{rot}$')
ax.set_xlabel('$\\dot{v}_1$')
ax.set_ylabel('$\\dot{v}_2$')
ax.set_zlabel('$\\dot{v}_3$')
ax.set_title('Accelerations of the rotated flying trajectories')
ax.legend()

# Plot Euler angles of rotated trajectories
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(eul_rot[0], eul_rot[1], eul_rot[2], color='r')
ax.set_xlabel('$\\phi$')
ax.set_ylabel('$\\theta$')
ax.set_zlabel('$\\psi$')
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_zlim(-180, 180)
ax.set_title('Euler angles of the rotated flying trajectories')

# Plot angular accelerations of rotated trajectories
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(omega_rot[0], omega_rot[1], omega_rot[2], color='r')
ax.set_xlabel('$\\omega_1$')
ax.set_ylabel('$\\omega_2$')
ax.set_zlabel('$\\omega_3$')
ax.set_title('Angular velocities of the rotated flying trajectories')

# Plot angular accelerations of rotated trajectories
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(domega_rot[0], domega_rot[1], domega_rot[2], color='r')
ax.set_xlabel('$\\dot{\\omega}_1$')
ax.set_ylabel('$\\dot{\\omega}_2$')
ax.set_zlabel('$\\dot{\\omega}_3$')
ax.set_title('Angular accelerations of the rotated flying trajectories')

plt.show()
