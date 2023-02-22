import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from data.data_handler import DataHandler
from data.mechanics import sample_random_states, eul2quat, quat2eul
from eval.functions.metrics import integrate_trajectories_disc

# Get random initial states
horizon_length = 20
Sigma_threshold = 1e-1
pos_error = 0.1
n_samples = 21 ** 2

# Mesh for x-velocities
delta_v = 1
v_x_mesh = np.linspace(1, 4, int(np.sqrt(n_samples)))
v_z_mesh = np.linspace(-5, 2, int(np.sqrt(n_samples)))
X_rand = np.zeros((n_samples, 13))

# Target position
o_des = [1, 0, 0.5]

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

store_path = '../unc_prop_eval_data/Sigma_th_' + str(Sigma_threshold) + '_n_samples_' + str(n_samples) + '/'

obj = 0

# Get all data points
data_path_all = ['../../data/extracted/benchmark_box/small_dist', '../../data/extracted/benchmark_box/med_dist',
                 '../../data/extracted/white_box/small_dist',
                 '../../data/extracted/white_box/med_dist']
n_traj_all = [20, 19, 18, 21]
if obj == 0:
    n_traj = n_traj_all[0:2]
    data_path = data_path_all[0:2]
else:
    n_traj = n_traj_all[2:4]
    data_path = data_path_all[2:4]

dh = DataHandler(dt=0.01, filter_size=7, cont_time=False, rot_to_plane=True)
dh.add_trajectories(data_path[0], np.arange(1, n_traj[0] + 1), 'test')
dh.add_trajectories(data_path[1], np.arange(1, n_traj[1] + 1), 'test')

# Get initial trajectory states from the dataset
X_init_all = np.zeros((13, sum(n_traj)))
for ii in range(n_traj[0]):
    X_init_all[:, ii] = dh.X_test[:, int(np.sum(dh.T_vec[:ii]) - ii)]
for ii in range(n_traj[1]):
    X_init_all[:, ii + n_traj[0]] = dh.X_test[:, int(np.sum(dh.T_vec[:ii + n_traj[0]]) - (ii + n_traj[0]))]

# eul_avg = np.average(dh.X_train[3:7], axis=1)
# Angle: Average euler angles from initial values as quaternions
eul_avg = np.average(quat2eul(X_init_all[3:7]), axis=1)
quat_avg = eul2quat(eul_avg[:, np.newaxis]).reshape(4)
mask = [True, True, True, False, False, False, False, False, True, False, True, True, True]
# Replace quaternion by average from the training data
X_rand[:, 3:7] = np.repeat(quat_avg[np.newaxis, :], n_samples, axis=0)
X_rand[:, mask] = np.average(X_init_all.T[:, mask], axis=0)
# Insert velocity meshgrid
for ii in range(len(v_z_mesh)):
    for jj in range(len(v_x_mesh)):
        X_rand[jj + len(v_x_mesh) * ii, 7] = v_x_mesh[jj]
        X_rand[jj + len(v_x_mesh) * ii, 9] = v_z_mesh[ii]

if __name__ == "__main__":
    model = joblib.load('../cross_valid_models_disc/obj_0/kernel_1/F_30/3/model.sav')
    model.get_kernel_params()
    model.compute_grams()

    # Store number of timesteps until the position uncertainty exceeds a certain threshold
    n_max = np.zeros(n_samples)
    good_initial_state = np.zeros(n_samples)

    for ii in range(n_samples):
        # Integrate trajectories with uncertainty propagation
        X_tmp = np.repeat(X_rand[ii, :].reshape(13, 1), horizon_length, axis=1)
        _, _, Sigma_prop_int, Sigma_int = integrate_trajectories_disc(model, X_tmp, np.array([horizon_length]), unc_prop=True)

        # Position uncertainty
        Sigma_prop_int_pos = np.zeros((3, horizon_length))
        for kk in range(horizon_length):
            Sigma_prop_int_pos[:, kk] = np.diag(Sigma_prop_int[:3, :3, kk])

        # Check how long the position uncertainty stays below the upper threshold
        n_max[ii] = sum(np.linalg.norm(Sigma_prop_int_pos, axis=0) <= Sigma_threshold)
        print('n_max = ' + str(n_max[ii]))

        # Integrate trajectories without uncertainty propagation, but for a longer horizon
        X_int, _, _, _ = integrate_trajectories_disc(model, X_tmp, np.array([75]), unc_prop=False)

        # Check if the desired position can be reached from current initial state
        min_diff = np.min(np.linalg.norm(X_int[0:3].T - o_des, axis=1))
        if min_diff <= pos_error:
            good_initial_state[ii] = 1
        print('State good: ' + str(bool(good_initial_state[ii])) + ', min. difference = ' + str(round(min_diff, 3)) + ' m.')

    if not os.path.exists(store_path):
        os.makedirs(store_path)

    # Store data
    joblib.dump(X_rand, store_path + 'X_rand.sav')
    joblib.dump(n_max, store_path + 'n_max.sav')
    joblib.dump(good_initial_state, store_path + 'good_initial_state.sav')

    # Plot
    mask = good_initial_state.astype(bool)
    plt.scatter(X_rand[mask, 7], X_rand[mask, 9], c=n_max[mask], cmap='viridis', marker="o")
    plt.scatter(X_rand[~mask, 7], X_rand[~mask, 9], c=n_max[~mask], cmap='viridis', marker=".")
    print(n_max)
    plt.show()
