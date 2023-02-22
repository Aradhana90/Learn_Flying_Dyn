import joblib
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from data.data_handler import DataHandler
from eval.functions.metrics import integrate_trajectories_disc
from data.mechanics import quat2eul, eul2quat

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

obj = 0
horizon_length = 25

X_rand = np.zeros(13)
data_path_all = ['../data/extracted/benchmark_box/small_dist', '../data/extracted/benchmark_box/med_dist',
                 '../data/extracted/white_box/small_dist',
                 '../data/extracted/white_box/med_dist']
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
mask = [True, True, True, False, False, False, False, True, True, True, True, True, True]
# Replace quaternion by average from the training data
X_rand[3:7] = quat_avg
X_rand[mask] = np.average(X_init_all.T[:, mask], axis=0)
X_rand[7] = 2.5
X_rand[9] = -1

if __name__ == "__main__":
    model = joblib.load('../eval/cross_valid_models_disc/obj_0/kernel_1/F_6/0/model.sav')
    traj1 = joblib.load('../eval/cross_valid_models_disc/obj_0/kernel_1/F_6/0/train_traj1.sav')
    print(traj1)
    # model = joblib.load('./gp_models/obj_0_runs_1to4_kernel_0.sav')
    model.get_kernel_params()
    model.compute_grams()

    # Integrate trajectories with uncertainty propagation
    X_tmp = np.repeat(X_rand.reshape(13, 1), horizon_length, axis=1)
    X_int, Eul_int, Sigma_prop_int, Sigma_int = integrate_trajectories_disc(model, X_tmp, np.array([horizon_length]), unc_prop=True)

    # Store uncertainties at the end of the trajectory
    Sigma_prop_final = Sigma_prop_int[:, :, -1].diagonal()

    # PLOT
    traj_len = horizon_length

    # Plot individual states with uncertainty tube around them
    fig, axes = plt.subplots(2, 4)
    # Position
    for ii in range(3):
        axes[0, ii].plot(np.arange(traj_len), X_int[ii], 'k-')
        axes[0, ii].fill_between(np.arange(traj_len), X_int[ii] - Sigma_int[ii], X_int[ii] + Sigma_int[ii], color='r', alpha=0.2)
        axes[0, ii].fill_between(np.arange(traj_len), X_int[ii] - Sigma_prop_int[ii, ii], X_int[ii] + Sigma_prop_int[ii, ii], color='b',
                                 alpha=0.2)
        axes[0, ii].set_ylabel('$o_' + str(ii + 1) + '$')

    # Orientation
    for ii in range(3, 7):
        axes[1, ii - 3].plot(np.arange(traj_len), X_int[ii], 'k-')
        axes[1, ii - 3].fill_between(np.arange(traj_len), X_int[ii] - Sigma_int[ii], X_int[ii] + Sigma_int[ii], color='r', alpha=0.2)
        axes[1, ii - 3].fill_between(np.arange(traj_len), X_int[ii] - Sigma_prop_int[ii, ii], X_int[ii] + Sigma_prop_int[ii, ii], color='b',
                                     alpha=0.2)
        axes[1, ii - 3].set_ylabel('$q_' + str(ii - 2) + '$')

    tikzplotlib.save("./tex_files/unc_prop_oq.tex")

    fig, axes = plt.subplots(2, 4)
    # Linear velocity
    for ii in range(7, 10):
        axes[0, ii - 7].plot(np.arange(traj_len), X_int[ii], 'k-')
        axes[0, ii - 7].fill_between(np.arange(traj_len), X_int[ii] - Sigma_int[ii], X_int[ii] + Sigma_int[ii], color='r', alpha=0.2)
        axes[0, ii - 7].fill_between(np.arange(traj_len), X_int[ii] - Sigma_prop_int[ii, ii], X_int[ii] + Sigma_prop_int[ii, ii], color='b',
                                     alpha=0.2)
        axes[0, ii - 7].set_ylabel('$v_' + str(ii - 6) + '$')

    # Angular velocity
    for ii in range(10, 13):
        axes[1, ii - 10].plot(np.arange(traj_len), X_int[ii], 'k-')
        axes[1, ii - 10].fill_between(np.arange(traj_len), X_int[ii] - Sigma_int[ii], X_int[ii] + Sigma_int[ii], color='r', alpha=0.2)
        axes[1, ii - 10].fill_between(np.arange(traj_len), X_int[ii] - Sigma_prop_int[ii, ii], X_int[ii] + Sigma_prop_int[ii, ii], color='b',
                                      alpha=0.2)
        axes[1, ii - 10].set_ylabel('$\\omega_' + str(ii - 9) + '$')

    tikzplotlib.save("./tex_files/unc_prop_vw.tex")

    print(Sigma_prop_final)
    plt.show()
