import joblib
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from data.data_handler import DataHandler
from eval.functions.metrics import integrate_trajectories_disc, integrate_trajectories_cont, eul_norm

# Continuous or discrete time
cont = True

# GP or SVR
gp = True

N_CV = 10
obj = 0
kernel_idx = 9

eval_on = 'test'

F_vec = [10]

data_path = ['../../data/extracted/benchmark_box/small_dist', '../../data/extracted/benchmark_box/med_dist',
             '../../data/extracted/white_box/small_dist',
             '../../data/extracted/white_box/med_dist']
n_traj = [20, 19, 18, 21]
if obj == 0:
    n_traj = n_traj[0:2]
    data_path = data_path[0:2]
else:
    n_traj = n_traj[2:4]
    data_path = data_path[2:4]

diff_pos = np.zeros((len(F_vec), N_CV))
diff_pos_std = np.zeros((len(F_vec), N_CV))
diff_ori = np.zeros((len(F_vec), N_CV))
diff_ori_std = np.zeros((len(F_vec), N_CV))

for ii in range(len(F_vec)):
    F = F_vec[ii]
    for kk in range(N_CV):
        print(kk)
        # Get model
        if gp:
            if cont:
                model_path = '../cross_valid_models_cont/obj_' + str(obj) + '/kernel_' + str(kernel_idx) + '/F_' + str(F) + '/' + str(kk)
            else:
                model_path = '../cross_valid_models_disc/obj_' + str(obj) + '/kernel_' + str(kernel_idx) + '/F_' + str(F) + '/' + str(kk)
        else:
            if cont:
                model_path = '../cross_valid_models_cont_svr/obj_' + str(obj) + '/F_' + str(F) + '/' + str(kk)
            else:
                model_path = '../cross_valid_models_disc_svr/obj_' + str(obj) + '/F_' + str(F) + '/' + str(kk)
        model = joblib.load(model_path + '/model.sav')

        # Get test data
        dh = DataHandler(dt=0.01, filter_size=7, cont_time=cont, rot_to_plane=True)

        training_runs1 = joblib.load(model_path + '/train_traj1.sav')
        training_runs2 = joblib.load(model_path + '/train_traj2.sav')

        # Delete trajectories used for training from the test set
        if eval_on == 'test':
            test_runs1 = np.arange(1, n_traj[0] + 1)
            test_runs2 = np.arange(1, n_traj[1] + 1)
            test_runs1 = np.delete(test_runs1, np.ravel([np.where(test_runs1 == i) for i in training_runs1]))
            test_runs2 = np.delete(test_runs2, np.ravel([np.where(test_runs2 == i) for i in training_runs2]))
        else:
            test_runs1 = training_runs1
            test_runs2 = training_runs2

        print(test_runs1)
        dh.add_trajectories(data_path[0], test_runs1, 'test')
        dh.add_trajectories(data_path[1], test_runs2, 'test')

        # Predict trajectories with GP
        if cont:
            X_int, Eul_int = integrate_trajectories_cont(model, x_test=dh.X_test, T_vec=dh.T_vec)
        else:
            X_int, Eul_int, _, _ = integrate_trajectories_disc(model, dh.X_test, dh.T_vec)

        # Predict trajectories with projectile model
        # X_proj, Eul_proj, _, _ = integrate_trajectories_disc(model, dh.X_test, dh.T_vec, projectile=True)

        # Deviations in final position and orientation
        diff_pos_tmp, diff_ori_tmp = np.zeros(len(dh.T_vec)), np.zeros(len(dh.T_vec))
        for jj in range(len(dh.T_vec)):
            if cont:
                diff_pos_tmp[jj] = np.linalg.norm(dh.X_test[0:3, np.sum(dh.T_vec[:jj + 1]) - 1] - X_int[0:3, np.sum(dh.T_vec[:jj + 1]) - 1])
                diff_ori_tmp[jj] = eul_norm(dh.Eul_test[:, np.sum(dh.T_vec[:jj + 1]) - 1], Eul_int[:, np.sum(dh.T_vec[:jj + 1]) - 1])
            else:
                diff_pos_tmp[jj] = np.linalg.norm(dh.X_test[0:3, np.sum(dh.T_vec[:jj + 1] - 1) - 1] - X_int[0:3, np.sum(dh.T_vec[:jj + 1]) - 1])
                diff_ori_tmp[jj] = eul_norm(dh.Eul_test[:, np.sum(dh.T_vec[:jj + 1] - 1) - 1], Eul_int[:, np.sum(dh.T_vec[:jj + 1]) - 1])

        diff_pos[ii, kk] = np.mean(diff_pos_tmp)
        diff_pos_std[ii, kk] = np.std(diff_pos_tmp)
        diff_ori[ii, kk] = np.mean(diff_ori_tmp)
        diff_ori_std[ii, kk] = np.std(diff_ori_tmp)
        print(f"Mean is {diff_pos[ii, kk] * 100:.1f} +- {diff_pos_std[ii, kk] * 100:.1f} cm.")
        print(f"Mean is: {diff_ori[ii, kk]:.1f} +- {diff_ori_std[ii, kk]:.1f} deg.")

# Plot
fig, axes = plt.subplots(1, 2)
# Standard deviations
print(np.mean(diff_pos, axis=1) * 100)
print(np.std(diff_pos, axis=1) * 100)
print(np.mean(diff_ori, axis=1))
print(np.std(diff_ori, axis=1))
axes[0].plot(F_vec, np.mean(diff_pos, axis=1) * 100, label='$\\mathrm{diff_{pos}}$')
axes[1].plot(F_vec, np.mean(diff_ori, axis=1), label='$\\mathrm{diff_{ori}}$')

# tikzplotlib.save("../../plot/tex_files/disc_obj" + str(obj) + "_svr_traj.tex")

# plt.show()
