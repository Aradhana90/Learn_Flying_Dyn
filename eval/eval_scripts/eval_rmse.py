import joblib
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from data.data_handler import DataHandler
from eval.functions.metrics import get_rmse, get_rmse_eul

# Continuous or discrete time
cont = True

# GP or SVR
gp = True

N_CV = 10
obj = 0
kernel_idx = 1

eval_on = 'train'

F_vec = [2, 4, 6, 8, 10, 14, 18, 22, 26, 30]
# F_vec = [2, 4]

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

if cont:
    rmse_dv = np.zeros((len(F_vec), N_CV))
    rmse_dw = np.zeros((len(F_vec), N_CV))
else:
    rmse_o = np.zeros((len(F_vec), N_CV))
    rmse_q = np.zeros((len(F_vec), N_CV))
    rmse_v = np.zeros((len(F_vec), N_CV))
    rmse_w = np.zeros((len(F_vec), N_CV))

for ii in range(len(F_vec)):
    F = F_vec[ii]
    print(F)
    for kk in range(N_CV):
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

        # Predict
        if gp:
            Y_pred, Sigma_pred = model.predict(dh.X_test)
        else:
            Y_pred = model.predict(dh.X_test)

        # Compute RMSE
        if cont:
            rmse_dv[ii, kk] = get_rmse(dh.Y_test[0:3], Y_pred[0:3])
            rmse_dw[ii, kk] = get_rmse(dh.Y_test[3:6], Y_pred[3:6])
            print(f"RMSE in linear acceleration in m/s^2 is {rmse_dv[ii, kk]:.3f}.")
            print(f"RMSE in angular acceleration in rad/s^2 is {rmse_dw[ii, kk]:.3f}.")
        else:
            rmse_o[ii, kk] = get_rmse(dh.Y_test[0:3], Y_pred[0:3]) * 1e5
            rmse_q[ii, kk] = get_rmse_eul(dh.Y_test[3:7], Y_pred[3:7]) * 100
            rmse_v[ii, kk] = get_rmse(dh.Y_test[7:10], Y_pred[7:10]) * 1000
            rmse_w[ii, kk] = get_rmse(dh.Y_test[10:13], Y_pred[10:13]) * 360 / (2 * np.pi)
            print(f"RMSE in position in m * 10^5 is {rmse_o[ii, kk]:.3f}.")
            print(f"RMSE in linear velocity in m/s * 1000 is {rmse_v[ii, kk]:.3f}.")
            print(f"RMSE in orientation in deg * 10^2 is {rmse_q[ii, kk]:.3f}.")
            print(f"RMSE in angular velocity in deg/s is {rmse_w[ii, kk]:.3f}.")

# Plot
if cont:
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(F_vec, np.mean(rmse_dv, axis=1), label='$\\mathrm{RMSE}_{\\dot{v}}$')
    axes[1].plot(F_vec, np.mean(rmse_dw, axis=1), label='$\\mathrm{RMSE}_{\\dot{\\omega}}$')
else:
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(F_vec, np.mean(rmse_o, axis=1), label='$\\mathrm{RMSE}_{o}$')
    axes[0, 1].plot(F_vec, np.mean(rmse_q, axis=1), label='$\\mathrm{RMSE}_{q}$')
    axes[1, 0].plot(F_vec, np.mean(rmse_v, axis=1), label='$\\mathrm{RMSE}_{v}$')
    axes[1, 1].plot(F_vec, np.mean(rmse_w, axis=1), label='$\\mathrm{RMSE}_{\\omega}$')

tikzplotlib.save("../../plot/tex_files/disc_obj" + str(obj) + "_svr.tex")

plt.show()
