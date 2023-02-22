import sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from data.data_handler import DataHandler
from eval.functions.metrics import get_rmse, integrate_trajectories_cont, eul_norm
from learning.gpr.GPR_cont import GPRegressor
from learning.svr.SVR_cont import SVRegressor

# Specify algorithm and system model
alg = 'gpr'
only_pos = False
ang_vel = True
prior = True
kernel_indices = [1]
obj = 0
path = ['../data/extracted/benchmark_box/small_dist', '../data/extracted/benchmark_box/med_dist', '../data/extracted/white_box/small_dist',
        '../data/extracted/white_box/med_dist']
n_traj = [20, 19, 18, 21]

# Specify which trajectories to use for training and testing
training_runs = np.arange(1, 11)
test_runs = []
for ii in range(4):
    test_runs.append(np.delete(np.arange(1, n_traj[ii] + 1), training_runs - 1))

filter_size = 7

if __name__ == "__main__":
    # Training data
    dh = DataHandler(dt=0.01, filter_size=7, cont_time=True, rot_to_plane=True)
    if obj == 0:
        dh.add_trajectories(path[0], training_runs, 'train')
        dh.add_trajectories(path[0], test_runs[0], 'test')
        dh.add_trajectories(path[1], training_runs, 'train')
        dh.add_trajectories(path[1], test_runs[1], 'test')
    else:
        dh.add_trajectories(path[2], training_runs, 'train')
        dh.add_trajectories(path[2], test_runs[2], 'test')
        dh.add_trajectories(path[3], training_runs, 'train')
        dh.add_trajectories(path[3], test_runs[3], 'test')

    print('Created training and test data.')
    # Train models and predict acceleration
    for k in range(len(kernel_indices)):
        models = []
        if alg == 'svr':
            model = SVRegressor(n_features=10, n_targets=6, kernel='rbf')
        elif alg == 'gpr':
            model = GPRegressor(n_features=10, n_targets=6, prior=True, kernel_idx=kernel_indices[k], n_restarts=4)

        if alg == 'proj':
            y_pred = np.zeros((6, dh.X_test.shape[1]))
            y_pred[2] -= 9.81
        elif alg == 'svr' or alg == 'gpr':
            model.train(dh.X_train, dh.Y_train)
            y_pred, _ = model.predict(dh.X_test)
        else:
            sys.exit('Model type unknown.')

        # Predict trajectories in the test set
        x_int, eul_int = integrate_trajectories_cont(model, x_test=dh.X_test, T_vec=dh.T_vec)

        print(kernel_indices[k])
        # Compute RMSE
        rmse_pos = get_rmse(dh.Y_test[0:3], y_pred[0:3])
        print(f"RMSE in linear acceleration in m/s^2 is {rmse_pos:.3f}.")
        if not only_pos:
            rmse_ori = get_rmse(dh.Y_test[3:6], y_pred[3:6])
            print(f"RMSE in angular acceleration in  rad/s^2 is {rmse_ori:.3f}.")

        # Compute deviation of predicted and real endpoint
        diff_pos, diff_ori = np.zeros(len(dh.T_vec)), np.zeros(len(dh.T_vec))
        for ii in range(len(dh.T_vec)):
            diff_pos[ii] = la.norm(dh.X_test[0:3, np.sum(dh.T_vec[:ii + 1]) - 1] - x_int[0:3, np.sum(dh.T_vec[:ii + 1]) - 1])
        print("End position deviations in cm are: ", diff_pos * 100)
        print(f"Mean is {np.mean(diff_pos) * 100:.1f} +- {np.std(diff_pos) * 100:.1f} cm.")

        if not only_pos:
            for ii in range(len(dh.T_vec)):
                # dist_ori = la.norm(dh.X_test[3:7, T_test[0] - 1] - x_int[3:7, -1])
                diff_ori[ii] = eul_norm(dh.Eul_test[:, np.sum(dh.T_vec[:ii + 1]) - 1], eul_int[:, np.sum(dh.T_vec[:ii + 1]) - 1])
            print("End euler angle deviations in deg are: ", diff_ori)
            print(f"Mean is: {np.mean(diff_ori):.1f} +- {np.std(diff_ori):.1f} deg.")

    # PLOT #############################################################################################################
    # Figure out dimensions
    D_pos, D_vel, D_ori, D_ang_vel = 3, 3, 0, 0
    if not only_pos:
        D_ori = 4
        if ang_vel:
            D_ang_vel = 3
        else:
            D_ang_vel = 4

    # Plot position, linear velocity and acceleration
    fig, axes = plt.subplots(3, 3)
    for ii in range(3):
        axes[0, ii].plot(range(dh.T_vec[0]), dh.X_test[ii, :dh.T_vec[0]], color='b', label='$o_' + str(ii) + '$')
        axes[0, ii].plot(range(dh.T_vec[0]), x_int[ii, :dh.T_vec[0]], color='r', label='$\\hat{o}_' + str(ii) + '$')
        axes[1, ii].plot(range(dh.T_vec[0]), dh.X_test[ii + D_pos + D_ori, :dh.T_vec[0]], color='b',
                         label='$v_' + str(ii) + '$')
        axes[1, ii].plot(range(dh.T_vec[0]), x_int[ii + D_pos + D_ori, :dh.T_vec[0]], color='r', label='$\\hat{v}_' + str(ii) + '$')
        axes[2, ii].plot(range(dh.T_vec[0]), dh.Y_test[ii, :dh.T_vec[0]], color='b', label='$\\dot{v}_' + str(ii) + '$')
        axes[2, ii].plot(range(dh.T_vec[0]), y_pred[ii, :dh.T_vec[0]], color='r', label='$\\dot{\\hat{v}}_' + str(ii) + '$')

        axes[0, ii].legend()
        axes[1, ii].legend()
        axes[2, ii].legend()

    # Plot orientation (as quat), angular velocity (as omega or dquat) and angular acceleration (as domega or ddquat)
    if not only_pos:
        fig, axes = plt.subplots(3, 4)

        # # Plot orientation as quaternions
        # for ii in range(0, D_ori):
        #     axes[0, ii].plot(range(dh.T_vec[0]), dh.X_test[ii + 3, dh.T_vec[0]], color='b', label='$q_' + str(ii) + '$')
        #     axes[0, ii].plot(range(dh.T_vec[0]), x_int[ii + 3, :], color='r', label='$\\hat{q}_' + str(ii) + '$')
        #     axes[0, ii].legend(shadow=True, fancybox=True)

        # Plot orientation as euler angles
        axes[0, 0].plot(range(dh.T_vec[0]), dh.Eul_test[0, :dh.T_vec[0]], color='b', label='$\\phi$')
        axes[0, 0].plot(range(dh.T_vec[0]), eul_int[0, :dh.T_vec[0]], color='r', label='$\\hat{\\phi}$')
        axes[0, 1].plot(range(dh.T_vec[0]), dh.Eul_test[1, :dh.T_vec[0]], color='b', label='$\\theta$')
        axes[0, 1].plot(range(dh.T_vec[0]), eul_int[1, :dh.T_vec[0]], color='r', label='$\\hat{\\theta}$')
        axes[0, 2].plot(range(dh.T_vec[0]), dh.Eul_test[2, :dh.T_vec[0]], color='b', label='$\\psi$')
        axes[0, 2].plot(range(dh.T_vec[0]), eul_int[2, :dh.T_vec[0]], color='r', label='$\\hat{\\psi}$')
        for ii in range(0, 3):
            axes[0, ii].legend()

    # Plot angular velocity and acceleration
    for ii in range(0, D_ang_vel):
        axes[1, ii].plot(range(dh.T_vec[0]), dh.X_test[ii + D_pos + D_vel, :dh.T_vec[0]], color='b',
                         label='$\\omega_' + str(ii) + ' \\:or\\: \\dot{q}_' + str(ii) + '$')
        axes[1, ii].plot(range(dh.T_vec[0]), x_int[ii + D_pos + D_vel, :dh.T_vec[0]], color='r',
                         label='$\\hat{\\omega}_' + str(ii) + ' \\:or\\: \\dot{\\hat{q}}_' + str(ii) + '$')
        axes[2, ii].plot(range(dh.T_vec[0]), dh.Y_test[ii + D_pos, :dh.T_vec[0]], color='b',
                         label='$\\dot{\\omega}_' + str(ii) + ' \\:or\\: \\ddot{q}_' + str(ii) + '$')
        axes[2, ii].plot(range(dh.T_vec[0]), y_pred[ii + D_pos, :dh.T_vec[0]], color='r',
                         label='$\\dot{\\hat{\\omega}}_' + str(ii) + ' \\:or\\: \\ddot{\\hat{q}}_' + str(ii) + '$')
        axes[1, ii].legend()
        axes[2, ii].legend()

    # 3D plot of real and predicted trajectory
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(dh.X_test[0, :dh.T_vec[0]], dh.X_test[1, :dh.T_vec[0]], dh.X_test[2, :dh.T_vec[0]], color='b',
               label='$\\mathbf{p}}$')
    ax.scatter(x_int[0, :dh.T_vec[0]], x_int[1, :dh.T_vec[0]], x_int[2, :dh.T_vec[0]], color='r',
               label='$\\mathbf{\\hat{p}}}$')
    ax.set_xlabel('$o_1$')
    ax.set_ylabel('$o_2$')
    ax.set_zlabel('$o_3$')
    ax.set_ylim(-0.25, 0.25)
    ax.set_title('Predicted and real flying trajectories')
    ax.legend()
    plt.show()
