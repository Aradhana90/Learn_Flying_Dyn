import sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from data.filter_and_diff import get_trajectory, get_trajectories
from metrics import get_rmse, integrate_trajectories_old, eul_norm
from trash.GPR import train_gpr, pred_gpr
from trash.SVR import train_svr, pred_svr

# Specify algorithm and system model
alg = 'proj'
only_pos = False
ang_vel = True
sys_rep = 'cont'
aug_factor = 0
prior = True
kernel_indices = [0]
dset = 3

if dset == 0:
    obj, run, n_traj = 'benchmark_box', 'small_dist', 20
elif dset == 1:
    obj, run, n_traj = 'benchmark_box', 'med_dist', 19
elif dset == 2:
    obj, run, n_traj = 'white_box', 'small_dist', 18
else:
    obj, run, n_traj = 'white_box', 'med_dist', 21

# Specify which trajectories to use for training and testing
training_dir = '../data/extracted/' + obj + '/' + run
training_runs = np.arange(1, 11)
test_dir = '../data/extracted/' + obj + '/' + run
# test_runs = np.delete(np.arange(1, n_traj + 1), training_runs - 1)
test_runs = [11]

filter_size = 7

if __name__ == "__main__":
    # Training data
    x_train, y_train, _, _ = get_trajectories(training_dir, filter_size=filter_size, runs=training_runs,
                                              only_pos=only_pos, ang_vel=ang_vel, aug_factor=aug_factor)

    # Test data: One trajectory for whole prediction
    traj1_path = training_dir + '/' + str(test_runs[0]) + '.csv'
    t_test, _, _ = get_trajectory(traj1_path, filter_size=filter_size, only_pos=only_pos)
    x_test, y_test, eul_test, T_test = get_trajectories(test_dir, filter_size=filter_size, runs=test_runs,
                                                        only_pos=only_pos, ang_vel=ang_vel)

    print('Created training and test data.')
    # Train models and predict acceleration
    for k in range(len(kernel_indices)):
        models = []
        if alg == 'svr':
            models = train_svr(x_train[3:], y_train)
            y_pred = pred_svr(x_test[3:], models)
        elif alg == 'gpr':
            models = train_gpr(x_train[3:], y_train, prior=prior, kernel_idx=kernel_indices[k])
            y_pred, _ = pred_gpr(x_test[3:], models, prior=prior)
        elif alg == 'proj':
            y_pred = np.zeros((6, x_test.shape[1]))
            y_pred[2] -= 9.81
        else:
            sys.exit('Select either gpr or svr model')

        # Predict trajectories in the test set
        x_int, eul_int = integrate_trajectories_old(models, x_test=x_test, T_vec=T_test, estimator=alg, only_pos=only_pos,
                                                    ang_vel=ang_vel, prior=prior)

        print(kernel_indices[k])
        # Compute RMSE
        rmse_pos = get_rmse(y_test[0:3], y_pred[0:3])
        print(f"RMSE in linear acceleration in m/s^2 is {rmse_pos:.3f}.")
        if not only_pos:
            rmse_ori = get_rmse(y_test[3:6], y_pred[3:6])
            print(f"RMSE in angular acceleration in  rad/s^2 is {rmse_ori:.3f}.")

        # Compute deviation of predicted and real endpoint
        diff_pos, diff_ori = np.zeros(len(test_runs)), np.zeros(len(test_runs))
        for ii in range(len(T_test)):
            diff_pos[ii] = la.norm(x_test[0:3, np.sum(T_test[:ii + 1]) - 1] - x_int[0:3, np.sum(T_test[:ii + 1]) - 1])
        print("End position deviations in cm are: ", diff_pos * 100)
        print(f"Mean is {np.mean(diff_pos) * 100:.1f} +- {np.std(diff_pos) * 100:.1f} cm.")

        if not only_pos:
            for ii in range(len(T_test)):
                # dist_ori = la.norm(x_test[3:7, T_test[0] - 1] - x_int[3:7, -1])
                diff_ori[ii] = eul_norm(eul_test[:, np.sum(T_test[:ii + 1]) - 1], eul_int[:, np.sum(T_test[:ii + 1]) - 1])
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
        axes[0, ii].plot(t_test, x_test[ii, :T_test[0]], color='b', label='$o_' + str(ii) + '$')
        axes[0, ii].plot(t_test, x_int[ii, :T_test[0]], color='r', label='$\\hat{o}_' + str(ii) + '$')
        axes[1, ii].plot(t_test, x_test[ii + D_pos + D_ori, :T_test[0]], color='b',
                         label='$v_' + str(ii) + '$')
        axes[1, ii].plot(t_test, x_int[ii + D_pos + D_ori, :T_test[0]], color='r', label='$\\hat{v}_' + str(ii) + '$')
        axes[2, ii].plot(t_test, y_test[ii, :T_test[0]], color='b', label='$\\dot{v}_' + str(ii) + '$')
        axes[2, ii].plot(t_test, y_pred[ii, :T_test[0]], color='r', label='$\\dot{\\hat{v}}_' + str(ii) + '$')

        axes[0, ii].legend()
        axes[1, ii].legend()
        axes[2, ii].legend()

    # Plot orientation (as quat), angular velocity (as omega or dquat) and angular acceleration (as domega or ddquat)
    if not only_pos:
        fig, axes = plt.subplots(3, 4)

        # # Plot orientation as quaternions
        # for ii in range(0, D_ori):
        #     axes[0, ii].plot(t_test, x_test[ii + 3, :T_test[0]], color='b', label='$q_' + str(ii) + '$')
        #     axes[0, ii].plot(t_test, x_int[ii + 3, :], color='r', label='$\\hat{q}_' + str(ii) + '$')
        #     axes[0, ii].legend(shadow=True, fancybox=True)

        # Plot orientation as euler angles
        axes[0, 0].plot(t_test, eul_test[0, :T_test[0]], color='b', label='$\\phi$')
        axes[0, 0].plot(t_test, eul_int[0, :T_test[0]], color='r', label='$\\hat{\\phi}$')
        axes[0, 1].plot(t_test, eul_test[1, :T_test[0]], color='b', label='$\\theta$')
        axes[0, 1].plot(t_test, eul_int[1, :T_test[0]], color='r', label='$\\hat{\\theta}$')
        axes[0, 2].plot(t_test, eul_test[2, :T_test[0]], color='b', label='$\\psi$')
        axes[0, 2].plot(t_test, eul_int[2, :T_test[0]], color='r', label='$\\hat{\\psi}$')
        for ii in range(0, 3):
            axes[0, ii].legend()

    # Plot angular velocity and acceleration
    for ii in range(0, D_ang_vel):
        axes[1, ii].plot(t_test, x_test[ii + D_pos + D_vel, :T_test[0]], color='b',
                         label='$\\omega_' + str(ii) + ' \\:or\\: \\dot{q}_' + str(ii) + '$')
        axes[1, ii].plot(t_test, x_int[ii + D_pos + D_vel, :T_test[0]], color='r',
                         label='$\\hat{\\omega}_' + str(ii) + ' \\:or\\: \\dot{\\hat{q}}_' + str(ii) + '$')
        axes[2, ii].plot(t_test, y_test[ii + D_pos, :T_test[0]], color='b',
                         label='$\\dot{\\omega}_' + str(ii) + ' \\:or\\: \\ddot{q}_' + str(ii) + '$')
        axes[2, ii].plot(t_test, y_pred[ii + D_pos, :T_test[0]], color='r',
                         label='$\\dot{\\hat{\\omega}}_' + str(ii) + ' \\:or\\: \\ddot{\\hat{q}}_' + str(ii) + '$')
        axes[1, ii].legend()
        axes[2, ii].legend()

    # 3D plot of real and predicted trajectory
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x_test[0, :T_test[0]], x_test[1, :T_test[0]], x_test[2, :T_test[0]], color='b',
               label='$\\mathbf{p}}$')
    ax.scatter(x_int[0, :T_test[0]], x_int[1, :T_test[0]], x_int[2, :T_test[0]], color='r',
               label='$\\mathbf{\\hat{p}}}$')
    ax.set_xlabel('$p_1$')
    ax.set_ylabel('$p_2$')
    ax.set_zlabel('$p_3$')
    ax.set_title('Predicted and real flying trajectories')
    ax.legend()
    plt.show()
