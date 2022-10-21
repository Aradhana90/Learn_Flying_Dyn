import sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

from data.filter_and_diff import get_trajectory, get_trajectories
from data.mechanics import quat2eul
from metrics import get_rmse, integrate_trajectories, eul_norm
from learning.gpr.GPR import train_gpr, pred_gpr
from learning.svr.SVR import train_svr, pred_svr

# Specify algorithm and system model
alg = 'gpr'
only_pos = False
ang_vel = True

# Specify which trajectories to use for training and testing
train_object = 'benchmark_box'
train_run = 'med_dist'
training_dir = '../data/extracted/' + train_object + '/' + train_run
training_runs = np.arange(4, 20)

test_object = 'benchmark_box'
test_run = 'med_dist'
test_dir = '../data/extracted/' + test_object + '/' + test_run
test_runs = [2, 1, 3]
# test_runs = ([17])

filter_size = 5

if __name__ == "__main__":
    # Training data
    x_train, y_train, _, _ = get_trajectories(training_dir, filter_size=filter_size, runs=training_runs,
                                              only_pos=only_pos, ang_vel=ang_vel)

    # Test data: One trajectory for whole prediction
    traj1_path = training_dir + '/' + str(test_runs[0]) + '.csv'
    t_test, _, _ = get_trajectory(traj1_path, filter_size=filter_size, only_pos=only_pos)
    x_test, y_test, eul_test, T_test = get_trajectories(test_dir, filter_size=filter_size, runs=test_runs,
                                                        only_pos=only_pos, ang_vel=ang_vel)

    # Train models and predict acceleration
    if alg == 'svr':
        models = train_svr(x_train[3:], y_train)
        y_pred = pred_svr(x_test[3:], models)
    elif alg == 'gpr':
        models = train_gpr(x_train[3:], y_train, kernel=RBF() + WhiteKernel())
        y_pred, _ = pred_gpr(x_test[3:], models)
    else:
        sys.exit('Select either gpr or svr model')

    # Predict first trajectory in the test set
    x_int, eul_int = integrate_trajectories(models, x_test=x_test, T_vec=T_test, estimator=alg, only_pos=only_pos,
                                            ang_vel=ang_vel)

    # Compute RMSE
    rmse_pos = get_rmse(y_test[0:3], y_pred[0:3])
    print(f"RMSE in linear acceleration in  m/s^2 is {rmse_pos:.3f}.")
    if not only_pos:
        rmse_ori = get_rmse(y_test[3:6], y_pred[3:6])
        print(f"RMSE in angular acceleration in  rad/s^2 is {rmse_ori:.3f}.")

    # Compute deviation of predicted and real endpoint
    dist, dist_ori = np.zeros(len(test_runs)), np.zeros(len(test_runs))
    for ii in range(len(T_test)):
        dist[ii] = la.norm(x_test[0:3, np.sum(T_test[:ii + 1]) - 1] - x_int[0:3, np.sum(T_test[:ii + 1]) - 1])
    print("End position deviations in cm are: ", dist * 100)
    print(f"Mean is {np.mean(dist) * 100:.1f} cm.")

    if not only_pos:
        for ii in range(len(T_test)):
            # dist_ori = la.norm(x_test[3:7, T_test[0] - 1] - x_int[3:7, -1])
            dist_ori[ii] = eul_norm(eul_test[:, np.sum(T_test[:ii + 1]) - 1], eul_int[:, np.sum(T_test[:ii + 1]) - 1])
        print("End euler angle deviations in deg are: ", dist_ori)
        print(f"Mean is: {np.mean(dist_ori):.1f} deg.")

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
