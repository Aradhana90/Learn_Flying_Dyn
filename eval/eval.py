import sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from data.filter_and_diff import get_trajectory, get_trajectories
from metrics import get_rmse, integrate_trajectory
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
training_runs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

test_object = 'benchmark_box'
test_run = 'med_dist'
test_dir = '../data/extracted/' + test_object + '/' + test_run
test_runs = ([16, 17, 18])
# test_runs = ([17])


if __name__ == "__main__":
    # Training data
    x_train, y_train = get_trajectories(training_dir, runs=training_runs, only_pos=only_pos, ang_vel=ang_vel)

    # Test data: One trajectory for whole prediction
    traj1_path = training_dir + '/' + str(test_runs[0]) + '.csv'
    t_test, _, _ = get_trajectory(traj1_path, only_pos=only_pos)
    x_test, y_test = get_trajectories(test_dir, runs=test_runs, only_pos=only_pos, ang_vel=ang_vel)

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
    _, x_int = integrate_trajectory(models, x_test[:, 0], t_eval=t_test, estimator=alg, only_pos=only_pos,
                                    ang_vel=ang_vel)

    # Compute RMSE
    rmse_pos = get_rmse(y_test[0:3], y_pred[0:3])
    print(f"RMSE in position is {rmse_pos:.3f}")
    if not only_pos:
        rmse_ori = get_rmse(y_test[3:7], y_pred[3:7])
        print(f"RMSE in orientation is {rmse_ori:.3f}")

    # Compute deviation of predicted and real endpoint
    dist = la.norm(x_test[0:3, len(t_test) - 1] - x_int[0:3, -1])
    print(f"End position deviation is {dist:.3f}")
    if not only_pos:
        dist_ori = la.norm(x_test[3:7, len(t_test) - 1] - x_int[3:7, -1])
        print(f"End orientation deviation is {dist_ori:.3f}")

    # PLOT #################################################
    # Figure out dimensions
    D_pos, D_vel = 3, 3
    if only_pos:
        D_ori, D_ang_vel = 0, 0
    else:
        D_ori = 4
        if ang_vel:
            D_ang_vel = 3
        else:
            D_ang_vel = 4

    # Plot position, linear velocity and acceleration
    fig, axes = plt.subplots(3, 3)
    for ii in range(3):
        axes[0, ii].plot(t_test, x_test[ii, :len(t_test)], color='b', label='$\\xi_' + str(ii) + '$')
        axes[0, ii].plot(t_test, x_int[ii, :], color='r', label='$\\hat{\\xi}_' + str(ii) + '$')
        axes[1, ii].plot(t_test, x_test[ii + D_pos + D_ori, :len(t_test)], color='b',
                         label='$\\dot{\\xi}_' + str(ii) + '$')
        axes[1, ii].plot(t_test, x_int[ii + D_pos + D_ori, :], color='r', label='$\\dot{\\hat{\\xi}}_' + str(ii) + '$')
        axes[2, ii].plot(t_test, y_test[ii, :len(t_test)], color='b', label='$\\ddot{\\xi}_' + str(ii) + '$')
        axes[2, ii].plot(t_test, y_pred[ii, :len(t_test)], color='r', label='$\\ddot{\\hat{\\xi}}_' + str(ii) + '$')

        axes[0, ii].legend(shadow=True, fancybox=True)
        axes[1, ii].legend(shadow=True, fancybox=True)
        axes[2, ii].legend(shadow=True, fancybox=True)

    # Plot orientation (as quat), angular velocity (as omega or dquat) and angular acceleration (as domega or ddquat)
    if not only_pos:
        fig, axes = plt.subplots(3, 4)

    # Plot orientation
    for ii in range(0, D_ori):
        axes[0, ii].plot(t_test, x_test[ii + 3, :len(t_test)], color='b', label='$q_' + str(ii) + '$')
        axes[0, ii].plot(t_test, x_int[ii + 3, :], color='r', label='$\\hat{q}_' + str(ii) + '$')
        axes[0, ii].legend(shadow=True, fancybox=True)

    # Plot angular velocity and acceleration
    for ii in range(0, D_ang_vel):
        axes[1, ii].plot(t_test, x_test[ii + D_pos + D_vel, :len(t_test)], color='b',
                         label='$\\dot{q}_' + str(ii) + ' \\:or\\: \\omega_' + str(ii) + '$')
        axes[1, ii].plot(t_test, x_int[ii + D_pos + D_vel, :], color='r',
                         label='$\\dot{\\hat{q}}_' + str(ii) + ' \\:or\\: \\hat{\\omega}_' + str(ii) + '$')
        axes[2, ii].plot(t_test, y_test[ii + D_pos, :len(t_test)], color='b',
                         label='$\\ddot{q}_' + str(ii) + ' \\:or\\: \\dot{\\omega}_' + str(ii) + '$')
        axes[2, ii].plot(t_test, y_pred[ii + D_pos, :len(t_test)], color='r',
                         label='$\\ddot{\\hat{q}}_' + str(ii) + ' \\:or\\: \\dot{\\hat{\\omega}}_' + str(ii) + '$')
        axes[1, ii].legend(shadow=True, fancybox=True)
        axes[2, ii].legend(shadow=True, fancybox=True)

    # 3D plot of real and predicted trajectory
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x_test[0, :len(t_test)], x_test[1, :len(t_test)], x_test[2, :len(t_test)], color='b',
               label='$\\mathbf{p}}$')
    ax.scatter(x_int[0, :], x_int[1, :], x_int[2, :], color='r', label='$\\mathbf{\\hat{p}}}$')
    ax.set_xlabel('$p_1$')
    ax.set_ylabel('$p_2$')
    ax.set_zlabel('$p_3$')
    ax.set_title('Predicted and real flying trajectories')
    ax.legend(shadow=True, fancybox=True)
    plt.show()
