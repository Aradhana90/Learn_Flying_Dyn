import sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import tikzplotlib

from data.filter_and_diff import get_trajectory, get_trajectories
from data.mechanics import quat2eul
from eval.metrics import get_rmse, integrate_trajectories, eul_norm

# Specify algorithm and system model
alg = 'proj'
only_pos = False
ang_vel = True
sys_rep = 'cont'
aug_factor = 0
prior = True
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
test_runs = np.delete(np.arange(1, n_traj + 1), training_runs - 1)
# test_runs = [14]

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

    y_pred = np.zeros((6, x_test.shape[1]))
    y_pred[2] -= 9.81

    # Predict trajectories in the test set
    x_int, eul_int = integrate_trajectories([], x_test=x_test, T_vec=T_test, estimator=alg, only_pos=only_pos,
                                            ang_vel=ang_vel, prior=prior, projectile=True)

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
    # 3D plot of real and predicted trajectory
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x_test[0, :T_test[0]], x_test[1, :T_test[0]], x_test[2, :T_test[0]], color='b',
               label='Measured')
    ax.scatter(x_int[0, :T_test[0]], x_int[1, :T_test[0]], x_int[2, :T_test[0]], color='r',
               label='Projectile model')
    ax.set_xlim(0.5, 1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    # ax.set_title('Predicted and real flying trajectories')
    ax.legend(loc='center right')
    # tikzplotlib.save('drag.tex')
    plt.show()
