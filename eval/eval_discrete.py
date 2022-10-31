import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern

from data.data_handler import DataHandler
from learning.gpr.GPR_class import GPRegressor
from metrics import get_rmse, eul_norm, integrate_trajectories_disc

# Specify algorithm and system model
alg = 'gpr'
gpr_kernel = ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel()
ang_vel = True
sys_rep = 'cont'
aug_factor = 0

# Specify which trajectories to use for training and testing
train_object = 'benchmark_box'
train_run = 'med_dist'
training_dir = '../data/extracted/' + train_object + '/' + train_run
training_runs = np.arange(1, 11)

test_object = 'benchmark_box'
test_run = 'med_dist'
test_dir = '../data/extracted/' + test_object + '/' + test_run
test_runs = np.arange(11, 20)
# test_runs = ([17])

filter_size = 5

if __name__ == "__main__":
    # Training data
    dh = DataHandler()
    dh.add_trajectories(training_dir, training_runs, 'train')
    dh.add_trajectories(test_dir, test_runs, 'test')

    print('Created training and test data.')

    # Create GPRegressor object handling the training and prediction
    gpr = GPRegressor(n_features=13, n_targets=13, kernel=gpr_kernel, prior=True, n_restarts=4)

    # Train
    gpr.train(dh.X_train, dh.Y_train)

    # Predict
    Y_pred, Sigma_pred = gpr.predict(dh.X_test)

    # Compute RMSE
    rmse_pos = get_rmse(dh.Y_test[0:3], Y_pred[0:3])
    rmse_ori = get_rmse(dh.Y_test[3:7], Y_pred[3:7])
    rmse_lin = get_rmse(dh.Y_test[7:10], Y_pred[7:10])
    rmse_ang = get_rmse(dh.Y_test[10:13], Y_pred[10:13])
    print(f"RMSE in position in m is {rmse_pos:.3f}.")
    print(f"RMSE in orientation acceleration is {rmse_ori:.3f}.")
    print(f"RMSE in linear velocity in m/s is {rmse_lin:.3f}.")
    print(f"RMSE in angular acceleration in rad/s is {rmse_ang:.3f}.")

    # Predict trajectories with GP
    X_int, Eul_int = integrate_trajectories_disc(gpr, dh.X_test, dh.T_vec)

    # Predict trajectories with projectile model
    X_proj, Eul_proj = integrate_trajectories_disc(gpr, dh.X_test, dh.T_vec, projectile=True)

    # Deviations in final position and orientation
    diff_pos, diff_ori = np.zeros(len(test_runs)), np.zeros(len(test_runs))
    for ii in range(len(dh.T_vec)):
        diff_pos[ii] = la.norm(
            dh.Y_test[0:3, np.sum(dh.T_vec[:ii + 1] - 1) - 1] - X_int[0:3, np.sum(dh.T_vec[:ii + 1]) - 1])
        diff_ori[ii] = eul_norm(dh.Eul_test[:, np.sum(dh.T_vec[:ii + 1] - 1) - 1],
                                Eul_int[:, np.sum(dh.T_vec[:ii + 1]) - 1])
    print("End position deviations in cm are: ", diff_pos * 100)
    print("End euler angle deviations in deg are: ", diff_ori)
    print(f"Mean is {np.mean(diff_pos) * 100:.1f} +- {np.std(diff_pos) * 100:.1f} cm.")
    print(f"Mean is: {np.mean(diff_ori):.1f} +- {np.std(diff_ori):.1f} deg.")

    # 3D plot of real and first predicted trajectory
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    traj_len = dh.T_vec[0]
    ax.scatter(dh.X_test[0, :traj_len - 1], dh.X_test[1, :traj_len - 1], dh.X_test[2, :traj_len - 1], color='b',
               label='$\\mathbf{o}}$')
    ax.scatter(X_int[0, :traj_len], X_int[1, :traj_len], X_int[2, :traj_len], color='r',
               label='$\\mathbf{\\hat{o}}_\\mathrm{GP}$')
    ax.scatter(X_proj[0, :traj_len], X_proj[1, :traj_len], X_proj[2, :traj_len], color='g',
               label='$\\mathbf{\\hat{o}}_\\mathrm{proj}$')
    ax.set_xlabel('$o_1$')
    ax.set_ylabel('$o_2$')
    ax.set_zlabel('$o_3$')
    ax.set_title('Predicted and real flying trajectories')
    ax.legend()
    plt.show()
