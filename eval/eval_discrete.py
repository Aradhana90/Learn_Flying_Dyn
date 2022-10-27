import sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel

from data.filter_and_diff import get_trajectory, get_trajectories
from learning.gpr.GPR_class import GPRegressor
from data.data_handler import DataHandler
from data.mechanics import quat2eul
from metrics import get_rmse, integrate_trajectories, eul_norm, integrate_trajectories_disc
from learning.gpr.GPR import train_gpr, pred_gpr
from learning.svr.SVR import train_svr, pred_svr

# Specify algorithm and system model
alg = 'gpr'
gpr_kernel = RBF() + Matern()
only_pos = False
ang_vel = True
sys_rep = 'cont'
aug_factor = 0

# Specify which trajectories to use for training and testing
train_object = 'benchmark_box'
train_run = 'small_dist'
training_dir = '../data/extracted/' + train_object + '/' + train_run
training_runs = np.arange(1, 11)

test_object = 'benchmark_box'
test_run = 'small_dist'
test_dir = '../data/extracted/' + test_object + '/' + test_run
test_runs = np.arange(11, 21)
# test_runs = ([17])

filter_size = 5

if __name__ == "__main__":
    # Training data
    dh = DataHandler()
    dh.add_trajectories(training_dir, training_runs, 'train')
    dh.add_trajectories(test_dir, test_runs, 'test')

    print('Created training and test data.')

    # Create GPRegressor object handling the training and prediction
    gpr = GPRegressor(n_features=13, n_targets=13, kernel=gpr_kernel)

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

    # Predict trajectories
    # X_int, Eul_int = integrate_trajectories_disc(gpr, dh.X_test, dh.T_vec)
