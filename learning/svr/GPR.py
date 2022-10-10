from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from data.filter_and_diff import get_trajectory
from eval.metrics import get_rmse
import numpy as np
import time
import matplotlib.pyplot as plt

# Specify which trajectories to use
which_object = 'benchmark_box'
run = 'small_dist'
training_nr = 6
test_nr = 18
training_path = '../../data/extracted/' + which_object + '/' + run + '/' + str(training_nr) + '.csv'
test_path = '../../data/extracted/' + which_object + '/' + run + '/' + str(test_nr) + '.csv'


def train_gpr(x, y, kernel=RBF()):
    """
        Train a GPR model
        x: Training samples of shape (n_features, n_samples)
        y: Training targets of shape (n_targets, n_samples)
    """
    gpr = GaussianProcessRegressor()
    gpr.fit(x.T, y.T)
    return gpr


def pred_gpr(x, gpr):
    """
        Predict with a GPR model
        x:      Test data of shape (n_features, n_samples)
        return: Predicted targets of shape (n_targets, n_samples)
    """
    y_pred, sigma_pred = gpr.predict(x.T, return_std=True)
    y_pred = y_pred.T
    return y_pred, sigma_pred


if __name__ == "__main__":
    # Training data
    _, x_train, y_train = get_trajectory(training_path)

    # Test data
    t_test, x_test, y_test = get_trajectory(test_path)

    # Train GPR model
    gpr_kernel = RBF()
    gpr_model = train_gpr(x_train[3:], y_train, gpr_kernel)

    # Predict with trained SVR models
    y_pred, sigma_pred = pred_gpr(x_test[3:], gpr_model)

    # Compute RMSE
    err = get_rmse(y_test, y_pred)
    print(f"RMSE: {err:.3f}")

    # Plot real and predicted targets
    fig, axes = plt.subplots(3, 3)
    for kk in range(3):
        axes[0, kk].plot(t_test, x_test[kk, :], label='$\\xi_' + str(kk + 1) + '$')
        axes[1, kk].plot(t_test, x_test[kk + 3, :], label='$\\dot{\\xi}_' + str(kk + 1) + '$')
        axes[2, kk].plot(t_test, y_test[kk, :], label='$\\ddot{\\xi}_' + str(kk + 1) + '$')
        axes[2, kk].plot(t_test, y_pred[kk, :], color='r', label='$\\ddot{\\xi}_{SVR,' + str(kk + 1) + '}$')

        axes[0, kk].legend(shadow=True, fancybox=True)
        axes[1, kk].legend(shadow=True, fancybox=True)
        axes[2, kk].legend(shadow=True, fancybox=True)

    #     figManager = plt.get_current_fig_manager()

    plt.show()
