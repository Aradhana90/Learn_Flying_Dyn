import matplotlib.pyplot as plt
import numpy as np
import time


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from data.filter_and_diff import get_trajectory

# from eval.metrics import get_rmse

# Specify which trajectories to use
which_object = 'benchmark_box'
run = 'small_dist'
training_nr = 6
test_nr = 18
training_path = '../../data/extracted/' + which_object + '/' + run + '/' + str(training_nr) + '.csv'
test_path = '../../data/extracted/' + which_object + '/' + run + '/' + str(test_nr) + '.csv'


def get_kernels(idx=0):
    if idx == 0:
        kernels = [ConstantKernel() * RBF() + WhiteKernel(),
                   ConstantKernel() * RBF() + WhiteKernel(),
                   ConstantKernel() * RBF() + WhiteKernel(),
                   ConstantKernel() * RBF() + WhiteKernel(),
                   ConstantKernel() * RBF() + WhiteKernel(),
                   ConstantKernel() * RBF() + WhiteKernel()
                   ]
    elif idx == 1:
        kernels = [ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel()
                   ]
    elif idx == 2:
        kernels = [ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * Matern() + WhiteKernel()
                   ]
    elif idx == 3:
        kernels = [ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel()
                   ]
    elif idx == 4:
        kernels = [ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel()
                   ]
    elif idx == 5:
        kernels = [ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel()
                   ]
    elif idx == 6:
        kernels = [ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel()
                   ]
    elif idx == 7:
        kernels = [ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel()
                   ]

    return kernels


def train_gpr(x, y, prior=True, kernel_idx=0):
    """
        Train a GPR model
        x: Training samples of shape (n_features, n_samples)
        y: Training targets of shape (n_targets, n_samples)
    """
    # Get output dimension
    n_targets = y.shape[0]
    y_train = np.copy(y)

    # Subtract prior mean from z-acceleration, i.e., \mu(o_z) = -9.81
    if prior:
        y_train[2] += 9.81  # continuous-time case

    kernels = get_kernels(idx=kernel_idx)

    # Create GPR models
    gpr = []
    for ii in range(n_targets):
        gpr_model = GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=4, kernel=kernels[ii])
        gpr.append(gpr_model)
        t0 = time.time()
        gpr[ii].fit(x.T, y_train[ii].T)
        gpr_fit = time.time() - t0
        # print(f"GP model {ii} fitted in {gpr_fit:.2f} seconds, kernel is: {gpr[ii].kernel_}")

    return gpr


def pred_gpr(x, gpr, prior=True):
    """
        Predict with a GPR model
        x:      Test data of shape (n_features, n_samples)
        return: Predicted targets of shape (n_targets, n_samples)
    """
    if isinstance(gpr, list):
        out_dim = len(gpr)
        n_samples = x.shape[1]
        y_pred, sigma_pred = np.empty((out_dim, n_samples)), np.empty((out_dim, n_samples))
        for ii in range(out_dim):
            y_pred[ii, :], sigma_pred[ii, :] = gpr[ii].predict(x.T, return_std=True)
    else:
        y_pred, sigma_pred = gpr.predict(x.T, return_std=True)

    # If prior mean has been specified, add back
    if prior:
        y_pred[2] -= 9.81  # continuous-time case

    return y_pred, sigma_pred


if __name__ == "__main__":
    # Training data
    _, x_train, y_train = get_trajectory(training_path)

    # Test data
    t_test, x_test, y_test = get_trajectory(test_path)

    # Train GPR model
    gpr_kernel = RBF()
    gpr_models = train_gpr(x_train[3:], y_train, gpr_kernel)

    # Predict with trained SVR models
    y_pred, sigma_pred = pred_gpr(x_test[3:], gpr_models)

    # Compute RMSE
    # err = get_rmse(y_test, y_pred)
    # print(f"RMSE: {err:.3f}")

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
