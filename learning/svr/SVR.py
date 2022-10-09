from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
from data.filter_and_diff import get_trajectory
from eval.metrics import get_rmse
import time
import matplotlib.pyplot as plt

# Specify which trajectories to use
which_object = 'benchmark_box'
run = 'small_dist'
training_nr = 6
test_nr = 18
training_path = '../../data/extracted/' + which_object + '/' + run + '/' + str(training_nr) + '.csv'
test_path = '../../data/extracted/' + which_object + '/' + run + '/' + str(test_nr) + '.csv'


def train_svr(x, y, kernel='rbf'):
    """
        Train multiple 1D SVR models
        Input:  x: Input data       shape: (in_dim, n_data_points)
                y: Output data      shape: (out_dim, n_data_points)
    """
    # Get output_dimension
    out_dim = y.shape[0]

    # Create SVR models
    svr = []
    for ii in range(out_dim):
        svr_model = GridSearchCV(
            SVR(kernel=kernel, gamma=0.1),
            param_grid={"C": [1e-1, 1e0, 1e1, 1e2, 1e3], "gamma": [1e-2, 1e-1, 1e0, 1e1, 1e2]},
        )
        svr.append(svr_model)

    # Fit SVR models
    for ii in range(3):
        t0 = time.time()
        svr[ii].fit(x[:].T, y[ii, :])
        svr_fit = time.time() - t0
        print(f"Best SVR with params: {svr[ii].best_params_} and R2 score: {svr[ii].best_score_:.3f}")
        print("SVR complexity and bandwidth selected and model fitted in %.3f s" % svr_fit)

    return svr


def pred_svr(x, svr):
    if isinstance(svr, list):
        out_dim = len(svr)
        n_data_points = x.shape[1]
        y_pred = np.zeros((out_dim, n_data_points))
        for ii in range(out_dim):
            y_pred[ii, :] = svr[ii].predict(x.T)
    else:
        y_pred = svr.predict(x.T)

    return y_pred


if __name__ == "__main__":
    # Training data
    _, x_train, y_train = get_trajectory(training_path)

    # Test data
    t_test, x_test, y_test = get_trajectory(test_path)

    # Train SVR models
    svr_models = train_svr(x_train[3:], y_train)

    # Predict with trained SVR models
    y_pred = pred_svr(x_test[3:], svr_models)

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

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
