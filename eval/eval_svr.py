from metrics import *
from learning.svr.SVR import train_svr, pred_svr
from data.filter_and_diff import get_trajectory
import matplotlib.pyplot as plt

# Specify which trajectories to use
which_object = 'white_box'
run = 'med_dist'
training_runs = np.array([1])
# training_runs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
test_nr = 1
test_path = '../data/extracted/' + which_object + '/' + run + '/' + str(test_nr) + '.csv'

if __name__ == "__main__":
    # Training data
    training_path = '../data/extracted/' + which_object + '/' + run + '/' + str(training_runs[0]) + '.csv'
    _, x_train, y_train = get_trajectory(training_path)
    if len(training_runs) > 1:
        for ii in range(1, len(training_runs)):
            training_path = '../data/extracted/' + which_object + '/' + run + '/' + str(training_runs[ii]) + '.csv'
            _, x_train_tmp, y_train_tmp = get_trajectory(training_path)
            x_train = np.concatenate((x_train, x_train_tmp), axis=1)
            y_train = np.concatenate((y_train, y_train_tmp), axis=1)

    # Test data
    t_test, x_test, y_test = get_trajectory(test_path)

    # Train SVR models
    svr_models = train_svr(x_train[3:], y_train)

    # Predict acceleration with trained SVR models
    y_pred = pred_svr(x_test[3:], svr_models)

    # Predict trajectory
    _, x_int = integrate_trajectory(svr_models, x_test[:, 0], t_eval=t_test)

    # Compute RMSE
    err = get_rmse(y_test, y_pred)
    print(f"RMSE is {err:.3f}")

    # Plot real and predicted position, velocity and acceleration
    fig, axes = plt.subplots(3, 3)
    for ii in range(3):
        axes[0, ii].plot(t_test, x_test[ii, :], color='b', label='$\\xi_' + str(ii) + '$')
        axes[0, ii].plot(t_test, x_int[ii, :], color='r', label='$\\hat{\\xi}_' + str(ii) + '$')
        axes[1, ii].plot(t_test, x_test[ii + 3, :], color='b', label='$\\dot{\\xi}_' + str(ii) + '$')
        axes[1, ii].plot(t_test, x_int[ii + 3, :], color='r', label='$\\dot{\\hat{\\xi}}_' + str(ii) + '$')
        axes[2, ii].plot(t_test, y_test[ii, :], color='b', label='$\\ddot{\\xi}_' + str(ii) + '$')
        axes[2, ii].plot(t_test, y_pred[ii, :], color='r', label='$\\ddot{\\hat{\\xi}}_' + str(ii) + '$')

        axes[0, ii].legend(shadow=True, fancybox=True)
        axes[1, ii].legend(shadow=True, fancybox=True)
        axes[2, ii].legend(shadow=True, fancybox=True)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    # 3D plot of real and predicted trajectory
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x_test[0, :], x_test[1, :], x_test[2, :], color='b', label='$\\mathbf{\\xi}}$')
    ax.scatter(x_int[0, :], x_int[1, :], x_int[2, :], color='r', label='$\\mathbf{\\hat{\\xi}}}$')
    ax.legend(shadow=True, fancybox=True)
    plt.show()
