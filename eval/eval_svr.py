import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
from eval.functions.metrics import get_rmse, integrate_trajectory_old
from trash.SVR import train_svr, pred_svr
from data.filter_and_diff import get_trajectory

# Specify which trajectories to use
which_object = 'white_box'
run = 'med_dist'
# training_runs = np.array([1])
training_runs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# test_runs = ([11, 12, 13, 14, 15, 16, 17, 18])
test_runs = ([13])

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
    test_path = '../data/extracted/' + which_object + '/' + run + '/' + str(test_runs[0]) + '.csv'
    t_test, x_test, y_test = get_trajectory(test_path)
    if len(test_runs) > 1:
        for ii in range(1, len(test_runs)):
            test_path = '../data/extracted/' + which_object + '/' + run + '/' + str(test_runs[ii]) + '.csv'
            _, x_test_tmp, y_test_tmp = get_trajectory(test_path)
            x_test = np.concatenate((x_test, x_test_tmp), axis=1)
            y_test = np.concatenate((y_test, y_test_tmp), axis=1)

    # Train SVR models
    svr_models = train_svr(x_train[3:], y_train)

    # Predict acceleration with trained SVR models
    y_pred = pred_svr(x_test[3:], svr_models)

    # Predict first trajectory in the test set
    _, x_int = integrate_trajectory_old(svr_models, x_test[:, 0], t_eval=t_test, estimator='svr')

    # Compute RMSE
    rmse = get_rmse(y_test, y_pred)
    print(f"RMSE is {rmse:.3f}")

    # Compute deviation of predicted and real endpoint
    dist = la.norm(x_test[0:3, len(t_test) - 1] - x_int[0:3, -1])
    print(f"End point deviation is {dist:.3f}")

    # Plot real and predicted position, velocity and acceleration
    fig, axes = plt.subplots(3, 3)
    for ii in range(3):
        axes[0, ii].plot(t_test, x_test[ii, :len(t_test)], color='b', label='$\\xi_' + str(ii) + '$')
        axes[0, ii].plot(t_test, x_int[ii, :], color='r', label='$\\hat{\\xi}_' + str(ii) + '$')
        axes[1, ii].plot(t_test, x_test[ii + 3, :len(t_test)], color='b', label='$\\dot{\\xi}_' + str(ii) + '$')
        axes[1, ii].plot(t_test, x_int[ii + 3, :], color='r',
                         label='$\\dot{\\hat{\\xi}}_' + str(ii) + '$')
        axes[2, ii].plot(t_test, y_test[ii, :len(t_test)], color='b', label='$\\ddot{\\xi}_' + str(ii) + '$')
        axes[2, ii].plot(t_test, y_pred[ii, :len(t_test)], color='r', label='$\\ddot{\\hat{\\xi}}_' + str(ii) + '$')

        axes[0, ii].legend(shadow=True, fancybox=True)
        axes[1, ii].legend(shadow=True, fancybox=True)
        axes[2, ii].legend(shadow=True, fancybox=True)

    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()

    # 3D plot of real and predicted trajectory
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x_test[0, :len(t_test)], x_test[1, :len(t_test)], x_test[2, :len(t_test)], color='b',
               label='$\\mathbf{\\xi}}$')
    ax.scatter(x_int[0, :], x_int[1, :], x_int[2, :], color='r', label='$\\mathbf{\\hat{\\xi}}}$')
    ax.set_xlabel('$\\xi_1$')
    ax.set_ylabel('$\\xi_2$')
    ax.set_zlabel('$\\xi_3$')
    ax.set_title('Predicted and real flying trajectories')
    ax.legend(shadow=True, fancybox=True)
    plt.show()
