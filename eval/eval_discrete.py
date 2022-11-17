import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern

from data.data_handler import DataHandler
from learning.gpr.GPR_disc import GPRegressor
from learning.svr.SVR_disc import SVRegressor
from metrics import get_rmse, get_rmse_eul, eul_norm, integrate_trajectories_disc

# Specify algorithm and system model
alg = 'gpr'
ang_vel = True
sys_rep = 'cont'
aug_factor = 0
kernel_indices = [0, 1, 5, 6, 7]
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
# test_runs = ([17])

if __name__ == "__main__":
    # Training data
    dh = DataHandler(dt=0.01, filter_size=7, cont_time=False)
    dh.add_trajectories(training_dir, training_runs, 'train')
    dh.add_trajectories(test_dir, test_runs, 'test')

    print('Created training and test data.')

    for k in range(len(kernel_indices)):
        # Create GPRegressor object handling the training and prediction
        if alg == 'gpr':
            model = GPRegressor(n_features=13, n_targets=13, prior=True, kernel_idx=kernel_indices[k], n_restarts=4, use_gpflow=False)
        else:
            model = SVRegressor(n_features=13, n_targets=13, prior=True)

        # Train
        model.train(dh.X_train, dh.Y_train)

        # Predict
        if alg == 'gpr':
            Y_pred, Sigma_pred = model.predict(dh.X_test)
        else:
            Y_pred = model.predict(dh.X_test)

        # Compute RMSE
        rmse_pos = get_rmse(dh.Y_test[0:3], Y_pred[0:3]) * 1e5
        rmse_ori = get_rmse_eul(dh.Y_test[3:7], Y_pred[3:7]) * 100
        rmse_lin = get_rmse(dh.Y_test[7:10], Y_pred[7:10]) * 1000
        rmse_ang = get_rmse(dh.Y_test[10:13], Y_pred[10:13]) * 360 / (2 * np.pi)
        print(kernel_indices[k])
        print(f"RMSE in position in m * 10^5 is {rmse_pos:.3f}.")
        print(f"RMSE in linear velocity in m/s * 1000 is {rmse_lin:.3f}.")
        print(f"RMSE in orientation in deg * 10^2 is {rmse_ori:.3f}.")
        print(f"RMSE in angular velocity in deg/s is {rmse_ang:.3f}.")

        # Predict trajectories with GP
        X_int, Eul_int = integrate_trajectories_disc(model, dh.X_test, dh.T_vec)

        # Predict trajectories with projectile model
        X_proj, Eul_proj = integrate_trajectories_disc(model, dh.X_test, dh.T_vec, projectile=True)

        # Deviations in final position and orientation
        diff_pos, diff_ori = np.zeros(len(test_runs)), np.zeros(len(test_runs))
        for ii in range(len(dh.T_vec)):
            diff_pos[ii] = la.norm(
                dh.Y_test[0:3, np.sum(dh.T_vec[:ii + 1] - 1) - 1] - X_int[0:3, np.sum(dh.T_vec[:ii + 1]) - 1])
            diff_ori[ii] = eul_norm(dh.Eul_test[:, np.sum(dh.T_vec[:ii + 1] - 1) - 1],
                                    Eul_int[:, np.sum(dh.T_vec[:ii + 1]) - 1])
        # print("End position deviations in cm are: ", diff_pos * 100)
        # print("End euler angle deviations in deg are: ", diff_ori)
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

    # Plot position and linear velocity
    fig, axes = plt.subplots(2, 3)
    for ii in range(3):
        axes[0, ii].plot(range(traj_len - 1), dh.X_test[ii, :dh.T_vec[0] - 1], color='b',
                         label='$o_' + str(ii) + '$')
        axes[0, ii].plot(range(traj_len), X_int[ii, :dh.T_vec[0]], color='r', label='$\\hat{o}_' + str(ii) + '$')
        axes[1, ii].plot(range(traj_len - 1), dh.X_test[ii + 7, :dh.T_vec[0] - 1], color='b',
                         label='$v_' + str(ii) + '$')
        axes[1, ii].plot(range(traj_len), X_int[ii + 7, :dh.T_vec[0]], color='r',
                         label='$\\hat{v}_' + str(ii) + '$')
        axes[0, ii].legend()
        axes[1, ii].legend()

    # Plot orientation and angular velocity
    fig, axes = plt.subplots(2, 3)
    axes[0, 0].plot(range(traj_len - 1), dh.Eul_test[0, :dh.T_vec[0] - 1], color='b', label='$\\phi$')
    axes[0, 0].plot(range(traj_len), Eul_int[0, :dh.T_vec[0]], color='r', label='$\\hat{\\phi}$')
    axes[0, 1].plot(range(traj_len - 1), dh.Eul_test[1, :dh.T_vec[0] - 1], color='b', label='$\\theta$')
    axes[0, 1].plot(range(traj_len), Eul_int[1, :dh.T_vec[0]], color='r', label='$\\hat{\\theta}$')
    axes[0, 2].plot(range(traj_len - 1), dh.Eul_test[2, :dh.T_vec[0] - 1], color='b', label='$\\psi$')
    axes[0, 2].plot(range(traj_len), Eul_int[2, :dh.T_vec[0]], color='r', label='$\\hat{\\psi}$')
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[0, 2].legend()
    for ii in range(3):
        axes[1, ii].plot(range(traj_len - 1), dh.X_test[ii + 10, :dh.T_vec[0] - 1], color='b',
                         label='$\\omega_' + str(ii) + '$')
        axes[1, ii].plot(range(traj_len), X_int[ii + 10, :dh.T_vec[0]], color='r',
                         label='$\\hat{\\omega}_' + str(ii) + '$')
        axes[0, ii].legend()
        axes[1, ii].legend()

    plt.show()
