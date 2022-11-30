import joblib
import numpy as np
import matplotlib.pyplot as plt
from data.data_handler import DataHandler
from metrics import integrate_trajectories_disc

obj = 0
path = ['../data/extracted/benchmark_box/small_dist', '../data/extracted/benchmark_box/med_dist', '../data/extracted/white_box/small_dist',
        '../data/extracted/white_box/med_dist']
n_traj = [20, 19, 18, 21]
test_run = [15]

if __name__ == "__main__":
    # Load trained mode
    model = joblib.load('./gp_models/gp_model.sav')
    model.compute_grams()
    model.get_kernel_params()

    # Define initial state
    dh = DataHandler(dt=0.01, filter_size=7, cont_time=False, rot_to_plane=True)
    dh.add_trajectories(path[0], test_run, 'test')

    # Integrate trajectories with uncertainty propagation
    X_int, Eul_int, Sigma_prop_int, Sigma_int = integrate_trajectories_disc(model, dh.X_test, dh.T_vec, unc_prop=True)

    # Predict trajectories with projectile model
    X_proj, Eul_proj, _, _ = integrate_trajectories_disc(model, dh.X_test, dh.T_vec, projectile=True)

    # Visualize trajectory in 3D
    # Position
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

    # Plot individual states with uncertainty tube around them
    fig, axes = plt.subplots(4, 4)
    # Position
    for ii in range(3):
        axes[0, ii].plot(np.arange(traj_len), X_int[ii], 'k-')
        axes[0, ii].fill_between(np.arange(traj_len), X_int[ii] - Sigma_int[ii], X_int[ii] + Sigma_int[ii], color='r', alpha=0.2)
        axes[0, ii].fill_between(np.arange(traj_len), X_int[ii] - Sigma_prop_int[ii], X_int[ii] + Sigma_prop_int[ii], color='b', alpha=0.2)
        axes[0, ii].set_ylabel('$o_' + str(ii + 1) + '$')

    # Orientation
    for ii in range(3, 7):
        axes[1, ii - 3].plot(np.arange(traj_len), X_int[ii], 'k-')
        axes[1, ii - 3].fill_between(np.arange(traj_len), X_int[ii] - Sigma_int[ii], X_int[ii] + Sigma_int[ii], color='r', alpha=0.2)
        axes[1, ii - 3].fill_between(np.arange(traj_len), X_int[ii] - Sigma_prop_int[ii], X_int[ii] + Sigma_prop_int[ii], color='b', alpha=0.2)
        axes[1, ii - 3].set_ylabel('$q_' + str(ii - 2) + '$')

    # Linear velocity
    for ii in range(7, 10):
        axes[2, ii - 7].plot(np.arange(traj_len), X_int[ii], 'k-')
        axes[2, ii - 7].fill_between(np.arange(traj_len), X_int[ii] - Sigma_int[ii], X_int[ii] + Sigma_int[ii], color='r', alpha=0.2)
        axes[2, ii - 7].fill_between(np.arange(traj_len), X_int[ii] - Sigma_prop_int[ii], X_int[ii] + Sigma_prop_int[ii], color='b', alpha=0.2)
        axes[2, ii - 7].set_ylabel('$v_' + str(ii - 6) + '$')

    # Angular velocity
    for ii in range(10, 13):
        axes[3, ii - 10].plot(np.arange(traj_len), X_int[ii], 'k-')
        axes[3, ii - 10].fill_between(np.arange(traj_len), X_int[ii] - Sigma_int[ii], X_int[ii] + Sigma_int[ii], color='r', alpha=0.2)
        axes[3, ii - 10].fill_between(np.arange(traj_len), X_int[ii] - Sigma_prop_int[ii], X_int[ii] + Sigma_prop_int[ii], color='b', alpha=0.2)
        axes[3, ii - 10].set_ylabel('$\\omega_' + str(ii - 9) + '$')

    plt.show()
