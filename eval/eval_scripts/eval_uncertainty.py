import joblib
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from data.mechanics import sample_random_states
from data.data_handler import DataHandler

cont = True

N_CV = 10
obj = 0
kernel_idx = 0
n_samples = 1000
X_rand = sample_random_states(cont=cont, n_samples=n_samples)

# Get all data points
data_path_all = ['../data/extracted/benchmark_box/small_dist', '../data/extracted/benchmark_box/med_dist', '../data/extracted/white_box/small_dist',
                 '../data/extracted/white_box/med_dist']
n_traj_all = [20, 19, 18, 21]
if obj == 0:
    n_traj = n_traj_all[0:2]
    data_path = data_path_all[0:2]
else:
    n_traj = n_traj_all[2:4]
    data_path = data_path_all[2:4]
dh = DataHandler(dt=0.01, filter_size=7, cont_time=cont, rot_to_plane=True)
dh.add_trajectories(data_path[0], np.arange(1, n_traj[0] + 1), 'train')
dh.add_trajectories(data_path[1], np.arange(1, n_traj[1] + 1), 'train')

# Replace certain components of the random samples by the average of the training data
X_rand[3:-1]

# F_vec = [2, 4, 6, 8, 10, 14, 18, 22, 26, 30]
F_vec = [30]

Sigma_o = np.zeros((len(F_vec), N_CV))
Sigma_q = np.zeros((len(F_vec), N_CV))
Sigma_v = np.zeros((len(F_vec), N_CV))
Sigma_w = np.zeros((len(F_vec), N_CV))
for ii in range(len(F_vec)):
    F = F_vec[ii]
    for kk in range(N_CV):
        print(kk)
        # Get model
        if cont:
            model_path = '../cross_valid_models_cont/obj_' + str(obj) + '/kernel_' + str(kernel_idx) + '/F_' + str(F) + '/' + str(kk) + '/model.sav'
        else:
            model_path = '../cross_valid_models_disc/obj_' + str(obj) + '/kernel_' + str(kernel_idx) + '/F_' + str(F) + '/' + str(kk) + '/model.sav'
        model = joblib.load(model_path)
        _, Sigma = model.predict(X_rand.T)
        # Sigma = np.sqrt(Sigma)
        if cont:
            Sigma_v[ii, kk] = np.mean(np.linalg.norm(Sigma[0:3], axis=0))
            Sigma_w[ii, kk] = np.mean(np.linalg.norm(Sigma[3:6], axis=0))
        else:
            Sigma_o[ii, kk] = np.mean(np.linalg.norm(Sigma[0:3], axis=0))
            Sigma_q[ii, kk] = np.mean(np.linalg.norm(Sigma[3:7], axis=0))
            Sigma_v[ii, kk] = np.mean(np.linalg.norm(Sigma[7:10], axis=0))
            Sigma_w[ii, kk] = np.mean(np.linalg.norm(Sigma[10:13], axis=0))

# Plot
if cont:
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(F_vec, np.mean(Sigma_v, axis=1))
    axes[1].plot(F_vec, np.mean(Sigma_w, axis=1))
else:
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(F_vec, np.mean(Sigma_o, axis=1))
    axes[0, 1].plot(F_vec, np.mean(Sigma_q, axis=1))
    axes[1, 0].plot(F_vec, np.mean(Sigma_v, axis=1))
    axes[1, 1].plot(F_vec, np.mean(Sigma_w, axis=1))

tikzplotlib.save("matern.tex")

plt.show()
