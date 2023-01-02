import random
import numpy as np
import joblib
import os

from data.data_handler import DataHandler
from learning.gpr.GPR_disc import GPRegressor

# Continuous or discrete time
cont = True
if cont:
    from learning.gpr.GPR_cont import GPRegressor
else:
    from learning.gpr.GPR_disc import GPRegressor

# Specify algorithm and system model
alg = 'gpr'
only_pos = False
ang_vel = True
prior = True
obj = [1]
kernel_index = [1]
N_cross_validation = 10
F = [2, 4, 6, 8, 10, 14, 18, 22, 26, 30]  # must be even

data_path_all = ['../data/extracted/benchmark_box/small_dist', '../data/extracted/benchmark_box/med_dist', '../data/extracted/white_box/small_dist',
                 '../data/extracted/white_box/med_dist']
n_traj_all = [20, 19, 18, 21]

filter_size = 7

if __name__ == "__main__":
    # Iterate over objects
    for o in obj:
        if o == 0:
            n_traj = n_traj_all[0:2]
            data_path = data_path_all[0:2]
        else:
            n_traj = n_traj_all[2:4]
            data_path = data_path_all[2:4]

        for k in kernel_index:
            # Iterate over number of training trajectories
            for f in F:
                # Iterate over models
                for _ in range(N_cross_validation):
                    # Training data
                    dh = DataHandler(dt=0.01, filter_size=7, cont_time=cont, rot_to_plane=True)

                    training_runs1 = np.sort(np.array(random.sample(list(np.arange(1, n_traj[0] + 1)), int(f / 2))))
                    training_runs2 = np.sort(np.array(random.sample(list(np.arange(1, n_traj[1] + 1)), int(f / 2))))

                    print('Training runs 1: ', training_runs1, ' Training runs 2: ', training_runs2)

                    dh.add_trajectories(data_path[0], training_runs1, 'train')
                    dh.add_trajectories(data_path[1], training_runs2, 'train')

                    if cont:
                        n_targets = 6
                        store_path = './cross_valid_models_cont/obj_' + str(o) + '/kernel_' + str(k) + '/F_' + str(f) + '/' + str(_)
                    else:
                        n_targets = 13
                        store_path = './cross_valid_models_disc/obj_' + str(o) + '/kernel_' + str(k) + '/F_' + str(f) + '/' + str(_)

                    model = GPRegressor(n_features=13, n_targets=n_targets, prior=prior, kernel_idx=k, n_restarts=4)

                    model.train(dh.X_train, dh.Y_train)

                    if not os.path.exists(store_path):
                        os.makedirs(store_path)

                    # Store model
                    joblib.dump(model, store_path + '/model.sav')
                    joblib.dump(training_runs1, store_path + '/train_traj1.sav')
                    joblib.dump(training_runs2, store_path + '/train_traj2.sav')
