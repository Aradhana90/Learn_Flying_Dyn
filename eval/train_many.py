import random
import numpy as np
import joblib
import os

from data.data_handler import DataHandler
from learning.gpr.GPR_disc import GPRegressor

# Continuous or discrete time
cont = True

# SVR or GPR
gp = True

if gp:
    if cont:
        from learning.gpr.GPR_cont import GPRegressor as Regressor
    else:
        from learning.gpr.GPR_disc import GPRegressor as Regressor
else:
    if cont:
        from learning.svr.SVR_cont import SVRegressor as Regressor
    else:
        from learning.svr.SVR_disc import SVRegressor as Regressor

# Specify algorithm and system model
obj = [0]
kernel_index = [9]
N_cross_validation = 10
F = [10]  # must be even

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

                    print('Training runs 1: ', training_runs1, ', Training runs 2: ', training_runs2)

                    dh.add_trajectories(data_path[0], training_runs1, 'train')
                    dh.add_trajectories(data_path[1], training_runs2, 'train')

                    if cont:
                        n_targets = 6
                        if gp:
                            store_path = './cross_valid_models_cont/obj_' + str(o) + '/kernel_' + str(k) + '/F_' + str(f) + '/' + str(_)
                        else:
                            store_path = './cross_valid_models_cont_svr/obj_' + str(o) + '/F_' + str(f) + '/' + str(_)
                    else:
                        n_targets = 13
                        if gp:
                            store_path = './cross_valid_models_disc/obj_' + str(o) + '/kernel_' + str(k) + '/F_' + str(f) + '/' + str(_)
                        else:
                            store_path = './cross_valid_models_disc_svr/obj_' + str(o) + '/F_' + str(f) + '/' + str(_)

                    if gp:
                        model = Regressor(n_features=13, n_targets=n_targets, prior=True, kernel_idx=k, n_restarts=4)
                    else:
                        model = Regressor(n_features=13, n_targets=n_targets)

                    model.train(dh.X_train, dh.Y_train)

                    if not os.path.exists(store_path):
                        os.makedirs(store_path)

                    # Store model
                    joblib.dump(model, store_path + '/model.sav')
                    joblib.dump(training_runs1, store_path + '/train_traj1.sav')
                    joblib.dump(training_runs2, store_path + '/train_traj2.sav')
