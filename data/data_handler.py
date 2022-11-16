import numpy as np

from .filter_and_diff import get_trajectories
from .mechanics import quat2eul


class DataHandler:

    def __init__(self, dt=0.01, filter_size=5, cont_time=False):
        self.dt = dt
        self.filter_size = filter_size
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.Eul_test = []
        self.T_vec = []
        self.cont_time = cont_time

    def add_trajectories(self, path_to_dir, indices, which='train'):
        xi, _, eul, T_vec = get_trajectories(path_to_dir, indices, only_pos=False, ang_vel=True,
                                             filter_size=self.filter_size, cont_time=self.cont_time)

        # Initialize X and Y with the states
        X = np.copy(xi)
        Y = np.copy(xi)

        # Delete the last value of each trajectory from X and the first value of each trajectory from Y
        del_x = np.zeros(len(T_vec))
        del_y = np.zeros(len(T_vec))
        for ii in range(len(T_vec)):
            del_x[ii] = np.sum(T_vec[:ii + 1]) - 1
            del_y[ii] = np.sum(T_vec[:ii])

        # As X: every trajectory except from last datapoint
        # As Y: every trajectory except from first datapoint
        X = np.delete(X, del_x.astype(int), axis=1)
        Y = np.delete(Y, del_y.astype(int), axis=1)

        if which == 'train':
            if not self.X_train:
                self.X_train = X
                self.Y_train = Y
            else:
                self.X_train = np.concatenate((self.X_train, X), axis=1)
                self.Y_train = np.concatenate((self.Y_train, Y), axis=1)
        else:
            if not self.X_test:
                self.X_test = X
                self.Y_test = Y
                self.T_vec = T_vec
                self.Eul_test = quat2eul(self.X_test[3:7])
            else:
                self.X_test = np.concatenate((self.X_test, X), axis=1)
                self.Y_test = np.concatenate((self.Y_test, Y), axis=1)
                self.T_vec = np.concatenate((self.T_vec, T_vec))
                self.Eul_test = np.concatenate(self.Eul_test, quat2eul(self.X_test[3:7]))
