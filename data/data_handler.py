import numpy as np

from .filter_and_diff import get_trajectories
from .mechanics import quat2eul

class DataHandler:

    def __init__(self, dt=0.1):
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.Eul_test = []
        self.dt = dt
        self.T_vec = []

    def add_trajectories(self, path_to_dir, indices, which='train'):
        xi, _, eul, T_vec = get_trajectories(path_to_dir, indices, only_pos=False, ang_vel=True)

        X = np.copy(xi)
        Y = np.copy(xi)

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
