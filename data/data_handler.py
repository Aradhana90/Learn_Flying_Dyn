import numpy as np

from .filter_and_diff import get_trajectories


class DataHandler:

    def __init__(self, dt=0.1):
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.dt = dt
        self.T_vec = []

    def add_trajectories(self, path_to_dir, indices, which='train'):
        xi, _, eul, self.T_vec = get_trajectories(path_to_dir, indices, only_pos=False, ang_vel=True)
        X = xi[:, :-1]
        Y = xi[:, 1:]
        if which == 'train':
            if not self.X_train:
                self.X_train = X
                self.Y_train = Y
            else:
                np.concatenate((self.X_train, X), axis=1)
                np.concatenate((self.Y_train, Y), axis=1)
        else:
            if not self.X_test:
                self.X_test = X
                self.Y_test = Y
            else:
                np.concatenate((self.X_test, X), axis=1)
                np.concatenate((self.Y_test, Y), axis=1)
