import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from data.mechanics import quatmul


def sub_prior(X, Y, sub=True):
    """
    :param X:   Training data of shape (13, N_samples)
    :param Y:   Training targets of shape (13, N_samples)
    :param sub: True if the mean should be subtracted, False if the mean should be added back
    :return:    Mean-subtracted/added targets of shape (13, N_samples)
    """
    if X.ndim == 1:
        X = X[:, np.newaxis]
        Y = Y[:, np.newaxis]

    n_samples = X.shape[1]
    dt = 0.01

    if sub:
        sign = -1
    else:
        sign = 1

    # Position: o_next = o_cur + v_cur * dt + 0.5 * [0, 0, -9.81] * dt**2
    Y_new = np.copy(Y)
    Y_new[0] += sign * X[0] + sign * X[7] * dt
    Y_new[1] += sign * X[1] + sign * X[8] * dt
    Y_new[2] += sign * X[2] + sign * X[9] * dt - sign * 0.5 * 9.81 * dt ** 2

    # Orientation: q_next = q_cur + 0.5 * omega_cur * q_cur
    tmp = np.empty((4, n_samples))
    for ii in range(n_samples):
        q = X[3:7, ii]
        omega = np.append([0], X[10:13, ii])
        tmp[:, ii] = 0.5 * quatmul(omega, q) * dt

    Y_new[3:7] += sign * X[3:7] + sign * tmp

    # Linear velocity: v_next = v_cur + [0, 0, -9.81 dt]
    Y_new[7] += sign * X[7]
    Y_new[8] += sign * X[8]
    Y_new[9] += sign * X[9] - sign * 9.81 * dt

    # Angular velocity:
    Y_new[10:13] += sign * X[10:13]

    return Y_new


class GPRegressor:
    dt = 0.01

    def __init__(self, n_features, n_targets, kernel, prior=True, sys_rep='discrete', n_restarts=4):
        self.models = []
        self.n_features = n_features
        self.n_targets = n_targets
        self.kernel = kernel
        self.prior = prior
        self.sys_rep = sys_rep
        self.n_restarts = n_restarts
        self.ignore_state_components = np.array([[1, 2],
                                                 [0, 2],
                                                 [0, 1],
                                                 [0, 1, 2],
                                                 [0, 1, 2],
                                                 [0, 1, 2],
                                                 [0, 1, 2],
                                                 [0, 1, 2],
                                                 [0, 1, 2],
                                                 [0, 1, 2],
                                                 [0, 1, 2],
                                                 [0, 1, 2],
                                                 [0, 1, 2]], dtype=list)
        # self.ignore_state_components = np.array([])

    def train(self, X, Y):
        if self.prior:
            Y_train = sub_prior(X, Y)
        else:
            Y_train = Y

        # if self.sys_rep == 'discrete':
        t_total = time.time()
        for ii in range(self.n_targets):
            gpr_model = GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=self.n_restarts,
                                                 kernel=self.kernel)
            self.models.append(gpr_model)
            t0 = time.time()
            if not self.ignore_state_components.any():
                X_use = np.copy(X)
            else:
                X_use = np.delete(X, self.ignore_state_components[ii], axis=0)
            self.models[ii].fit(X_use.T, Y_train[ii].T)
            print(f"GP model {ii} fitted in {time.time() - t0:.2f} seconds, kernel is: {self.models[ii].kernel_}")
        print(f"GP fitting took {time.time() - t_total:.2f} seconds.")

    def predict(self, X):
        if X.ndim == 1:
            X = X[:, np.newaxis]

        if X.shape[0] is not self.n_features:
            raise SystemExit('Test input does not have appropriate dimensions!')

        n_samples = X.shape[1]
        Y_tmp, sigma_pred = np.empty((self.n_targets, n_samples)), np.empty((self.n_targets, n_samples))
        for ii in range(self.n_targets):
            if not self.ignore_state_components.any():
                X_use = np.copy(X)
            else:
                X_use = np.delete(X, self.ignore_state_components[ii], axis=0)
            Y_tmp[ii], sigma_pred[ii] = self.models[ii].predict(X_use.T, return_std=True)

        if self.prior:
            Y_pred = sub_prior(X, Y_tmp, sub=False)
        else:
            Y_pred = Y_tmp

        # Normalize quaternion
        Y_pred[3:7] = Y_pred[3:7] / np.linalg.norm(Y_pred[3:7], axis=0)
        # if self.sys_rep == 'discrete':

        return Y_pred, sigma_pred
