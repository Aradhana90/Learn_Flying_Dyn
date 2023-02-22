import time

import numpy as np


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern
from data.mechanics import sub_prior


def get_kernels(idx=0):
    if idx == 0:
        kernels = [ConstantKernel() * RBF() + WhiteKernel(),
                   ConstantKernel() * RBF() + WhiteKernel(),
                   ConstantKernel() * RBF() + WhiteKernel(),
                   ConstantKernel() * RBF() + WhiteKernel(),
                   ConstantKernel() * RBF() + WhiteKernel(),
                   ConstantKernel() * RBF() + WhiteKernel()
                   ]
    elif idx == 1:
        kernels = [ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel()
                   ]
    elif idx == 2:
        kernels = [ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * Matern() + WhiteKernel()
                   ]
    elif idx == 3:
        kernels = [ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel()
                   ]
    elif idx == 4:
        kernels = [ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel()
                   ]
    elif idx == 5:
        kernels = [ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel()
                   ]
    elif idx == 6:
        kernels = [ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel()
                   ]
    elif idx == 7:
        kernels = [ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel()
                   ]
    elif idx == 8:
        kernels = [ConstantKernel() * RBF(length_scale=np.ones(10), length_scale_bounds=(1e-1, 1e5)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10), length_scale_bounds=(1e-1, 1e5)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10), length_scale_bounds=(1e-1, 1e5)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10), length_scale_bounds=(1e-1, 1e5)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10), length_scale_bounds=(1e-1, 1e5)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale=np.ones(10), length_scale_bounds=(1e-1, 1e5)) + WhiteKernel()
                   ]
    elif idx == 9:
        kernels = [ConstantKernel() * RBF(length_scale_bounds=(1e-1, 1e5)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale_bounds=(1e-1, 1e5)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale_bounds=(1e-1, 1e5)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale_bounds=(1e-1, 1e5)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale_bounds=(1e-1, 1e5)) + WhiteKernel(),
                   ConstantKernel() * RBF(length_scale_bounds=(1e-1, 1e5)) + WhiteKernel()
                   ]

    return kernels


class GPRegressor:
    dt = 0.01

    def __init__(self, n_features, n_targets, kernel_idx=0, prior=True, n_restarts=4):
        self.models = []
        self.n_features = n_features
        self.n_targets = n_targets
        self.kernel = get_kernels(kernel_idx)
        self.prior = prior
        self.n_restarts = n_restarts
        self.ignore_state_components = np.array([[0, 1, 2],
                                                 [0, 1, 2],
                                                 [0, 1, 2],
                                                 [0, 1, 2],
                                                 [0, 1, 2],
                                                 [0, 1, 2]])

    def train(self, X, Y):
        if self.prior:
            Y_train = sub_prior(X, Y, sys_rep='cont')
        else:
            Y_train = Y

        # if self.sys_rep == 'discrete':
        t_total = time.time()
        for ii in range(self.n_targets):
            # Delete useless state components
            if not self.ignore_state_components.any():
                X_use = np.copy(X)
            else:
                X_use = np.delete(X, self.ignore_state_components[ii], axis=0)

            # Create model
            t0 = time.time()
            gpr_model = GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=self.n_restarts,
                                                 kernel=self.kernel[ii], alpha=1e-5)
            gpr_model.fit(X_use.T, Y_train[ii].T)
            self.models.append(gpr_model)

            print(f"GP model {ii} fitted in {time.time() - t0:.2f} seconds, kernel is: {self.models[ii].kernel_}")

        print(f"GP fitting took {time.time() - t_total:.2f} seconds.")

    def predict(self, X):
        if X.ndim == 1:
            X = X[:, np.newaxis]

        # if X.shape[0] is not self.n_features:
        #     raise SystemExit('Test input does not have appropriate dimensions!')

        n_samples = X.shape[1]
        Y_tmp, sigma_pred = np.empty((self.n_targets, n_samples)), np.empty((self.n_targets, n_samples))
        for ii in range(self.n_targets):
            # Remove useless state components
            if not self.ignore_state_components.any():
                X_use = np.copy(X)
            else:
                X_use = np.delete(X, self.ignore_state_components[ii], axis=0)

            Y_tmp[ii], sigma_pred[ii] = self.models[ii].predict(X_use.T, return_std=True)

        # Add prior mean function back
        if self.prior:
            Y_pred = sub_prior(X, Y_tmp, sub=False, sys_rep='cont')
        else:
            Y_pred = Y_tmp

        return Y_pred, sigma_pred
