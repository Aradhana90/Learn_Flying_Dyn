import time

import numpy as np


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
import warnings
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from data.mechanics import sub_prior


class SVRegressor:
    dt = 0.01
    kernel = 'rbf'
    ignore_state_components = np.array([[0, 1, 2],
                                        [0, 1, 2],
                                        [0, 1, 2],
                                        [0, 1, 2],
                                        [0, 1, 2],
                                        [0, 1, 2]])

    def __init__(self, n_features, n_targets, kernel=kernel):
        self.models = []
        self.n_features = n_features
        self.n_targets = n_targets
        self.kernel = kernel

    def train(self, X, Y):
        t_total = time.time()
        for ii in range(self.n_targets):
            # Delete useless state components
            if not self.ignore_state_components.any():
                X_use = np.copy(X)
            else:
                X_use = np.delete(X, self.ignore_state_components[ii], axis=0)

            # Create model
            t0 = time.time()
            svr_model = GridSearchCV(
                SVR(kernel=self.kernel, gamma=0.1),
                param_grid={"C": [1e-2, 1e-1, 1e0, 1e1], "gamma": [1e-2, 1e-1, 1e0]},
            )
            svr_model.fit(X_use.T, Y[ii].T)
            self.models.append(svr_model)

            print(f"SVR model {ii} fitted in {time.time() - t0:.2f} seconds, params are: {self.models[ii].best_params_}")

        print(f"SVR fitting took {time.time() - t_total:.2f} seconds.")

    def predict(self, X):
        if X.ndim == 1:
            X = X[:, np.newaxis]

        # if X.shape[0] is not self.n_features:
        #     raise SystemExit('Test input does not have appropriate dimensions!')

        n_samples = X.shape[1]
        Y_pred = np.empty((self.n_targets, n_samples))
        for ii in range(self.n_targets):
            # Remove useless state components
            if not self.ignore_state_components.any():
                X_use = np.copy(X)
            else:
                X_use = np.delete(X, self.ignore_state_components[ii], axis=0)

            Y_pred[ii] = self.models[ii].predict(X_use.T)

        return Y_pred, []
