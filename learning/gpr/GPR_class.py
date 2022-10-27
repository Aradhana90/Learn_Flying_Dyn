import numpy as np
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class GPRegressor:

    def __init__(self, n_features, n_targets, kernel, use_prior=True, sys_rep='discrete'):
        self.models = []
        self.n_features = n_features
        self.n_targets = n_targets
        self.kernel = kernel
        self.use_prior = use_prior
        self.sys_rep = sys_rep

    def train(self, X, Y):
        # if self.prior:
        # if self.sys_rep == 'discrete':

        for ii in range(self.n_targets):
            gpr_model = GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=4, kernel=self.kernel)
            self.models.append(gpr_model)
            t0 = time.time()
            self.models[ii].fit(X.T, Y[ii].T)
            gpr_fit = time.time() - t0
            print(f"GP model {ii} fitted in {gpr_fit:.2f} seconds, kernel is: {self.models[ii].kernel_}")

    def predict(self, X):
        if X.ndim == 1:
            X = X[:, np.newaxis]

        if X.shape[0] is not self.n_features:
            raise SystemExit('Test input does not have appropriate dimensions!')

        n_samples = X.shape[1]
        y_pred, sigma_pred = np.empty((self.n_targets, n_samples)), np.empty((self.n_targets, n_samples))
        for ii in range(self.n_targets):
            y_pred[ii], sigma_pred[ii] = self.models[ii].predict(X.T, return_std=True)

        # if self.prior:
        # if self.sys_rep == 'discrete':

        return y_pred, sigma_pred
