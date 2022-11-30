import time

import numpy as np
import numpy.linalg as la


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern
from data.mechanics import sub_prior, grad_proj


def get_kernels(idx=0):
    if idx == 0:
        kernels = [ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ]
    elif idx == 1:
        kernels = [ConstantKernel() * RBF(length_scale=np.ones(11)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(11)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(11)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ]
    elif idx == 2:
        kernels = [ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1))
                   ]
    elif idx == 3:
        kernels = [ConstantKernel() * Matern(length_scale=np.ones(11)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern(length_scale=np.ones(11)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern(length_scale=np.ones(11)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ]
    elif idx == 4:
        kernels = [ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1))
                   ]
    elif idx == 5:
        kernels = [ConstantKernel() * RBF(length_scale=np.ones(11)) + ConstantKernel() * Matern(length_scale=np.ones(11)) + WhiteKernel(
            noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(11)) + ConstantKernel() * Matern(length_scale=np.ones(11)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(11)) + ConstantKernel() * Matern(length_scale=np.ones(11)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ]
    elif idx == 6:
        kernels = [ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ]
    elif idx == 7:
        kernels = [ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern() + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ConstantKernel() * RBF(length_scale=np.ones(10)) + ConstantKernel() * Matern(length_scale=np.ones(10)) + WhiteKernel(
                       noise_level_bounds=(1e-5, 1e-1)),
                   ]

    return kernels


class GPRegressor:
    dt = 0.01
    ignore_state_components = np.array([[1, 2],
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

    def __init__(self, n_features, n_targets, kernel_idx=0, prior=True, sys_rep='discrete', n_restarts=4,
                 use_gpflow=False):
        self.models = []
        self.n_features = n_features
        self.n_targets = n_targets
        self.kernel = get_kernels(kernel_idx)
        self.prior = prior
        self.sys_rep = sys_rep
        self.n_restarts = n_restarts
        self.X_Train = []
        self.Y_train = []

    def train(self, X, Y):
        self.X_train = X
        self.Y_train = Y

        if self.prior:
            Y_train = sub_prior(X, Y, sys_rep='disc')
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
            # if not self.use_gpflow:
            gpr_model = GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=self.n_restarts,
                                                 kernel=self.kernel[ii], alpha=1e-5)
            gpr_model.fit(X_use.T, Y_train[ii].T)
            # else:
            #     X_tf, Y_tf = tf.convert_to_tensor(X_use.T, dtype=tf.float64), \
            #                  tf.convert_to_tensor(Y_train[np.newaxis, ii].T, dtype=tf.float64)
            #     gpr_model = gpflow.models.GPR((X_tf, Y_tf), kernel=gpflow.kernels.SquaredExponential())
            #     opt = gpflow.optimizers.Scipy()
            #     opt.minimize(gpr_model.training_loss, gpr_model.trainable_variables)
            self.models.append(gpr_model)

            # if not self.use_gpflow:
            print(f"GP model {ii} fitted in {time.time() - t0:.2f} seconds, kernel is: {self.models[ii].kernel_}")
            # else:
        print(f"GP fitting took {time.time() - t_total:.2f} seconds.")

    def predict(self, X):
        """
        :param X:   Test inputs of shape (sys_dim, n_samples)
        :return:
        """

        if X.ndim == 1:
            X = X[:, np.newaxis]

        if X.shape[0] is not self.n_features:
            raise SystemExit('Test input does not have appropriate dimensions!')

        n_samples = X.shape[1]
        mu_d, Sigma_d = np.empty((self.n_targets, n_samples)), np.empty((self.n_targets, n_samples))
        for ii in range(self.n_targets):
            # Remove useless state components
            if not self.ignore_state_components.any():
                X_use = np.copy(X)
            else:
                X_use = np.delete(X, self.ignore_state_components[ii], axis=0)

            mu_d[ii], Sigma_d[ii] = self.models[ii].predict(X_use.T, return_std=True)

        # Add prior mean function back
        if self.prior:
            Y_pred = sub_prior(X, mu_d, sub=False, sys_rep='disc')
        else:
            Y_pred = mu_d

        # Normalize quaternion
        Y_pred[3:7] = Y_pred[3:7] / la.norm(Y_pred[3:7], axis=0)
        # if self.sys_rep == 'discrete':

        return Y_pred, Sigma_d

    def predict_unc_prop(self, mu_k, Sigma_k):
        """
        :param mu_k:    Mean of the current state distribution of shape (sys_dim,)
        :param Sigma_k: Variance of the current state distribution stored in a vector of shape (sys_dim,)
        :return:        Predicted mean of shape (sys_dim,)
                        Predicted covariance of shape (sys_dim,)
                        Predicted propagated covariance of shape (sys_dim,)
        """

        mu_k = mu_k[:, np.newaxis]

        n_samples = mu_k.shape[1]
        mu_d, Sigma_d = np.empty((self.n_targets, 1)), np.empty((self.n_targets, 1))
        for ii in range(self.n_targets):
            # Remove useless state components
            if not self.ignore_state_components.any():
                X_use = np.copy(mu_k)
            else:
                X_use = np.delete(mu_k, self.ignore_state_components[ii], axis=0)

            mu_d[ii], Sigma_d[ii] = self.models[ii].predict(X_use.T, return_std=True)

        # Add prior mean function back
        if self.prior:
            mu_kp1 = sub_prior(mu_k, mu_d, sub=False, sys_rep='disc')
        else:
            mu_kp1 = mu_d
        # Normalize quaternion
        mu_kp1[3:7] = mu_kp1[3:7] / la.norm(mu_kp1[3:7], axis=0)

        # Propagate uncertainty (Hewing et al., 2018)
        A = np.concatenate((grad_proj(mu_k), np.eye(self.n_targets)), axis=1)

        # lower_left = np.zeros((self.n_targets, self.n_targets))  # TODO
        lower_left = self.grad_mu(mu_k) @ np.diag(Sigma_k)
        B = np.block([
            [np.diag(Sigma_k), lower_left.T],
            [lower_left, np.diag(Sigma_d[:, 0])]
        ])
        Sigma_kp1 = A @ B @ A.T

        return mu_kp1, Sigma_kp1.diagonal(), Sigma_d

    def compute_grams(self):
        """
        :return:    Gram matrices of shape (self.n_targets, n_training_samples, n_training_samples)
        """
        n_samples = self.X_train.shape[1]

        self.K = np.empty((self.n_targets, n_samples, n_samples))

        for ii in range(self.n_targets):
            # Delete useless state components
            if not self.ignore_state_components.any():
                X = np.copy(self.X_train)
            else:
                X = np.delete(self.X_train, self.ignore_state_components[ii], axis=0)

            self.K[ii] = self.models[ii].kernel_(X.T)

    def get_kernel_params(self):
        # Get kernel parameters
        self.L = []
        self.C = []
        self.noise = []
        for ii in range(self.n_targets):
            kernel = self.models[ii].kernel_
            # Coefficient
            self.C.append(kernel.k1.k1.constant_value)

            # Length scales
            l = kernel.k1.k2.length_scale
            # Set length scales of not considered state components to very high value
            L = np.ones(self.n_features) * 1e+06
            counter = 0
            for kk in range(self.n_features):
                if kk not in self.ignore_state_components[ii]:
                    L[kk] = l[counter]
                    counter += 1
            self.L.append(np.diag(L))

            # Noise
            self.noise.append(kernel.k2.noise_level)

    def grad_k(self, x1, x2, ii):
        """
        :param x1:  input of size (n_features,)
        :param x2:  input of size (n_features,)
        :param ii:  Output component for which to compute the gradient
        :return:    Gradient of the RBF kernel function with respect to x1
        """

        # Compute gradient of \mu with respect to the x1
        grad_k = - self.C[ii] * np.exp(-0.5 * (x1 - x2).T @ la.inv(self.L[ii]) @ (x1 - x2)) * la.inv(self.L[ii]) @ (x1 - x2)

        return grad_k.reshape(grad_k.shape[0])

    def grad_mu(self, x):
        """
        :param x:   Test input for which to compute the gradient of the mean function
        :return:
        """
        n_train = self.X_train.shape[1]
        grad_mu = np.zeros((self.n_features, self.n_features))

        for ii in range(self.n_targets):
            grad_K = np.zeros((self.n_features, n_train))
            for kk in range(n_train):
                grad_K[:, kk] = self.grad_k(x, self.X_train[:, kk].reshape(self.n_features, 1), ii)
            # grad_mu[ii, :] = grad_K @ la.inv(self.K[ii] + np.eye(self.K[ii].shape[0]) * self.noise[ii]) @ self.Y_train[ii, :].T
            grad_mu[ii, :] = grad_K @ la.inv(self.K[ii]) @ self.Y_train[ii, :]

        return grad_mu
