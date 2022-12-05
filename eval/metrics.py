import numpy as np
from learning.gpr.GPR_disc import GPRegressor
from scipy.integrate import solve_ivp
from data.mechanics import quatmul, quat2eul, projectile_motion_disc


def get_rmse(y, y_pred, eul=False):
    # out_dim = 3
    # # Ensure that both y and ypred have shape (out_dim, n_test_points)
    # if y.shape[1] == out_dim:
    #     y.reshape(3, y.shape[0])
    #
    # if y_pred.shape[1] == out_dim:
    #     y_pred.reshape(3, y_pred.shape[0])

    if y.shape[1] != y_pred.shape[1]:
        raise SystemExit('Same number of predictions and targets needed!')

    y_tmp, y_pred_tmp = np.copy(y), np.copy(y_pred)
    # For Euler angles in (-180, 180], project negative values to (180, 360)
    if eul:
        y_tmp[y_tmp < 0] += 360
        y_pred_tmp[y_pred_tmp < 0] += 360

    # Compute RMSE
    err = np.linalg.norm(y_tmp - y_pred_tmp, ord=2, axis=0) ** 2
    err = np.sqrt(err.sum() / len(err))

    return err


def get_rmse_eul(quat1, quat2):
    """
    :param quat1: Quaternion array of shape (4,n_samples)
    :param quat2: Quaternion array of shape (4,n_samples)
    :return:
    """
    n_samples = quat1.shape[1]
    eul1 = quat2eul(quat1)
    eul2 = quat2eul(quat2)
    err = 0
    for ii in range(n_samples):
        err += eul_norm(eul1[:, ii], eul2[:, ii])

    err = np.sqrt(err) / n_samples
    return err


def eul_norm(eul1, eul2):
    """
    :param eul1:    Euler angles of shape (3,) in (-180, 180]
    :param eul2:    Euler angles of shape (3,) in (-180, 180]
    :return:        Norm considering the fact that -180 equals 180
    """

    # diff = abs(eul1 - eul2)
    # diff[(eul1 < 0) & (eul2 > 0)] = min(diff, abs(eul1 + 360 - eul2))[(eul1 < 0) & (eul2 > 0)]
    # diff[(eul1 > 0) & (eul2 < 0)] = min(diff, abs(eul1 - eul2 - 360))[(eul1 > 0) & (eul2 < 0)]

    diff = abs(eul1 - eul2)
    angle_range = [180, 90, 180]
    for ii in range(3):
        if eul1[ii] < 0 and eul2[ii] > 0:
            diff[ii] = min(diff[ii], abs(eul1[ii] + 2 * angle_range[ii] - eul2[ii]))
        elif eul1[ii] > 0 and eul2[ii] < 0:
            diff[ii] = min(diff[ii], abs(eul1[ii] - eul2[ii] - 2 * angle_range[ii]))

    # Compute norm
    err = np.linalg.norm(diff, ord=2)
    # err = np.sqrt(err.sum() / len(err))

    return err


def dx_dt_old(t, x, model, only_pos=True, ang_vel=False, estimator='svr', prior=True, projectile=False):
    """
    :param t:           Time
    :param x:           Current state
    :param model:       List of 1D svr models or a GPR mode to predict the acceleration
    :param estimator:   Specifies the type of the the learned model ('svr' or 'gpr')
    :return:            Predicted linear (and angular) acceleration
    """
    if not ang_vel or only_pos:
        if only_pos:
            sys_dim = 3
        else:
            sys_dim = 7

        dxi_dt = x[sys_dim:]
        ddxi_dt = np.zeros(sys_dim)
        if not projectile:
            for ii in range(sys_dim):
                ddxi_dt[ii] = model[ii].predict(np.expand_dims(x[3:], axis=0))

    else:
        q = x[3:7]
        v = x[7:10]
        omega = x[10:13]
        dxi_dt = np.zeros(7)
        dxi_dt[0:3] = v
        dxi_dt[3:7] = 0.5 * quatmul(np.append([0], omega), q)
        ddxi_dt = np.zeros(6)
        if not projectile:
            for ii in range(6):
                ddxi_dt[ii] = model[ii].predict(np.expand_dims(x[3:], axis=0))

    # Check if prior mean function has been specified
    if prior is True:
        ddxi_dt[2] -= 9.81

    dx_dt = np.concatenate((dxi_dt, ddxi_dt))
    return dx_dt


def integrate_trajectory_old(model, x_init, t_eval, only_pos=True, ang_vel=False, estimator='svr', prior=True, projectile=False):
    """
    :param model:       SVR or GP models to predict the linear and angular acceleration for the current state
    :param x_init:      Initial state from which to start the numerical integration
    :param t_eval:      Timesteps for which to store the integration results
    :param ang_vel:     If True, the angular acceleration is predicted instead of the second derivative of the quaternions.
    :param estimator:   Specifies what kind of predictor is used, 'SVR' or 'GP'
    :return:            Predicted trajectory
    """

    if only_pos:
        sys_dim = 3
    else:
        sys_dim = 7

    sol = solve_ivp(dx_dt_old, t_span=(0, t_eval[-1]), y0=x_init, t_eval=t_eval, args=(model, only_pos, ang_vel, estimator, prior, projectile))

    if not only_pos:
        eul = quat2eul(sol.y[3:7])
    else:
        eul = []

    return sol.t, sol.y, eul


def integrate_trajectories_old(model, x_test, T_vec, only_pos=True, ang_vel=False, estimator='svr', prior=True, projectile=False):
    """
    :param model:       SVR or GP models to predict the linear and angular acceleration for the current state
    :param x_test:      N_tilde test trajectories, shape is (13, N)
    :param T_vec:       Contains as integers the lengths of the N_tilde trajectories, shape is (N_tilde,). Values have to add up to N.
    :param ang_vel:     If True, the angular acceleration is predicted instead of the second derivative of the quaternions.
    :param estimator:   Specifies what kind of predictor is used, 'svr' or 'gpr'
    :return:            Predicted trajectories
    """

    x_init = x_test[:, 0]
    t_eval = np.linspace(0, (T_vec[0] - 1) * 0.01, T_vec[0])
    _, X_int, Eul_int = integrate_trajectory_old(model=model, x_init=x_init, t_eval=t_eval, only_pos=only_pos,
                                                 ang_vel=ang_vel, estimator=estimator, prior=prior, projectile=projectile)
    N_tilde = len(T_vec)
    for n in range(1, N_tilde):
        x_init = x_test[:, np.sum(T_vec[:n])]
        t_eval = np.linspace(0, (T_vec[n] - 1) * 0.01, T_vec[n])
        _, X_tmp, Eul_tmp = integrate_trajectory_old(model=model, x_init=x_init, t_eval=t_eval, only_pos=only_pos,
                                                     ang_vel=ang_vel, estimator=estimator, prior=prior, projectile=projectile)
        X_int = np.concatenate((X_int, X_tmp), axis=1)
        if not only_pos:
            Eul_int = np.concatenate((Eul_int, Eul_tmp), axis=1)

    return X_int, Eul_int


def dx_dt(t, x, model, projectile):
    """
        :param t:           Time
        :param x:           Current state
        :param model:       List of 1D svr models or a GPR mode to predict the acceleration
        :param projectile:  If true, the simple projectile model is used instead of the learned prediction model
        :return:            Predicted linear (and angular) acceleration
        """

    q = x[3:7]
    v = x[7:10]
    omega = x[10:13]
    dxi_dt = np.zeros(7)
    dxi_dt[0:3] = v
    dxi_dt[3:7] = 0.5 * quatmul(np.append([0], omega), q)
    if not projectile:
        ddxi_dt, _ = model.predict(x)
    else:
        ddxi_dt = np.zeros(6)
        ddxi_dt[2] = -9.81

    dx_dt = np.concatenate((dxi_dt, ddxi_dt.reshape((6,))))
    return dx_dt


def integrate_trajectory(model, x_init, t_eval, projectile=False):
    """
        :param model:       SVR or GP models to predict the linear and angular acceleration for the current state
        :param x_init:      Initial state from which to start the numerical integration
        :param t_eval:      Timesteps for which to store the integration results
        :param projectile:  If true, the simple projectile model is used instead of the learned prediction model
        :return:            Predicted trajectory
        """

    sol = solve_ivp(dx_dt, t_span=(0, t_eval[-1]), y0=x_init, t_eval=t_eval, args=(model, projectile))
    eul = quat2eul(sol.y[3:7])

    return sol.t, sol.y, eul


def integrate_trajectories_cont(model, x_test, T_vec, projectile=False):
    """
        :param model:       SVR or GP models to predict the linear and angular acceleration for the current state
        :param x_test:      N_tilde test trajectories, shape is (13, N)
        :param T_vec:       Contains as integers the lengths of the N_tilde trajectories, shape is (N_tilde,). Values have to add up to N.
        :param projectile:  If true, the simple projectile model is used instead of the learned prediction model
        :return:            Predicted trajectories
        """

    x_init = x_test[:, 0]
    t_eval = np.linspace(0, (T_vec[0] - 1) * 0.01, T_vec[0])
    _, X_int, Eul_int = integrate_trajectory(model=model, x_init=x_init, t_eval=t_eval, projectile=projectile)
    N_tilde = len(T_vec)
    for n in range(1, N_tilde):
        x_init = x_test[:, np.sum(T_vec[:n])]
        t_eval = np.linspace(0, (T_vec[n] - 1) * 0.01, T_vec[n])
        _, X_tmp, Eul_tmp = integrate_trajectory(model=model, x_init=x_init, t_eval=t_eval, projectile=projectile)
        X_int = np.concatenate((X_int, X_tmp), axis=1)
        Eul_int = np.concatenate((Eul_int, Eul_tmp), axis=1)

    return X_int, Eul_int


def integrate_trajectories_disc(predictor, x_test, T_vec, projectile=False, unc_prop=False):
    """
    :param predictor:   Class containing the GP models
    :param x_test:      Test set of shape (N_sys, N_samples) providing the initial values
    :param T_vec:       Vector containing the lengths of the trajectories in the test set. Must sum up to n_samples
    :param projectile:  If set to true, only the simple projectile motion model is used
    :param unc_prop:    Whether to propagate the uncertainty, i.e., consider the test input as Gaussian distributed
    :return:            Predicted trajectories of shape (N_sys, N_samples)
    """
    sys_dim = x_test.shape[0]

    # Number of trajectories
    N_tilde = len(T_vec)

    # Variables to store the integrated trajectories
    X_int = np.empty((sys_dim, np.sum(T_vec)))
    Sigma_int = np.empty((sys_dim, np.sum(T_vec)))
    Sigma_prop_int = np.empty((sys_dim, sys_dim, np.sum(T_vec)))

    # Integrate trajectories in forward time, i.e., x_k+1 = f(x_k)
    for n in range(0, N_tilde):
        x_init = x_test[:, np.sum(T_vec[:n] - 1)]  # Initial state
        X_tmp = np.zeros((sys_dim, T_vec[n]))  # States
        X_tmp[:, 0] = x_init
        Sigma_tmp = np.zeros((sys_dim, T_vec[n]))  # Covariance matrices
        Sigma_prop_tmp = np.zeros((sys_dim, sys_dim, T_vec[n]))  # Covariance matrices
        for k in range(T_vec[n] - 1):
            x_cur = X_tmp[:, k]
            if not projectile:
                if isinstance(predictor, GPRegressor):
                    if unc_prop is False:
                        x_next, sigma_next = predictor.predict(x_cur)
                    else:
                        x_next, sigma_prop_next, sigma_next = predictor.predict_unc_prop(x_cur, Sigma_prop_tmp[:, :, k])
                        # sigma_prop_next = sigma_prop_next.reshape(sys_dim)
                        Sigma_prop_tmp[:, :, k + 1] = sigma_prop_next
                else:
                    x_next = predictor.predict(x_cur)
                x_next = x_next.reshape(sys_dim)
                sigma_next = sigma_next.reshape(sys_dim)
                Sigma_tmp[:, k + 1] = sigma_next
            else:
                x_next = projectile_motion_disc(x_cur)
            X_tmp[:, k + 1] = x_next

        X_int[:, np.sum(T_vec[:n]): np.sum(T_vec[:n]) + T_vec[n]] = X_tmp
        Sigma_prop_int[:, :, np.sum(T_vec[:n]): np.sum(T_vec[:n]) + T_vec[n]] = Sigma_prop_tmp
        Sigma_int[:, np.sum(T_vec[:n]): np.sum(T_vec[:n]) + T_vec[n]] = Sigma_tmp

    Eul_int = quat2eul(X_int[3:7])
    return X_int, Eul_int, Sigma_prop_int, Sigma_int
