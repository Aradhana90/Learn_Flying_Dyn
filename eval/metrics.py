import numpy as np
from scipy.integrate import solve_ivp
from data.mechanics import quatmul, quat2eul


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
    err = np.linalg.norm(y_tmp - y_pred_tmp, ord=2, axis=0)
    err = np.sqrt(err.sum() / len(err))

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
    for ii in range(3):
        if eul1[ii] < 0 and eul2[ii] > 0:
            diff[ii] = min(diff[ii], abs(eul1[ii] + 360 - eul2[ii]))
        elif eul1[ii] > 0 and eul2[ii] < 0:
            diff[ii] = min(diff[ii], abs(eul1[ii] - eul2[ii] - 360))

    # Compute norm
    err = np.linalg.norm(diff, ord=2)
    # err = np.sqrt(err.sum() / len(err))

    return err


def dx_dt(t, x, model, only_pos=True, ang_vel=False, estimator='svr'):
    """
    :param t:           Time
    :param x:           Current state
    :param model:       List of 1D svr models or a GPR mode to predict the acceleration
    :param sys_dim:     Dimension of the system (3 for position only, 6 or 7 for position + velocity)
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
        if estimator == 'svr':
            for ii in range(sys_dim):
                ddxi_dt[ii] = model[ii].predict(np.expand_dims(x[3:], axis=0))
        elif estimator == 'gpr':
            ddxi_dt = model.predict(np.expand_dims(x[3:], axis=0)).reshape(-1)

    else:
        q = x[3:7]
        v = x[7:10]
        omega = x[10:13]
        dxi_dt = np.zeros(7)
        dxi_dt[0:3] = v
        dxi_dt[3:7] = 0.5 * quatmul(np.append([0], omega), q)
        ddxi_dt = np.zeros(6)
        if estimator == 'svr':
            for ii in range(6):
                ddxi_dt[ii] = model[ii].predict(np.expand_dims(x[3:], axis=0))
        elif estimator == 'gpr':
            ddxi_dt = model.predict(np.expand_dims(x[3:], axis=0)).reshape(-1)

    dx_dt = np.concatenate((dxi_dt, ddxi_dt))
    return dx_dt


def integrate_trajectory(model, x_init, t_eval, only_pos=True, ang_vel=False, estimator='svr'):
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
    sol = solve_ivp(dx_dt, t_span=(0, t_eval[-1]), y0=x_init, t_eval=t_eval, args=(model, only_pos, ang_vel, estimator))
    eul = quat2eul(sol.y[3:7])
    return sol.t, sol.y, eul


def integrate_trajectories(model, x_test, T_vec, only_pos=True, ang_vel=False, estimator='svr'):
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
    _, X_int, Eul_int = integrate_trajectory(model=model, x_init=x_init, t_eval=t_eval, only_pos=only_pos,
                                             ang_vel=ang_vel, estimator=estimator)
    N_tilde = len(T_vec)
    for n in range(1, N_tilde):
        x_init = x_test[:, np.sum(T_vec[:n])]
        t_eval = np.linspace(0, (T_vec[n] - 1) * 0.01, T_vec[n])
        _, X_tmp, Eul_tmp = integrate_trajectory(model=model, x_init=x_init, t_eval=t_eval, only_pos=only_pos,
                                                 ang_vel=ang_vel, estimator=estimator)
        X_int = np.concatenate((X_int, X_tmp), axis=1)
        Eul_int = np.concatenate((Eul_int, Eul_tmp), axis=1)

    return X_int, Eul_int
