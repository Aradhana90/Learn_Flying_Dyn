import numpy as np
from scipy.integrate import solve_ivp


def get_rmse(y, y_pred):
    out_dim = 3
    # Ensure that both y and ypred have shape (out_dim, n_test_points)
    if y.shape[1] == out_dim:
        y.reshape(3, y.shape[0])

    if y_pred.shape[1] == out_dim:
        y_pred.reshape(3, y_pred.shape[0])

    if y.shape[1] != y_pred.shape[1]:
        raise SystemExit('Same number of predictions and targets needed!')

    # Compute RMSE
    err = np.linalg.norm(y - y_pred, ord=2, axis=0)
    err = err.sum() / len(err)

    return err


def dx_dt(t, x, model, sys_dim=3, estimator='svr'):
    """
    :param t:           Time
    :param x:           Current state
    :param model:       List of 1D svr models or a GPR mode to predict the acceleration
    :param sys_dim:     Dimension of the system (3 for position only, 6 or 7 for position + velocity)
    :param estimator:   Specifies the type of the the learned model ('svr' or 'gpr')
    :return:            Predicted linear (and angular) acceleration
    """
    dxi_dt = x[sys_dim:]
    ddxi_dt = np.zeros(sys_dim)
    if estimator == 'svr':
        for ii in range(sys_dim):
            ddxi_dt[ii] = model[ii].predict(np.expand_dims(x[3:], axis=0))
    elif estimator == 'gpr':
        ddxi_dt = model.predict(np.expand_dims(x[3:], axis=0)).reshape(-1)

    dx_dt = np.concatenate((dxi_dt, ddxi_dt))
    return dx_dt


def integrate_trajectory(model, x_init, t_eval, sys_dim=3, estimator='svr'):
    """
    :param model:       SVR or GP models to predict the linear and angular acceleration for the current state
    :param x_init:      Initial state from which to start the numerical integration
    :param t_eval:      Timesteps for which to store the integration results
    :param estimator:   Specifies what kind of predictor is used, 'SVR' or 'GP'
    :param sys_dim:     Dimension of the system (3 for position only, 6 or 7 for position + velocity)
    :return:            Final state after t_end seconds
    """
    sol = solve_ivp(dx_dt, t_span=(0, t_eval[-1]), y0=x_init, t_eval=t_eval, args=(model, sys_dim, estimator))
    return sol.t, sol.y
