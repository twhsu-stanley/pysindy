import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg' if you installed PyQt5
import matplotlib.pyplot as plt

def wrapToPi(ang):
    """maps ang to [-pi, pi)"""
    ang = (ang + np.pi) % (2 * np.pi) - np.pi
    return ang

def generate_samples(data_range, num_samples):
    """Generates non-repeating samples within the specified ranges"""

    data_dim = data_range.shape[0]

    samples = np.random.uniform(low = [data_range[i][0] for i in range(data_dim)], 
                                high = [data_range[i][1] for i in range(data_dim)], 
                                size = (num_samples, data_dim))
    return samples

def gen_single_trajectory(dynamical_system, x0, u_fun, time_horzn, dt, ang_ind = [], **integrator_keywords):
    """Generate a single trajectory"""
    # dynamical_system: dynamical system
    # x0: deterministic initial condition
    # u_fun: control input (a function of time)
    
    time = np.arange(0, time_horzn, dt)
    time_span = (time[0], time[-1])

    # Simulate a single trajectory
    x_traj = solve_ivp(
        dynamical_system,
        time_span,
        x0,
        t_eval = time,
        args = (u_fun,),
        **integrator_keywords,
    ).y.T

    x_dot_traj = np.gradient(x_traj, dt, axis = 0)

    if len(ang_ind) > 0:
        for i in ang_ind:
            x_traj[:,i] = wrapToPi(x_traj[:,i])

    return x_traj, x_dot_traj, u_fun(time)

def test_model_prediction(dynamical_system, model, x0, u_fun, time_horzn, dt, ang_ind = [], **integrator_keywords):
    ## Assess results on a test trajectory
    # Evolve the equations in time using a different initial condition
    t_test = np.arange(0, time_horzn, dt)
    
    x_test, x_dot_test, u_test = gen_single_trajectory(dynamical_system, x0, u_fun, time_horzn, dt, ang_ind, **integrator_keywords)

    # Predict derivatives using the learned model
    x_dot_test_predicted = model.predict(x_test, u = u_test)

    # Compare SINDy-predicted derivatives with numerical derivatives
    #print("RMSE = %f" % np.sqrt(np.mean(np.square(x_dot_test_predicted[:, i] - x_dot_test[:, i]))))

    # Compute derivatives with a finite difference method, for comparison
    #x_dot_test_computed = model.differentiate(x_test, t = dt)
    x_dot_test_computed = x_dot_test

    fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
    for i in range(x_test.shape[1]):
        axs[i].plot(t_test, x_dot_test_computed[:, i], "k", label="numerical derivative")
        axs[i].plot(t_test, x_dot_test_predicted[:, i], "r--", label="model prediction")
        axs[i].legend()
        axs[i].set(xlabel="t", ylabel=r"$\dot x_{}$".format(i))
    fig.show()
    fig2, axs2 = plt.subplots(x_test.shape[1] + 1, 1, sharex=True, figsize=(7, 9))
    for i in range(x_test.shape[1]):
        axs2[i].plot(t_test, x_test[:, i], "k", label="ground truth state")
        axs2[i].legend()
        axs2[i].set(xlabel="t", ylabel=r"$x_{}$".format(i))
    axs2[x_test.shape[1]].plot(t_test, u_test, "k", label="control input")
    axs2[x_test.shape[1]].legend()
    axs2[x_test.shape[1]].set(xlabel="t", ylabel="u")
    fig2.show()

def plot_prediction(model, x_traj, u_traj, x_dot_traj, dt):
    "Plot actual x_dot data versus predicted x_dot data"
    time_steps = x_traj.shape[0]
    assert time_steps == u_traj.shape[0]
    assert time_steps == x_dot_traj.shape[0]

    x_dim = x_traj.shape[1]
    u_dim = u_traj.shape[1]

    x_dot_sindy = np.zeros((time_steps,x_dim))
    for t in range(time_steps):
        x_dot_sindy[t,:] = model.predict(x_traj[t,:].reshape(1,x_dim), u = u_traj[t,:].reshape(1,u_dim))

    fig, axs = plt.subplots(x_traj.shape[1], 1)
    time = np.arange(time_steps) * dt
    for i in range(x_traj.shape[1]):
        axs[i].plot(time, x_dot_sindy[:, i], "b", label="model prediction")
        axs[i].plot(time, x_dot_traj[:, i], "r--", label="actual data")
        axs[i].legend()
        axs[i].set(xlabel="time (s)", ylabel=r"$\dot x_{}$".format(i))
    fig.show()

def check_control_affine(model):
    """Check if the model is in the control affine form"""
    # TODO: read dim of Xt and Ut from model
    # TODO: Currently only works for dim(U) = 1?
    Xt = np.random.rand(1, model.n_features_in_-1)
    Ut = np.random.rand(1, model.n_control_features_) # TODO: or (model.n_control_features_, 1)
    #Xt = np.array([[0.0, 2.0, 0.5, 0.2, 3.0, 0.5]])
    #Ut = np.array([[5.0]])
    Theta = model.get_regressor(Xt, u = np.ones((1, model.n_control_features_))) # TODO: or (model.n_control_features_, 1)
    coeff = model.optimizer.coef_
    feature_names = model.get_feature_names()
    idx_x = [] # Indices for f(x)
    idx_u = [] # Indices for g(x)*u
    for i in range(len(feature_names)):
        if 'u0' in feature_names[i]:
            idx_u.append(i)
        else:
            idx_x.append(i)

    # Get f(x)
    f_of_x = Theta[:,idx_x] @ coeff[:,idx_x].T

    # Get g(x)
    g_of_x = Theta[:,idx_u] @ coeff[:,idx_u].T

    Err = abs(f_of_x + g_of_x * Ut - model.get_regressor(Xt, Ut) @ coeff.T)

    control_affine = True
    for i in range(len(Err[0])):
        if Err[0][i] > 1e-2:
            print("f_of_x + g_of_x * Ut = ", f_of_x + g_of_x * Ut)
            print("Theta @ coeff.T = ", model.get_regressor(Xt, Ut) @ coeff.T)
            control_affine = control_affine and False
            raise ValueError("f_of_x + g_of_x * Ut != Theta @ coeff.T; Check if the model is control affine")
    
    return control_affine

def set_derivative_coeff(model, idx, idx_d):
    """If a state x[idx_d] is known to be the time derivative of another state x[idx], 
       set all its coefficients to zeros except for the one corresponds to x1"""
    assert len(idx) == len(idx_d)
    # Example: idx = [0,2] and idx_d = [1,3] means x1 (resp. x3) is the time deriv of x0 (resp. x2)
    
    coeff = model.optimizer.coef_
    feature_names = model.get_feature_names()
    
    for i in range(len(idx)):  
        k = feature_names.index('x' + str(idx_d[i]))
        for j in range(len(feature_names)):
            if j == k:
                coeff[idx[i], j] = 1.0
            else:
                coeff[idx[i], j] = 0.0

    model.optimizer.coef_ = coeff

    return model

def get_conformal_quantile(model,
                           x_cal, u_cal, x_dot_cal, x_val, u_val, x_dot_val,
                           alpha = 0.05, norm = 2,
                           normalization = None):
    """Get conformal prediction quantile using randomly sampled data"""
    # dynamical_system: true model
    # model: SINDy model

    x_dim = x_cal.shape[1]
    u_dim = u_cal.shape[1]
    assert x_dim == x_val.shape[1]
    assert u_dim == u_val.shape[1]

    num_samples_cal = x_cal.shape[0]
    num_samples_val = x_val.shape[0]

    if normalization is not None:
        norm_inv = [norm ** -1 for norm in normalization]
        Tx_inv = np.diag(norm_inv)

    # Compute non-conformity scores
    nc_scores = []
    for i in range(num_samples_cal):
        # x_dot by the SINDy model
        Theta = model.get_regressor(x_cal[i,:].reshape(1,x_dim), u_cal[i,:].reshape(1,u_dim))
        coeff = model.optimizer.coef_
        x_dot_sindy = Theta @ coeff.T

        # Compute modeling error (non-conformity score)
        if normalization is not None:
            R = np.linalg.norm(Tx_inv @ (x_dot_sindy[0][:] - x_dot_cal[i,:]), norm)
        else:
            R = np.linalg.norm(x_dot_sindy[0][:] - x_dot_cal[i,:], norm)
        nc_scores.append(R)

    # Compute the quantile
    n = len(nc_scores)
    quantile = np.quantile(nc_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher")
    print("Quantile for alpha = %5.3f is %5.3f" % (alpha, quantile))

    # Compute the empirical coverage
    emp_scores = []
    for i in range(num_samples_val):
        # x_dot by the SINDy model
        Theta = model.get_regressor(x_val[i,:].reshape(1,x_dim), u_val[i,:].reshape(1,u_dim))
        coeff = model.optimizer.coef_
        x_dot_sindy = Theta @ coeff.T

        # Compute modeling error (non-conformity score)
        if normalization is not None:
            R = np.linalg.norm(Tx_inv @ (x_dot_sindy[0][:] - x_dot_val[i,:]), norm)
        else:
            R = np.linalg.norm(x_dot_sindy[0][:] - x_dot_val[i,:], norm)
        emp_scores.append(R)

    emp_coverage = sum(k < quantile for k in emp_scores) / len(emp_scores)
    print("Empirical Coverage = %5.3f vs. 1-alpha = %5.3f" % (emp_coverage, 1- alpha))

    return quantile

def get_conformal_traj_quantile(model,
                                x_cal, u_cal, x_dot_cal, x_val, u_val, x_dot_val,
                                alpha = 0.05, norm = 2,
                                normalization = None):
    """Get conformal prediction quantile using randomly sampled data"""
    # dynamical_system: true model
    # model: SINDy model

    x_dim = x_cal.shape[2]
    u_dim = u_cal.shape[2]
    assert x_dim == x_val.shape[2]
    assert u_dim == u_val.shape[2]

    num_traj_cal = x_cal.shape[0]
    num_traj_val = x_val.shape[0]

    time_steps = x_cal.shape[1]
    assert time_steps == u_cal.shape[1]
    assert time_steps == x_dot_cal.shape[1]

    if normalization is not None:
        norm_inv = [norm ** -1 for norm in normalization]
        Tx_inv = np.diag(norm_inv)

    # Compute non-conformity scores
    nc_scores = np.zeros(num_traj_cal)
    for i in range(num_traj_cal):
        traj_scores = np.zeros(time_steps)
        for t in range(time_steps):
            # x_dot by the SINDy model
            #Theta = model.get_regressor(x_cal[i,t,:].reshape(1,x_dim), u_cal[i,t,:].reshape(1,u_dim))
            #coeff = model.optimizer.coef_
            #x_dot_sindy = Theta @ coeff.T
            x_dot_sindy = model.predict(x_cal[i,t,:].reshape(1,x_dim), u = u_cal[i,t,:].reshape(1,u_dim))

            # Compute modeling error of each time step
            if normalization is not None:
                R = np.linalg.norm(Tx_inv @ (x_dot_sindy[0][:] - x_dot_cal[i,t,:]), norm)
            else:
                R = np.linalg.norm(x_dot_sindy[0][:] - x_dot_cal[i,t,:], norm)
            traj_scores[t] = R
            
        # Compute the sup over all time steps
        nc_scores[i] = np.max(traj_scores)

    # Compute the quantile
    n = num_traj_cal
    quantile = np.quantile(nc_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher")
    print("Quantile for alpha = %5.3f is %5.3f" % (alpha, quantile))

    # Compute the empirical coverage
    emp_score = np.zeros(num_traj_val)
    for i in range(num_traj_val):
        for t in range(time_steps):
            # x_dot by the SINDy model
            #Theta = model.get_regressor(x_val[i,t,:].reshape(1,x_dim), u_val[i,t,:].reshape(1,u_dim))
            #coeff = model.optimizer.coef_
            #x_dot_sindy = Theta @ coeff.T
            x_dot_sindy = model.predict(x_val[i,t,:].reshape(1,x_dim), u = u_val[i,t,:].reshape(1,u_dim))

            # Compute modeling error (non-conformity score)
            if normalization is not None:
                R = np.linalg.norm(Tx_inv @ (x_dot_sindy[0][:] - x_dot_val[i,t,:]), norm)
            else:
                R = np.linalg.norm(x_dot_sindy[0][:] - x_dot_val[i,t,:], norm)

            # Compute the sup over all time steps
            if R > emp_score[i]:
                emp_score[i] = R

    emp_coverage = sum(k < quantile for k in emp_score) / len(emp_score)
    print("Empirical Coverage = %5.3f vs. 1-alpha = %5.3f" % (emp_coverage, 1- alpha))

    return quantile
