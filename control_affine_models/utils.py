import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps
import matplotlib.pyplot as plt

def wrapTo2pi(ang):
    ang = ang % (2*np.pi)
    return ang

def gen_trajectory_dataset(dynamical_system, x0_fun, n_traj, time_horzn, dt, u_amp_range, u_freq_range, ang_ind = [], **integrator_keywords):
    """Generate a dataset of trajectories"""
    # dynamical_system: dynamical system
    # x0: function that generates radom initial conditions
    # u_amp_range: [u_amp min, u_amp max]
    # u_freq_range: [u_freq min, u_freq max]
    # ang_ind: indices of state variables that are angles (need to be wrapped to [0, 2*pi))

    time = np.arange(0, time_horzn, dt)
    time_span = (time[0], time[-1])

    x_data = []
    x_dot_data = []
    u_data = []

    for i in range(n_traj):
        # Random initial state distribution
        x0 = x0_fun()

        # Sinusoidal inputs with random amplitudes and frequencies
        u_amp = np.random.uniform(u_amp_range[0], u_amp_range[1])
        u_freq = np.random.uniform(u_freq_range[0], u_freq_range[1])
        u_fun = lambda t: u_amp * np.sin(2 * np.pi * u_freq * t)

        # Simulate the trajectory
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
                x_traj[:,i] = wrapTo2pi(x_traj[:,i])

        x_data.append(x_traj)
        x_dot_data.append(x_dot_traj)
        u_data.append(u_fun(time))

    return x_data, x_dot_data, u_data

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
            x_traj[:,i] = wrapTo2pi(x_traj[:,i])

    return x_traj, x_dot_traj, u_fun(time)

def test_model_prediction(dynamical_system, model, x0, u_fun, time_horzn, dt, ang_ind = [], **integrator_keywords):
    ## Assess results on a test trajectory
    # Evolve the equations in time using a different initial condition
    t_test = np.arange(0, time_horzn, dt)
    
    x_test, x_dot_test, u_test = gen_single_trajectory(dynamical_system, x0, u_fun, time_horzn, dt, ang_ind, **integrator_keywords)

    # Predict derivatives using the learned model
    x_dot_test_predicted = model.predict(x_test, u = u_test[0])

    # Compare SINDy-predicted derivatives with numerical derivatives
    print("RMSE = %f" % np.sqrt(np.mean(np.square(x_dot_test_predicted - x_dot_test))))

    # Compute derivatives with a finite difference method, for comparison
    #x_dot_test_computed = model.differentiate(x_test, t = dt)
    x_dot_test_computed = x_dot_test

    fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
    for i in range(x_test.shape[1]):
        axs[i].plot(t_test, x_dot_test_computed[:, i], "k", label="numerical derivative")
        axs[i].plot(t_test, x_dot_test_predicted[:, i], "r--", label="model prediction")
        axs[i].legend()
        axs[i].set(xlabel="t", ylabel=r"$\dot x_{}$".format(i))

    fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
    for i in range(x_test.shape[1]):
        axs[i].plot(t_test, x_test[:, i], "k", label="ground truth state")
        axs[i].legend()
        axs[i].set(xlabel="t", ylabel=r"$x_{}$".format(i))
    fig.show()

def check_control_affine(model):
    """Check if the model is in the control affine form"""
    # TODO: read dim of Xt and Ut from model
    model.n_features_in_
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
    for i in range(len(Err[0])):
        if Err[0][i] > 1e-2:
            print("f_of_x + g_of_x * Ut = ", f_of_x + g_of_x * Ut)
            print("Theta @ coeff.T = ", model.get_regressor(Xt, Ut) @ coeff.T)
            raise ValueError("f_of_x + g_of_x * Ut != Theta @ coeff.T; Check if the model is control affine")
