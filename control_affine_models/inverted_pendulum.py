import warnings
from contextlib import contextmanager
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
import pickle
import pysindy as ps

from utils import *

# Seed the random number generators for reproducibility
#np.random.seed(100)

# Set up simulation parameters
time_horzn = 1.0
dt = 0.01
ang_ind = [0]

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-9
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-9

# Randomized initial condition
def x0_fun(): return [np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)]

def x0_zero(): return [0.0, 0.0]

# Range of the amplitudes and frequencies of the randomized sine inputs
u_amp_range = [0, 10]
u_freq_range = [0, 5]

# Model parameters {"M": 1.0, "m": 1.0, "L": 0.5, "Kd": 10.0}
m = 1 # pendulum mass
L = 1 # length
b = 0.01 # friction coeff
g = 9.81

def ip_dyn(state, u):
    # Inverted pendulum
    # x_dot = f(x, u)

    theta, theta_dot = state
    theta_ddot = (-b*theta_dot + m*g*L*np.sin(theta)/2 ) / (m*L**2/3) - 1/(m*L**2/3) * u
    
    return [theta_dot, theta_ddot]

def ip(t, state, u_fun):
    # Inverted pendulum
    u = u_fun(t)
    return ip_dyn(state, u)

## Train a SINDYc model using trajectory data
# Generate the training dataset
t_data = np.arange(0, time_horzn, dt)
t_data_span = (t_data[0], t_data[-1])
n_traj_train = 1000
n_traj_zero = 100

x_train, x_dot_train, u_train = gen_trajectory_dataset(ip, x0_fun, n_traj_train, time_horzn, dt, 
                                          u_amp_range, u_freq_range, ang_ind, **integrator_keywords)

x_zero, x_dot_zero, u_zero = gen_trajectory_dataset(ip, x0_zero, n_traj_zero, time_horzn, dt, 
                                          [0.0, 0.0], [0.0, 0.0], ang_ind, **integrator_keywords)

x_train = [*x_train, *x_zero]
x_dot_train = [*x_dot_train, *x_dot_zero]
u_train = [*u_train, *u_zero]

#plt.plot(t_data, x_train[0])
#plt.show()
#plt.plot(t_data, x_train[0][:,1], t_data, x_dot_train[0][:,0])
#plt.show()
#plt.plot(t_data, x_train[0][:,3], t_data, x_dot_train[0][:,2])
#plt.show()
#plt.plot(t_data, u_train[0])
#plt.show()

# Instantiate and fit the SINDYc model
# Generalized Library (such that it's control affine)
generalized_library = ps.GeneralizedLibrary(
    [ps.PolynomialLibrary(degree = 2),
     #ps.FourierLibrary(n_frequencies = 1),
     #ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1),
     ps.IdentityLibrary() # for control input
    ],
    tensor_array = [[1,1]],
    inputs_per_library = [[0,1], [2]]
)

# Unconstrained model
model_uc = ps.SINDy(
    optimizer = ps.STLSQ(threshold = 0.01),
    feature_library = generalized_library,
)
model_uc.fit(x_train, x_dot = x_dot_train, u = u_train, t = dt)
model_uc.print()
print("Feature names:\n", model_uc.get_feature_names())

model = model_uc

control_affine = check_control_affine(model)
assert control_affine is True

model = set_derivative_coeff(model, [0], [1]) #x2 (resp. x3) is the time deriv of x0 (resp. x1)
model.print()

## Assess results on a test trajectory
# Evolve the equations in time using a different initial condition
x0 = x0_fun()
u_amp_data = np.random.uniform(0, 100)
u_freq_data = np.random.uniform(0, 5)
u_fun = lambda t: u_amp_data * np.sin(2 * np.pi * u_freq_data * t)
test_model_prediction(ip, model, x0, u_fun, time_horzn, dt, ang_ind, **integrator_keywords)

x0 = x0_zero()
u_zero = lambda t: 0.0 * t
test_model_prediction(ip, model, x0, u_zero, time_horzn, dt, ang_ind, **integrator_keywords)

## Compute conformal prediction quantile
theta_max = np.pi/4
theta_dot_max = 1.0
x_norm = [theta_max, theta_dot_max]

x_range = np.array([
     [-theta_max, theta_max],
     [-theta_dot_max, theta_dot_max]
])

u_range = np.array([
     [-5.0, 5.0]
])

alpha = 0.05
n_cal = 1000
n_val = 1000
norm = 2

quantile = get_conformal_prediction_quantile(ip_dyn, model, x_range, u_range,
                                      n_cal, n_val, alpha, norm)

# Save the quantile and alpha as paramters under the model
model_error = {"alpha": alpha, "quantile": quantile, "norm": norm, "normalization": x_norm}

model_saved = {"feature_names": model.get_feature_names(), "coefficients": model.optimizer.coef_, "model_error": model_error}

## Save the model and dataset
with open('./control_affine_models/saved_models/model_inverted_pendulum_sindy', 'wb') as file:
    pickle.dump(model_saved, file)
 
# Testing
with open('./control_affine_models/saved_models/' + 'model_inverted_pendulum_sindy', 'rb') as file:
	model2 = pickle.load(file)