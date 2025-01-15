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
import dill
import pysindy as ps

from utils import *

# Seed the random number generators for reproducibility
#np.random.seed(100)

# Set up simulation parameters
time_horzn = 1.0
dt = 0.01
ang_ind = [1]

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-9
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-9

# Randomized initial condition
def x0_fun(): return [0.0, np.random.uniform(-10, 10), 
                      np.random.uniform(-np.pi/2, np.pi/2), np.random.uniform(-np.pi*2, np.pi*2)]

def x0_zero(): return [0.0, 0.0, 0.0, 0.0]

# Range of the amplitudes and frequencies of the randomized sine inputs
u_amp_range = [0, 100]
u_freq_range = [0, 5]

# Model parameters {"M": 1.0, "m": 1.0, "L": 0.5, "Kd": 10.0}
m = 0.3 # pendulum mass
M = 1 # cart mass
L = 1 # length
b = 0 # friction coeff
g = 9.81

def cartpole_dyn(state, u):
    # Cart-pole (single inverted pendulum on a cart, 2-DOF)
    # x_dot = f(x, u)

    z, theta, z_dot, theta_dot = state
    F = u

    det = M + m * (np.sin(theta)**2)
    z_ddot = (F - b * z_dot - m * L * (theta_dot**2) * np.sin(theta)  + 0.5 * m * g * np.sin(2 * theta)) / det
    theta_ddot = (F * np.cos(theta) - 0.5 * m * L * (theta_dot**2) * np.sin(2 * theta) - b * (z_dot * np.cos(theta))
                + (m + M) * g * np.sin(theta)) / (det * L)
    
    return [z_dot, theta_dot, z_ddot, theta_ddot]

def cartpole(t, state, u_fun):
    # Cart-pole (single inverted pendulum on a cart, 2-DOF)
    u = u_fun(t)
    return cartpole_dyn(state, u)

## Train a SINDYc model using trajectory data
# Generate the training dataset
t_data = np.arange(0, time_horzn, dt)
t_data_span = (t_data[0], t_data[-1])
n_traj_train = 1000#3000
n_traj_zero = 100#00

x_train, x_dot_train, u_train = gen_trajectory_dataset(cartpole, x0_fun, n_traj_train, time_horzn, dt, 
                                          u_amp_range, u_freq_range, ang_ind, **integrator_keywords)

x_zero, x_dot_zero, u_zero = gen_trajectory_dataset(cartpole, x0_zero, n_traj_zero, time_horzn, dt, 
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
     ps.FourierLibrary(n_frequencies = 1),
     ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1),
     ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1),
     #ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1) * ps.FourierLibrary(n_frequencies = 1),
     ps.IdentityLibrary() # for control input
    ],
    tensor_array = [[0,1,0,0,1], [0,0,1,0,1], [0,0,0,1,1], [1,1,0,0,0], [1,0,1,0,0], [1,0,0,1,0]],
    inputs_per_library = [[2,3], [1], [1], [1], [4]]
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

model = set_derivative_coeff(model, [0,1], [2,3]) #x2 (resp. x3) is the time deriv of x0 (resp. x1)
model.print()

## Assess results on a test trajectory
# Evolve the equations in time using a different initial condition
x0 = x0_fun()
u_amp_data = np.random.uniform(0, 100)
u_freq_data = np.random.uniform(0, 5)
u_fun = lambda t: u_amp_data * np.sin(2 * np.pi * u_freq_data * t)
test_model_prediction(cartpole, model, x0, u_fun, time_horzn, dt, ang_ind, **integrator_keywords)

x0 = x0_zero()
u_zero = lambda t: 0.0 * t
test_model_prediction(cartpole, model, x0, u_zero, time_horzn, dt, ang_ind, **integrator_keywords)

## Compute conformal prediction quantile
z_max = 1.0
theta_max = np.pi/6
v_max = 1.5
omega_max = 1.0
x_norm = [z_max, theta_max, v_max, omega_max]

x_range = np.array([
     [-z_max, z_max],
     [-theta_max, theta_max],
     [-v_max, v_max],
     [-omega_max, omega_max]
])

u_range = np.array([
     [-10.0, 10.0]
])

alpha = 0.05
n_cal = 1000
n_val = 1000
norm = 2

quantile = get_conformal_prediction_quantile(cartpole_dyn, model, x_range, u_range,
                                      n_cal, n_val, alpha, norm,
                                      normalization = x_norm)

# Save the quantile and alpha as paramters under the model
model_error = {"alpha": alpha, "quantile": quantile, "norm": norm, "normalization": x_norm}
model.model_error = model_error

## Save the model and dataset
with open('./control_affine_models/saved_models/model_cartpole_sindy_coarse', 'wb') as file:
    dill.dump(model, file)
 
# Testing
with open('./control_affine_models/saved_models/' + 'model_cartpole_sindy_coars', 'rb') as file:
	model2 = dill.load(file)