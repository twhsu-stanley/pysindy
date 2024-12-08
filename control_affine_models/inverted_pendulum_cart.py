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
np.random.seed(100)

# Set up simulation parameters
time_horzn = 2.0
dt = 0.001
ang_ind = [2]

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-9
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-9

# Randomized initial condition
def x0_fun(): return [0.0, np.random.uniform(-10, 10), 
                      np.random.uniform(-np.pi/2, np.pi/2), np.random.uniform(-np.pi*2, np.pi*2)]

# Range of the amplitudes and frequencies of the randomized sine inputs
u_amp_range = [0, 100]
u_freq_range = [0, 5]

# Model parameters {"M": 1.0, "m": 1.0, "L": 0.5, "Kd": 10.0}
M = 1.0
m = 1.0
L = 0.5
Kd = 10.0
g = 9.80665

def inverted_pendulum_cart(t, state, u_fun):
    # Single inverted pendulum on a cart (2-DOF)

    z, z_dot, theta, theta_dot = state

    F = u_fun(t)

    z_ddot = (F - Kd*z_dot - m*L*theta_dot**2*np.sin(theta) + m*g*np.sin(theta)*np.cos(theta)) / (M + m*np.sin(theta)**2)
    
    theta_ddot = (g*np.sin(theta) + (F - Kd*z_dot - m*L*theta_dot**2*np.sin(theta))*np.cos(theta)/(M + m)) / (L - m*L*np.cos(theta)**2/(M + m))
          
    return [z_dot, z_ddot, theta_dot, theta_ddot]

## Train a SINDYc model using trajectory data
# Generate the training dataset
t_data = np.arange(0, time_horzn, dt)
t_data_span = (t_data[0], t_data[-1])
n_traj_train = 1000

x_train, x_dot_train, u_train = gen_trajectory_dataset(inverted_pendulum_cart, x0_fun, n_traj_train, time_horzn, dt, 
                                          u_amp_range, u_freq_range, ang_ind, **integrator_keywords)

#plt.plot(t_data, x_train[0])
#plt.show()
#plt.plot(t_data, x_train[0][:,1], t_data, x_dot_train[0][:,0])
#plt.show()
#plt.plot(t_data, x_train[0][:,3], t_data, x_dot_train[0][:,2])
#plt.show()
#plt.plot(t_data, u_train[0])
#plt.show()

# Instantiate and fit the SINDYc model
# Generalized Library
# Initialize the generalized library such that it's control affine
generalized_library = ps.GeneralizedLibrary(
    [ps.PolynomialLibrary(degree = 2),
     ps.FourierLibrary(n_frequencies = 2),
     ps.FourierLibrary(n_frequencies = 2) * ps.FourierLibrary(n_frequencies = 2),
     #ps.FourierLibrary(n_frequencies = 2) * ps.FourierLibrary(n_frequencies = 2) * ps.FourierLibrary(n_frequencies = 2),
     ps.IdentityLibrary() # for control input
    ],
    tensor_array = [[0,1,0,1], [0,0,1,1], [1,1,0,0], [1,0,1,0]],
    #tensor_array = [[0,1,0,0,1], [0,0,1,0,1], [0,0,0,1,1], [1,1,0,0,0], [1,0,1,0,0], [1,0,0,1,0]],
    inputs_per_library = [[1,3], [2], [2], [4]]
)

# Unconstrained model
model_uc = ps.SINDy(
    optimizer = ps.STLSQ(threshold = 0.01),
    feature_library = generalized_library,
)
model_uc.fit(x_train, x_dot = x_dot_train, u = u_train, t = dt)
model_uc.print()
print("Feature names:\n", model_uc.get_feature_names())

# Fit the model with constraints at equilibrium points
n_features = model_uc.n_output_features_
constraint_lhs = np.zeros((4, 4 * n_features))
Theta_0 = model_uc.get_regressor(np.zeros((1,4)), u = np.array([[0.0]]))
constraint_lhs[0, :n_features] = Theta_0
constraint_lhs[1, n_features:2*n_features] = Theta_0
constraint_lhs[2, 2*n_features:3*n_features] = Theta_0
constraint_lhs[3, 3*n_features:4*n_features] = Theta_0

optimizer_cnstr = ps.ConstrainedSR3(
    constraint_rhs = np.zeros((4,)),
    constraint_lhs = constraint_lhs,
    equality_constraints = True
)
model = ps.SINDy(
    optimizer = optimizer_cnstr,
    feature_library = generalized_library,
)
model.fit(x_train, x_dot = x_dot_train, u = u_train, t = dt)
model.print()
print("Feature names:\n", model.get_feature_names())

# Verify the constraints
coeff = model.optimizer.coef_
Theta = model.get_regressor(np.zeros((1,4)), u = np.array([[0.0]]))
#assert np.all(abs(Theta @ coeff.T - np.zeros((4,))) < 1e-1)

control_affine = check_control_affine(model)
assert control_affine is True

## Assess results on a test trajectory
# Evolve the equations in time using a different initial condition
x0 = x0_fun()
u_amp_data = np.random.uniform(0, 100)
u_freq_data = np.random.uniform(0, 5)
u_fun = lambda t: u_amp_data * np.sin(2 * np.pi * u_freq_data * t)
test_model_prediction(inverted_pendulum_cart, model, x0, u_fun, time_horzn, dt, ang_ind, **integrator_keywords)

## Test Conformal Prediction
n_traj_cal = 100
n_traj_val = 100
alpha = 0.05

quantile = test_conformal_prediction(inverted_pendulum_cart, model, x0_fun, time_horzn, dt, u_amp_range, u_freq_range,
                            ang_ind, n_traj_cal, n_traj_val, alpha, **integrator_keywords)

# Save the quantile and alpha as paramters under the model
model_error = {"alpha": alpha, "quantile": quantile}
model.model_error = model_error

## Save the model and dataset
with open('./control_affine_models/saved_models/model_inverted_pendulum_cart_sindy', 'wb') as file:
    dill.dump(model, file)
 
# Testing
with open('./control_affine_models/saved_models/' + 'model_inverted_pendulum_cart_sindy', 'rb') as file:
	model2 = dill.load(file)