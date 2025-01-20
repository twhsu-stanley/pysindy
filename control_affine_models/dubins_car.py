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
ang_ind = [2] # theta_

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-9
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-9

# Randomized initial condition
def x0_fun(): return [0.0, 0.0, np.random.uniform(-np.pi, np.pi)]

# Range of the amplitudes and frequencies of the randomized sine inputs
u_amp_range = [0, np.pi]
u_freq_range = [0, 5]

# Model parameter
v = 1.0

def dubins_car_dyn(state, u):
    # x_dot = f(x, u)

    x_, y_, theta_ = state

    x_dot = v * np.cos(theta_)
    y_dot = v * np.sin(theta_)
    theta_dot = u
    
    return [x_dot, y_dot, theta_dot]

def dubins_car(t, state, u_fun):
    u = u_fun(t)
    return dubins_car_dyn(state, u)

## Train a SINDYc model using trajectory data
# Generate the training dataset
t_data = np.arange(0, time_horzn, dt)
t_data_span = (t_data[0], t_data[-1])
n_traj_train = 5000

x_train, x_dot_train, u_train = gen_trajectory_dataset(dubins_car, x0_fun, n_traj_train, time_horzn, dt, 
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
# Generalized Library (such that it's control affine)
generalized_library = ps.GeneralizedLibrary(
    [ps.PolynomialLibrary(degree = 20),
     #ps.FourierLibrary(n_frequencies = 1),
     ps.IdentityLibrary() # for control input
    ],
    #tensor_array = [[0,1,0]],
    inputs_per_library = [[2], [3]]
)

# Unconstrained model
model_uc = ps.SINDy(
    optimizer = ps.STLSQ(threshold = 0.0001),
    feature_library = generalized_library,
)
model_uc.fit(x_train, x_dot = x_dot_train, u = u_train, t = dt)
model_uc.print()
print("Feature names:\n", model_uc.get_feature_names())

model = model_uc

control_affine = check_control_affine(model)
assert control_affine is True

## Assess results on a test trajectory
# Evolve the equations in time using a different initial condition
x0 = x0_fun()
u_amp_data = np.random.uniform(0, 100)
u_freq_data = np.random.uniform(0, 5)
u_fun = lambda t: u_amp_data * np.sin(2 * np.pi * u_freq_data * t)
test_model_prediction(dubins_car, model, x0, u_fun, time_horzn, dt, ang_ind, **integrator_keywords)
u_amp_data = np.random.uniform(0, 100)
u_freq_data = np.random.uniform(0, 5)
test_model_prediction(dubins_car, model, x0, u_fun, time_horzn, dt, ang_ind, **integrator_keywords)
u_amp_data = np.random.uniform(0, 100)
u_freq_data = np.random.uniform(0, 5)
test_model_prediction(dubins_car, model, x0, u_fun, time_horzn, dt, ang_ind, **integrator_keywords)

## Compute conformal prediction quantile
x_max = 10.0
y_max = 10.0
theta_max = np.pi

x_range = np.array([
     [-x_max, x_max],
     [-y_max, y_max],
     [-theta_max, theta_max]
])

u_range = np.array([
     [-np.pi, np.pi]
])

alpha = 0.05
n_cal = 1000
n_val = 1000
norm = 2

quantile = get_conformal_prediction_quantile(dubins_car_dyn, model, x_range, u_range,
                                      n_cal, n_val, alpha, norm)

# Save the quantile and alpha as paramters under the model
model_error = {"alpha": alpha, "quantile": quantile, "norm": norm}
model.model_error = model_error

## Save the model and dataset
with open('./control_affine_models/saved_models/model_dubins_car_sindy', 'wb') as file:
    dill.dump(model, file)
 
# Testing
with open('./control_affine_models/saved_models/' + 'model_dubins_car_sindy', 'rb') as file:
	model2 = dill.load(file)