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

t_end_train = 5
t_end_test = 5

# Seed the random number generators for reproducibility
np.random.seed(100)

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-9
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-9

# ## SINDy with control (SINDYc)
# Here we learn a inverted penduum model:
# $$ \dot{theta} = theta_dot $$
# $$ \dot{theta_dot} = g/L * sin(theta) -b/(m*L^2) * theta_dot + 1/(m*L^2) * u $$
def inverted_pendulum(t, x, u_fun, L = 1, m = 1, b = 0.01):
    g = 9.81
    u = u_fun(t)
    return [
        x[1],
        g/L * np.sin(x[0]) -b/(m*L^2) * x[1] + 1/(m*L^2) * u,
    ]

## Generate Training Data
dt = 0.002
t_train = np.arange(0, t_end_train, dt)
t_train_span = (t_train[0], t_train[-1])
x_train = []
u_train = []
n_traj = 50
for i in range(n_traj):
    u_amp_train = np.random.uniform(0, 10)
    u_freq_train = np.random.uniform(0, 5)
    u_fun = lambda t: u_amp_train * np.sin(2 * np.pi * u_freq_train * t)
    x0_train = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
    x_traj = solve_ivp(
        inverted_pendulum,
        t_train_span,
        x0_train,
        t_eval = t_train,
        args = (u_fun,),
        **integrator_keywords,
    ).y.T
    x_train.append(x_traj)
    u_train.append(u_fun(t_train))

#plt.plot(t_train, x_train[0])
#plt.show()
#plt.plot(t_train, u_train[0])
#plt.show()

# Instantiate and fit the SINDYc model
# 1. Concatenate two libraries
"""
poly_library = ps.PolynomialLibrary(degree=1)
fourier_library = ps.FourierLibrary(n_frequencies=1)
combined_library = poly_library + fourier_library

model = ps.SINDy(
    optimizer = ps.STLSQ(threshold = 0.05),
    feature_library = combined_library,
)
model.fit(x_train, u=u_train, t=dt)
model.print()
print("Feature names:\n", model.get_feature_names())
"""
## 2. Generalized Library
#"""
# Initialize two libraries
poly_library_x = ps.PolynomialLibrary(degree = 1)
fourier_library_x = ps.FourierLibrary(n_frequencies = 1)
poly_library_u = ps.PolynomialLibrary(degree = 1)

# Initialize the generalized library such that it's control affine
generalized_library = ps.GeneralizedLibrary(
    [poly_library_x, fourier_library_x, poly_library_u],
    tensor_array = [[1,1,0], [1,0,1], [0,1,1]],
    #exclude_libraries = [0,1],
    inputs_per_library = [[0,1], [0,1], [2]],
)

model = ps.SINDy(
    optimizer = ps.STLSQ(threshold = 0.05),
    feature_library = generalized_library,
)
model.fit(x_train, u=u_train, t=dt)
model.print()
print("Feature names:\n", model.get_feature_names())
#"""

# TESTING: separate f(x) from g(x)*u ################################
Xt = np.array([[10.0,2.0]])
Ut = np.array([[3.0]])
Theta = model.get_regressor(Xt, u = Ut)
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
Theta_x = Theta[:,idx_x]
coeff_x = coeff[:,idx_x]
f_of_x = Theta[:,idx_x] @ coeff[:,idx_x].T

# Get g(x)
# Note: this trick exploits the control affine form
Theta_temp = model.get_regressor(Xt, u = np.array([[1.0]]))
Theta_u = Theta_temp[:,idx_u]
g_of_x = Theta_u @ coeff[:,idx_u].T

Err = abs(f_of_x + g_of_x * Ut - Theta @ coeff.T)
for i in range(len(Err[0])):
    if Err[0][i] > 1e-2:
        print("f_of_x + g_of_x * Ut = ", f_of_x + g_of_x * Ut)
        print("Theta @ coeff.T = ", Theta @ coeff.T)
        raise ValueError("f_of_x + g_of_x * Ut != Theta @ coeff.T; Check if the model is control affine")
#####################################################################

## Assess results on a test trajectory
# Evolve the equations in time using a different initial condition
t_test = np.arange(0, t_end_test, dt)
t_test_span = (t_test[0], t_test[-1])
u_amp_test = np.random.uniform(0, 10)
u_freq_test = np.random.uniform(0, 5)
u_fun = lambda t: u_amp_test * np.sin(2 * np.pi * u_freq_test * t)
x0_test = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
x_test = solve_ivp(
    inverted_pendulum,
    t_test_span,
    x0_test,
    t_eval = t_test,
    args = (u_fun,),
    **integrator_keywords,
).y.T

u_test = u_fun(t_test)

# Compare SINDy-predicted derivatives with finite difference derivatives
print("Model score: %f" % model.score(x_test, u = u_test, t=dt))

### Predict derivatives with learned model

# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(x_test, u = u_test)

# Compute derivatives with a finite difference method, for comparison
x_dot_test_computed = model.differentiate(x_test, t = dt)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_dot_test_computed[:, i], "k", label="numerical derivative")
    axs[i].plot(t_test, x_dot_test_predicted[:, i], "r--", label="model prediction")
    axs[i].legend()
    axs[i].set(xlabel="t", ylabel=r"$\dot x_{}$".format(i))
fig.show()

# ### Simulate forward in time (control input function known)
# When working with control inputs `SINDy.simulate` requires a *function* to be passed in for the control inputs, `u`, because the default integrator used in `SINDy.simulate` uses adaptive time-stepping. We show what to do in the case when you do not know the functional form for the control inputs in the example following this one.

# Evolve the new initial condition in time with the SINDy model
x_test_sim = model.simulate(x0_test, t_test, u = u_fun, integrator_kws = integrator_keywords)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i], "k", label="true simulation")
    axs[i].plot(t_test, x_test_sim[:, i], "r--", label="model simulation")
    axs[i].legend()
    axs[i].set(xlabel="t", ylabel="$x_{}$".format(i))

fig.show()

# ### Simulate forward in time (unknown control input function)
# If you only have a vector of control input values at the times in `t_test` and do not know the functional form for `u`, the `simulate` function will internally form an interpolating function based on the vector of control inputs. As a consequence of this interpolation procedure, `simulate` will not give a state estimate for the last time point in `t_test`. This is because the default integrator, `scipy.integrate.solve_ivp` (with LSODA as the default solver), is adaptive and sometimes attempts to evaluate the interpolant outside the domain of interpolation, causing an error.
u_test = u_fun(t_test)
x_test_sim = model.simulate(x0_test, t_test, u = u_test, integrator_kws = integrator_keywords)

# Note that the output is one example short of the length of t_test
print("Length of t_test:", len(t_test))
print("Length of simulation:", len(x_test_sim))

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test[:-1], x_test[:-1, i], "k", label="true simulation")
    axs[i].plot(t_test[:-1], x_test_sim[:, i], "r--", label="model simulation")
    axs[i].set(xlabel="t", ylabel="$x_{}$".format(i))

plt.show(block=True)
plt.pause(0.001)

# TODO: Save the model
with open('./control_affine_models/saved_models/' + 'inverted_pendulum_sindy', 'wb') as file:
	pickle.dump(model, file)

# Testing
with open('./control_affine_models/saved_models/' + 'inverted_pendulum_sindy', 'rb') as file:
	model2 = pickle.load(file)