from __future__ import division
import numpy as np
import pandas as pd
import scipy
import sys
import os
import string
import shutil

import re
import time
import math
import copy

import numpy.fft
import numpy.linalg
import scipy.fftpack
import scipy.interpolate
import scipy.optimize

import matplotlib.pyplot as plt
#%%
'''
*************************
Exercise 1
*************************
'''
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex2\machine-learning-ex2\ex2')

exp1_data_df = pd.read_csv('ex2data1.txt', header=None)
exp1_data_df.columns = ['Score 1', 'Score 2', 'Admitted Flag']
exp1_data_df['x0'] = 1
color_mask_arr = (exp1_data_df['Admitted Flag'] == 1).values
color_arr = np.ones(color_mask_arr.shape).astype(np.str)
color_arr[color_mask_arr] = 'blue'
color_arr[np.logical_not(color_mask_arr)] = 'red'
#%%
fig, ax = plt.subplots()
exp1_data_df.plot(kind='scatter', x='Score 1', y='Score 2', ax=ax, color=color_arr)
plt.show()
#%%
'''
=================================
Using the gradient-descent method
=================================
'''
x_arr = exp1_data_df[['x0', 'Score 1', 'Score 2']].values
theta_arr = np.zeros((x_arr[0].shape[0], 1))
y_arr = exp1_data_df[['Admitted Flag']].values
#%%
#theta_arr = np.zeros(x_arr[0].shape[0])
theta_arr
#%%
def sig_func(x_arr):
    return 1 / (1 + np.exp(- 1 * x_arr))

def sig_J(theta_arr, x_arr, y_arr):
    h_arr = sig_func(x_arr @ theta_arr)
    m = x_arr.shape[0]
    return np.asscalar(1 / m * (-1*y_arr.T @ np.log(h_arr) - (1-y_arr).T @ np.log(1-h_arr)))

def sig_J_grad(theta_arr, x_arr, y_arr):
    h_arr = sig_func(x_arr @ theta_arr)
    m = x_arr.shape[0]

    return 1 / m * x_arr.T @ (h_arr - y_arr)

def min_sig_J(x_arr, y_arr, theta_0_arr, alpha, n_iter):
    theta_prev_arr = theta_0_arr.copy()
    theta_next_arr = theta_0_arr.copy()

    for i in range(n_iter):
        theta_next_arr = theta_prev_arr - alpha * sig_J_grad(theta_prev_arr, x_arr, y_arr)
        theta_prev_arr = theta_next_arr
    return theta_next_arr
#%%
theta_min_arr = min_sig_J(x_arr, y_arr, theta_0_arr=theta_arr, alpha=0.002, n_iter=400000)
theta_min_arr
#%%
sig_J(theta_min_arr, x_arr, y_arr)
#%%
sig_J_grad(theta_min_arr, x_arr, y_arr)
#%%
x_fit_arr = np.zeros((x_arr.shape[0] * 100, x_arr[0].shape[0]))

for i in range(x_fit_arr[0].shape[0]):
    x_fit_arr[:, i] = np.linspace(np.min(x_arr[:, i]), np.max(x_arr[:, i]), x_fit_arr.shape[0])

#%%
x2_arr = (-theta_min_arr[0] * x_fit_arr[:, 0] - theta_min_arr[1] * x_fit_arr[:, 1]) / theta_min_arr[2]
#%%
fig, ax = plt.subplots()

exp1_data_df.plot(kind='scatter', x='Score 1', y='Score 2', ax=ax, color=color_arr)
ax.plot(x_fit_arr[:,1], x2_arr)

plt.show()
#%%
'''
=================================
Using scipy gradient-conjugate optimization method
=================================
'''
def sig_J(theta_arr, x_arr, y_arr):
    h_arr = sig_func(x_arr @ theta_arr)
    m = x_arr.shape[0]
    return np.asscalar(1 / m * (-1*y_arr.T @ np.log(h_arr) - (1-y_arr).T @ np.log(1-h_arr)))

def sig_J_grad(theta_arr, x_arr, y_arr):
    h_arr = sig_func((x_arr @ theta_arr.reshape(theta_arr.shape[0], 1)))
    m = x_arr.shape[0]

    return (1 / m * x_arr.T @ (h_arr - y_arr)).T[0]

theta_arr = np.zeros(x_arr[0].shape[0])
#%%
sig_J_grad(theta_arr, x_arr, y_arr)
#%%
theta_min_arr = scipy.optimize.minimize(fun=sig_J, x0=theta_arr, args=(x_arr, y_arr), method='CG', jac=sig_J_grad, options={'disp': False, 'return_all': True})['x']
theta_min_arr
#%%
sig_J_grad(theta_min_arr, x_arr, y_arr)
#%%
x2_arr = (-theta_min_arr[0] * x_fit_arr[:, 0] - theta_min_arr[1] * x_fit_arr[:, 1]) / theta_min_arr[2]

fig, ax = plt.subplots()

exp1_data_df.plot(kind='scatter', x='Score 1', y='Score 2', ax=ax, color=color_arr)
ax.plot(x_fit_arr[:,1], x2_arr)

plt.show()
#%%
x_test_arr = np.array([1, 45, 85])
sig_func(theta_min_arr @ x_test_arr)
#%%
# Accuracy of the model %
y_arr[y_arr == (sig_func(x_arr @ theta_min_arr.reshape(theta_min_arr.shape[0], 1)) >= 0.5).astype(np.int)].shape[0] / y_arr.shape[0] * 100
#%%
'''
=================================
Using scipy gradient-conjugate optimization method with regularization
=================================
'''

os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex2\machine-learning-ex2\ex2')

exp2_data_df = pd.read_csv('ex2data2.txt', header=None)

exp2_data_df.columns = ['Test 1', 'Test 2', 'Pass Flag']
color_mask_arr = (exp2_data_df['Pass Flag'] == 1).values

#%%
color_arr = np.ones(color_mask_arr.shape).astype(np.str)
color_arr[color_mask_arr] = 'blue'
color_arr[np.logical_not(color_mask_arr)] = 'red'

fig, ax = plt.subplots()
exp2_data_df.plot(kind='scatter', x='Test 1', y='Test 2', ax=ax, color=color_arr)
plt.show()
#%%
# Add additional features of the form x_1^j x_2^i, such that i+j <= 6.
pow_max = 6

x_arr = np.zeros((exp2_data_df.shape[0], int((pow_max + 1)*(pow_max + 2) / 2)))

x_data_1_arr = exp2_data_df['Test 1']
x_data_2_arr = exp2_data_df['Test 2']
col = 0
for i in range(0, pow_max+1):
    for j in range(pow_max + 1 - i):
        x_arr[:, col] = (x_data_1_arr**j) * (x_data_2_arr**i)
        col = col + 1
        print(j, i)

y_arr = exp2_data_df[['Pass Flag']].values
theta_arr = np.zeros(x_arr[0].shape[0])
#%%

def sig_J(theta_arr, x_arr, y_arr, reg_const):
    h_arr = sig_func(x_arr @ theta_arr)
    m = x_arr.shape[0]
    return np.asscalar(1 / m * (-1*y_arr.T @ np.log(h_arr) - (1-y_arr).T @ np.log(1-h_arr)) + reg_const / (2*m) * theta_arr[1:].T @ theta_arr[1:])

def sig_J_grad(theta_arr, x_arr, y_arr, reg_const):
    h_arr = sig_func((x_arr @ theta_arr.reshape(theta_arr.shape[0], 1)))
    m = x_arr.shape[0]

    J_grad_arr = (1 / m * x_arr.T @ (h_arr - y_arr)).T[0] + reg_const / m * theta_arr
    J_grad_arr[0] = J_grad_arr[0] - reg_const / m * theta_arr[0]

    return J_grad_arr

#%%
reg_const = 1

theta_min_arr = scipy.optimize.minimize(fun=sig_J, x0=theta_arr, args=(x_arr, y_arr, reg_const), method='CG', jac=sig_J_grad, options={'disp': False, 'return_all': False})['x']
theta_min_arr
#%%
x_fit_arr = np.zeros((x_arr.shape[0] * 10, x_arr[0].shape[0]))

for i in range(x_fit_arr[0].shape[0]):
    x_fit_arr[:, i] = np.linspace(np.min(x_arr[:, i]), np.max(x_arr[:, i]), x_fit_arr.shape[0])

y_fit_arr = x_fit_arr @ theta_min_arr.reshape(theta_min_arr.shape[0], 1)
#%%
xx, yy = np.meshgrid(x_fit_arr[:, 1], x_fit_arr[:, 7])
z_arr = np.zeros((x_fit_arr[:, 1].shape[0], x_fit_arr[:, 7].shape[0]))

col = 0
for i in range(0, pow_max+1):
    for j in range(pow_max + 1 - i):
        z_arr = z_arr + theta_min_arr[col] * (xx**j) * (yy**i)
        col = col + 1
#%%
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
ax.set_aspect(1)
ax.contour(x_fit_arr[:, 1], x_fit_arr[:, 7], z_arr, colors='k', levels=[0])
exp2_data_df.plot(kind='scatter', x='Test 1', y='Test 2', ax=ax, color=color_arr)

# ax.set_xlim((-1.5, 1.5))
# ax.set_ylim((-1.5, 1.5))

plt.show()

#%%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
fig.set_size_inches(16, 16)
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x_fit_arr[:, 1], x_fit_arr[:, 7], z_arr)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
