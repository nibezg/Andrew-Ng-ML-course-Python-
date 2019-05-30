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
import numpy.random

import scipy.fftpack
import scipy.interpolate
import scipy.optimize
import scipy.io

import matplotlib.pyplot as plt
#%%
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex3\ex3')
exp1_data_mat = scipy.io.loadmat('ex3data1.mat')
x_no_x0_arr = exp1_data_mat['X']
y_arr = exp1_data_mat['y']

x_arr = np.ones((x_no_x0_arr.shape[0], x_no_x0_arr.shape[1]+1))
x_arr[:, 1:] = x_no_x0_arr
#%%
n_img = 100
ncols = int(np.sqrt(n_img))
nrows = int(np.sqrt(n_img))
img_index_arr = np.random.randint(low=0, high=x_no_x0_arr.shape[0], size=n_img).reshape(nrows, ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})

fig.set_size_inches(7, 7)

for i in range(nrows):
    for j in range(ncols):
        axes[i, j].imshow(x_no_x0_arr[img_index_arr[i, j]].reshape(20, 20).T * 255, aspect='auto', cmap=plt.cm.binary)
        axes[i, j].set_xticklabels([])
        axes[i, j].set_yticklabels([])
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
plt.show()
#%%
def sig_func(x_arr):
    return 1 / (1 + np.exp(- 1 * x_arr))

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
# Train the model
reg_const = 0.1

val_arr = pd.Series(y_arr[:, 0]).drop_duplicates().values
theta_arr = np.zeros((val_arr.shape[0], x_arr[0].shape[0]))

for k_index in range(val_arr.shape[0]):
    k = val_arr[k_index]

    # indeces of the training samples that correspond to the y_arr == k
    m_index_arr = np.nonzero(y_arr.T[0] == k)[0]

    y_k_arr = np.zeros(y_arr.shape)
    y_k_arr[m_index_arr] = 1
    theta_k_arr = np.zeros(x_arr[0].shape[0])

    theta_min_k_arr = scipy.optimize.minimize(fun=sig_J, x0=theta_k_arr, args=(x_arr, y_k_arr, reg_const), method='BFGS', jac=sig_J_grad, options={'disp': False, 'return_all': False})['x']

    theta_arr[k_index] = theta_min_k_arr
#%%
m_index = np.random.randint(low=0, high=x_no_x0_arr.shape[0], size=1)[0]
print('Prediction: ' + str(val_arr[np.argmax(sig_func(x_arr[m_index] @ theta_arr.T))]))

fig, ax = plt.subplots()

ax.imshow(x_no_x0_arr[m_index].reshape(20, 20).T * 255, aspect='auto', cmap=plt.cm.binary)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])
plt.show()
#%%
training_acc = y_arr.T[0][val_arr[np.argmax(sig_func(x_arr @ theta_arr.T), axis=1)] == y_arr.T[0]].shape[0] / y_arr.T[0].shape[0] * 100
training_acc
#%%
'''
Using a neural network with precomputed weights. Implementing forward propagation
'''
nn_weights_data_mat = scipy.io.loadmat('ex3weights.mat')
#%%
nn_weights_data_mat
#%%
theta_1_arr = nn_weights_data_mat['Theta1']
theta_2_arr = nn_weights_data_mat['Theta2']
#%%
theta_1_arr.shape
#%%
theta_2_arr.shape
#%%
# Input layer with the bias term already included
a1_arr = x_arr

# Hidden layer
a2_arr = np.ones((a1_arr.shape[0], theta_1_arr.shape[0]+1))
a2_arr[:, 1:] = sig_func(a1_arr @ theta_1_arr.T)

# Output layer
out_arr = sig_func(a2_arr @ theta_2_arr.T)
#%%
val2_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

training_nn_acc = y_arr.T[0][val2_arr[np.argmax(out_arr, axis=1)] == y_arr.T[0]].shape[0] / y_arr.T[0].shape[0] * 100
training_nn_acc
