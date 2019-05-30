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
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex1\ex1')

exp1_data_df = pd.read_csv('ex1data1.txt', header=None)

exp1_data_df.columns = ['Population', 'Profit']
exp1_data_df['x0'] = 1

#%%
exp1_data_df.plot(kind='scatter', x='Population', y='Profit')
#%%
def square_J(theta_arr, x_arr, y_arr):
    h_theta_arr = x_arr @ theta_arr
    m = x_arr.shape[0]
    return 1 / (2 * m) * (h_theta_arr - y_arr).T @ (h_theta_arr - y_arr)

def der_square_J(theta_arr, x_arr, y_arr):
    h_theta_arr = x_arr @ theta_arr
    m = x_arr.shape[0]
    return 1 / m * x_arr.T @ (h_theta_arr - y_arr)

def min_square_J(x_arr, y_arr, theta_0_arr, alpha, n_iter):
    theta_prev_arr = theta_0_arr.copy()
    theta_next_arr = theta_0_arr.copy()

    for i in range(n_iter):
        theta_next_arr = theta_prev_arr - alpha * der_square_J(theta_prev_arr, x_arr, y_arr)
        theta_prev_arr = theta_next_arr
    return theta_next_arr
#%%
x_arr = exp1_data_df[['x0', 'Population']].values
theta_arr = np.zeros((x_arr[0].shape[0], 1))
y_arr = exp1_data_df[['Profit']].values

square_J(theta_arr, x_arr, y_arr)
#%%
der_square_J(theta_arr, x_arr, y_arr)
#%%
theta_min_arr = min_square_J(x_arr, y_arr, theta_arr, alpha=0.02, n_iter=1000)

x_fit_arr = np.linspace(np.min(x_arr[:, 1]), np.max(x_arr[:, 1]), 100)
x_0_fit_arr = np.ones(x_fit_arr.shape)
x_fit_arr = np.row_stack((x_0_fit_arr, x_fit_arr)).T

y_fit_arr = x_fit_arr @ theta_min_arr
#%%
n_iter_arr = np.array([10, 100, 200, 300, 400, 600, 800, 1000, 1200, 1500, 2000])
alpha = 0.02
J_arr = np.zeros(n_iter_arr.shape[0])
for i in range(n_iter_arr.shape[0]):
    theta_min_arr = min_square_J(x_arr, y_arr, theta_arr, alpha=alpha, n_iter=n_iter_arr[i])
    J_arr[i] = J(theta_min_arr, x_arr, y_arr)
#%%
fig, ax = plt.subplots()
ax.plot(n_iter_arr, J_arr)
#%%
alpha_arr = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02])
J_arr = np.zeros(alpha_arr.shape[0])
for i in range(alpha_arr.shape[0]):
    theta_min_arr = min_square_J(x_arr, y_arr, theta_arr, alpha=alpha_arr[i], n_iter=1000)
    J_arr[i] = J(theta_min_arr, x_arr, y_arr)
#%%
fig, ax = plt.subplots()
ax.plot(alpha_arr, J_arr)
#%%
fig, ax = plt.subplots()
exp1_data_df.plot(kind='scatter', x='Population', y='Profit', ax=ax)
ax.plot(x_fit_arr[:, 1], y_fit_arr[:, 0], color='black')
plt.show()
#%%
np.min(x_arr)
#%%
os.chdir(r'C:\Users\Nikita\Downloads\machine-learning-ex1\machine-learning-ex1\ex1')

exp2_data_df = pd.read_csv('ex1data2.txt', header=None)

exp2_data_df.columns = ['Area', 'Number Of Bedrooms', 'Price']
exp2_data_df['x0'] = 1
#exp1_data_df['x0'] = 1
#%%
exp2_data_df.plot(kind='scatter', x='Area', y='Price')
#%%
exp2_data_df.plot(kind='scatter', x='Number Of Bedrooms', y='Price')
#%%
exp2_data_df.plot(kind='scatter', x='Area', y='Number Of Bedrooms')
#%%
x_arr = exp2_data_df[['x0', 'Area', 'Number Of Bedrooms']].values
theta_arr = np.zeros((x_arr[0].shape[0], 1))
y_arr = exp2_data_df[['Price']].values
#%%
def norm_feature(x_arr):
    x_part_arr = x_arr[:, 1:]
    return np.column_stack((x_arr[:, 0], (x_part_arr - np.mean(x_part_arr, axis=0)) / np.std(x_part_arr, axis=0, ddof=1)))
#%%
x_norm_arr = norm_feature(x_arr)
theta_min_arr = min_square_J(x_norm_arr, y_arr, theta_arr, alpha=1, n_iter=1000)
#%%
n_iter_arr = np.array([10, 100, 200, 300, 400, 600, 800, 1000, 1200, 1500, 2000])
alpha = 1
J_arr = np.zeros(n_iter_arr.shape[0])
for i in range(n_iter_arr.shape[0]):
    theta_min_arr = min_square_J(x_norm_arr, y_arr, theta_arr, alpha=alpha, n_iter=n_iter_arr[i])
    J_arr[i] = J(theta_min_arr, x_norm_arr, y_arr)
#%%
fig, ax = plt.subplots()
ax.plot(n_iter_arr, J_arr)
#%%
alpha_arr = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 1])
J_arr = np.zeros(alpha_arr.shape[0])
for i in range(alpha_arr.shape[0]):
    theta_min_arr = min_square_J(x_norm_arr, y_arr, theta_arr, alpha=alpha_arr[i], n_iter=1000)
    J_arr[i] = J(theta_min_arr, x_norm_arr, y_arr)
#%%
J_arr
#%%
fig, ax = plt.subplots()
ax.plot(alpha_arr, J_arr)
plt.show()
#%%
x_fit_norm_arr = np.zeros((x_norm_arr.shape[0] * 100, x_norm_arr[0].shape[0]))
x_fit_arr = np.zeros((x_arr.shape[0] * 100, x_arr[0].shape[0]))

for i in range(x_fit_norm_arr[0].shape[0]):
    x_fit_norm_arr[:, i] = np.linspace(np.min(x_norm_arr[:, i]), np.max(x_norm_arr[:, i]), x_fit_norm_arr.shape[0])

for i in range(x_fit_arr[0].shape[0]):
    x_fit_arr[:, i] = np.linspace(np.min(x_arr[:, i]), np.max(x_arr[:, i]), x_fit_arr.shape[0])

y_fit_arr = x_fit_norm_arr @ theta_min_arr
#%%
fig, ax = plt.subplots()

exp2_data_df.plot(kind='scatter', x='Area', y='Price', ax=ax)
ax.plot(x_fit_arr[:, 1], y_fit_arr[:, 0])

plt.show()
#%%
fig, ax = plt.subplots()

exp2_data_df.plot(kind='scatter', x='Number Of Bedrooms', y='Price', ax=ax)
ax.plot(x_fit_arr[:, 2], y_fit_arr[:, 0])

plt.show()
#%%
x_fit_norm_arr @ theta_min_arr
#%%
def solve_norm_eq(x_arr, y_arr):
    return np.linalg.inv(x_arr.T @ x_arr) @ x_arr.T @ y_arr
#%%
solve_norm_eq(x_norm_arr, y_arr)
#%%
theta_min_arr
