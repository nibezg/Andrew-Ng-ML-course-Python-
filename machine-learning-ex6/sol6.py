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
from scipy import sparse


import matplotlib.pyplot as plt

from sklearn import svm

import seaborn as sns

from nltk import PorterStemmer

from collections import Counter

#%%
# The first dataset. Using the Seaborn package for easier plotting.
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex6\ex6')
# os.chdir(r'C:\Users\user\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex6\ex6')
exp1_data_mat = scipy.io.loadmat('ex6data1.mat')
#%%
X = exp1_data_mat['X']
y_arr = exp1_data_mat['y']

exp1_data_df = pd.DataFrame(np.column_stack((X, y_arr)), columns=['x0', 'x1', 'y'])
#%%
fig, ax = plt.subplots()
sns.scatterplot(x='x0', y='x1', hue='y', data=exp1_data_df, ax=ax)
plt.show()
#%%
# Support-vector machine with Gaussian kernel

sigma = 2
clf = svm.SVC(C=100, kernel='linear')
svc_fit = clf.fit(X=X, y=y_arr[:, 0])
#%%
# Plotting the decision boundary
X_fit = np.zeros((X.shape[0] * 10, X[0].shape[0]))

for i in range(X_fit[0].shape[0]):
    X_fit[:, i] = np.linspace(np.min(X[:, i]), np.max(X[:, i]), X_fit.shape[0])

xx, yy = np.meshgrid(X_fit[:, 0], X_fit[:, 1])

X_plot = np.concatenate((xx.reshape(1, xx.size).T, yy.reshape(1, yy.size).T), axis=1)

y_plot_arr = clf.predict(X_plot).reshape(xx.shape[0], yy.shape[0])

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
ax.contour(X_fit[:, 0], X_fit[:, 1], y_plot_arr, alpha=0.6)
sns.scatterplot(x='x0', y='x1', hue='y', data=exp1_data_df, ax=ax)

#ax.set_ylim(np.min(X_fit[:, 1]), np.max(X_fit[:, 1]))
#ax.set_xlim(np.min(X_fit[:, 0]), np.max(X_fit[:, 0]))

plt.show()
#%%
# The second dataset
exp2_data_mat = scipy.io.loadmat('ex6data2.mat')

X = exp2_data_mat['X']
y_arr = exp2_data_mat['y']

exp2_data_df = pd.DataFrame(np.column_stack((X, y_arr)), columns=['x0', 'x1', 'y'])

fig, ax = plt.subplots()
sns.scatterplot(x='x0', y='x1', hue='y', data=exp2_data_df, ax=ax)
plt.show()
#%%
def build_gaussian(sigma):
    # This is the helper function to be able to define the value of sigma in the sklearn wrapper of the SVM algorithm.
    def gauss_kernel(X, X_loc):
        ''' My own implementaion of the Gaussian kernel

        X_loc is the array of localizations (equal to the features of the training set used to train the model)

        X is the array of features.

        The output has the shape of (X.shape[0], X_loc.shape[0])
        '''
        X_diff_sq = np.zeros((X_loc.shape[0], X.shape[0]))

        for i in range(X_loc.shape[0]):
            x_to_subtract_arr = np.ones(X.shape) * X_loc[i]
            X_diff_sq[i] = np.sum((X-x_to_subtract_arr)**2, axis=1)

        return np.exp(-X_diff_sq.T / (2 * sigma **2))

    return gauss_kernel

# Using my own Gaussian kernel function for running the SVM algorithm. Using the buil-in gaussian kernel has better implementation, because I do not run into memory problem with its usage. I am not sure what it is doing, however.
sigma = 0.1
#clf = svm.SVC(C=1, kernel='rbf', cache_size=200, gamma=1/(2*sigma**2))

clf = svm.SVC(C=1, kernel=build_gaussian(sigma))

svc_fit = clf.fit(X=X, y=y_arr[:, 0])
#%%
# Plotting the decision boundary
X_fit = np.zeros((X.shape[0]-660, X[0].shape[0]))

for i in range(X_fit[0].shape[0]):
    X_fit[:, i] = np.linspace(np.min(X[:, i]), np.max(X[:, i]), X_fit.shape[0])

xx, yy = np.meshgrid(X_fit[:, 0], X_fit[:, 1])

X_plot = np.concatenate((xx.reshape(1, xx.size).T, yy.reshape(1, yy.size).T), axis=1)

y_plot_arr = clf.predict(X_plot).reshape(xx.shape[0], yy.shape[0])

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
ax.contourf(X_fit[:, 0], X_fit[:, 1], y_plot_arr, alpha=0.6)
sns.scatterplot(x='x0', y='x1', hue='y', data=exp2_data_df, ax=ax)

#ax.set_ylim(np.min(X_fit[:, 1]), np.max(X_fit[:, 1]))
#ax.set_xlim(np.min(X_fit[:, 0]), np.max(X_fit[:, 0]))

plt.show()
#%%
# The third dataset
exp3_data_mat = scipy.io.loadmat('ex6data3.mat')

X = exp3_data_mat['X']
y_arr = exp3_data_mat['y']

X_val = exp3_data_mat['Xval']
y_val_arr = exp3_data_mat['yval']

exp3_data_df = pd.DataFrame(np.column_stack((X, y_arr)), columns=['x0', 'x1', 'y'])

fig, ax = plt.subplots()
sns.scatterplot(x='x0', y='x1', hue='y', data=exp3_data_df, ax=ax)
plt.show()
#%%
# Visualization of the validation data
exp3_val_data_df = pd.DataFrame(np.column_stack((X_val, y_val_arr)), columns=['x0', 'x1', 'y'])

fig, ax = plt.subplots()
sns.scatterplot(x='x0', y='x1', hue='y', data=exp3_val_data_df, ax=ax)
plt.show()
#%%
C_arr = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
#C_arr = np.linspace(0.01, 30, 40)
sigma_arr = C_arr.copy()

CC, ss = np.meshgrid(C_arr, sigma_arr)

fit_param_arr = np.concatenate((CC.reshape(1, CC.size).T, ss.reshape(1, ss.size).T), axis=1)

cross_val_acc_arr = np.zeros(fit_param_arr.shape[0])

for i in range(fit_param_arr.shape[0]):
    sigma = fit_param_arr[i, 1]
    #clf = svm.SVC(C=fit_param_arr[i, 0], kernel=build_gaussian(sigma), cache_size=200, tol=1E-3)
    clf = svm.SVC(C=fit_param_arr[i, 0], kernel='rbf', cache_size=200, gamma=1/(2*sigma**2))
    svc_fit = clf.fit(X=X, y=y_arr[:, 0])

    y_fit_arr = clf.predict(X_val)

    # I would naively try to use the mean-squared error, but it seems that the fraction of the matched outputs is better
    #cross_val_error_arr[i] = 1 / (2 * y_fit_arr.shape[0]) * (y_fit_arr - y_val_arr[:, 0]).T @ (y_fit_arr - y_val_arr[:, 0])
    cross_val_acc_arr[i] = y_val_arr[y_fit_arr == y_val_arr[:,0]].shape[0]/y_val_arr.shape[0]

#%%
fig, ax = plt.subplots()
ax.scatter(list(range(cross_val_acc_arr.shape[0])), cross_val_acc_arr)
plt.show()
#%%
cross_val_acc_arr[np.argmax(cross_val_acc_arr)]
#%%
# There are several identical maximal accuracy values. In a sense we are picking one of them
cross_val_acc_arr
#%%
fig, ax = plt.subplots()
fig.set_size_inches(10, 8)

c_plot = ax.contour(C_arr, sigma_arr, cross_val_acc_arr.reshape(C_arr.shape[0], sigma_arr.shape[0]), alpha=1)

cf_plot = ax.contourf(C_arr, sigma_arr, cross_val_acc_arr.reshape(C_arr.shape[0], sigma_arr.shape[0]), alpha=1)

# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig.colorbar(cf_plot)
# Add the contour line levels to the colorbar
cbar.add_lines(c_plot)

ax.set_xlabel('C')
ax.set_ylabel('sigma')
plt.show()
#%%
fit_param_max = fit_param_arr[np.argmax(cross_val_acc_arr)]
sigma = fit_param_max[1]
clf = svm.SVC(C=fit_param_max[0], kernel='rbf', cache_size=200, gamma=1/(2*sigma**2))

#clf = svm.SVC(C=1, kernel=build_gaussian(sigma))

svc_fit = clf.fit(X=X, y=y_arr[:, 0])
#%%
# Plotting the decision boundary
X_fit = np.zeros((X.shape[0], X[0].shape[0]))

for i in range(X_fit[0].shape[0]):
    X_fit[:, i] = np.linspace(np.min(X[:, i]), np.max(X[:, i]), X_fit.shape[0])

xx, yy = np.meshgrid(X_fit[:, 0], X_fit[:, 1])

X_plot = np.concatenate((xx.reshape(1, xx.size).T, yy.reshape(1, yy.size).T), axis=1)

y_plot_arr = clf.predict(X_plot).reshape(xx.shape[0], yy.shape[0])

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
ax.contourf(X_fit[:, 0], X_fit[:, 1], y_plot_arr, alpha=0.6)
sns.scatterplot(x='x0', y='x1', hue='y', data=exp3_data_df, ax=ax)

#ax.set_ylim(np.min(X_fit[:, 1]), np.max(X_fit[:, 1]))
#ax.set_xlim(np.min(X_fit[:, 0]), np.max(X_fit[:, 0]))

plt.show()
#%%
'''
Spam classification.
'''

def preprocessEmail(email_text, header_Q=True):
    if header_Q:
        # The header ends when there are two \n characters in a row
        p = re.compile('(\\n\\n){1,1}')
        email_text = email_text[p.search(email_text).span()[1]::]
    # Lower case
    mod_str = email_text.lower()

    # Replace HTML tagging with single space
    p = re.compile('<[^<>]+>')
    mod_str = p.sub(' ', mod_str)

    # Replace numbers by a word 'number'
    p = re.compile('[0-9]+')
    mod_str = p.sub('number', mod_str)

    # Replace http(s):// with 'httpaddr'. Also removes any nonwhitespace characters after
    p = re.compile('(http|https)://[^\s]*')
    mod_str = p.sub('httpaddr', mod_str)

    # Replace e-mail addresses with 'emailaddr'
    p = re.compile('[^\s]+@[^\s]+')
    mod_str = p.sub('emailaddr', mod_str)

    # Replace $ (dollar signs) with 'dollar'
    p = re.compile('[$]+')
    mod_str = p.sub('dollar', mod_str)

    # Tokenizing the string, remove punctiation, non-words (special characters), more than one consecture whitespaces
    p = re.compile(r'\W+|_+')
    mod_str = list(filter(None, p.split(mod_str)))

    # Stem the words using the Porter stemmer
    ps = PorterStemmer()
    word_list = [ps.stem(word) for word in mod_str]

    return word_list
#%%
def get_vocab():
    ''' Process the emails by using the data from 5 folders and return the counts of word occurences in all of the emails
    '''
    email_word_list = []

    # folders with the data
    folder_name_list = ['easy_ham', 'easy_ham_2', 'hard_ham', 'spam', 'spam_2']

    for folder_name in folder_name_list:

        print(folder_name)

        os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex6\ex6\email_data')

        os.chdir(folder_name)

        email_list = os.listdir()

        for email_file_name in email_list:
            email_text_file = open(email_file_name, "r")
            email_str = email_text_file.read()
            email_text_file.close()
            word_list = preprocessEmail(email_str, header_Q=True)

            # Removing very short words
            word_list = [word for word in word_list if len(word)>1]
            email_word_list.extend(word_list)

    counts_dict = dict(Counter(email_word_list))
    return counts_dict
#%%
counts_dict = get_vocab()
#%%
# Select words that occur at least min_occ times in the e-mails
min_occ = 100
vocab_list = [word for word, occurences in counts_dict.items() if occurences >= min_occ]

# Sort the vocabulary alphabetically
vocab_list.sort()
#%%
# Comparison with the vocabulary included with the problem set.
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex6\ex6')
vocab_ps_df = pd.read_csv(filepath_or_buffer='vocab.txt', sep='\t', index_col=0, header=None)
#%%
# Words contained in my analysis, but not in the problem set
mine_minus_ps = list(set(vocab_list).difference(set(vocab_ps_df[1].values)))
#%%
# Words contained in the problem set, but not in my analysis
ps_minus_mine = list((set(vocab_ps_df[1].values)).difference(set(vocab_list)))
#%%
# The size of the vocabulary is larger in my case. There are also words that are present in the vocabulary of the problem set, but not present in the vocabulary obtained from my analysis. I think the reason for this is the way the Porter stemmer is implemented.
print(len(mine_minus_ps))
print(len(ps_minus_mine))
#%%
# I still want to use my vocabulary. Because of this, I need to generate my own training, cross-validation and test sets.
#%%
# Firstly, I need to save the vocabulary list:
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex6\ex6')

vocab_df = pd.DataFrame(vocab_list, columns=['Word'])
vocab_df.index.names = ['Index']
vocab_df.to_csv('vocab_mine.csv')
#%%
# Load the vocabulary obtained from my analysis of the data
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex6\ex6')

vocab_df = pd.read_csv('vocab_mine.csv', index_col=0, header=0)
#%%
vocab_df
#%%
def create_feature_arr():
    ''' Creating feature arrays for each folder
    '''

    # folders with the data
    folder_name_list = ['easy_ham', 'easy_ham_2', 'hard_ham', 'spam', 'spam_2']

    # List of values for the respective folders
    y_val_list = [0, 0, 0, 1, 1]

    for folder_index in range(len(folder_name_list)):
        folder_name = folder_name_list[folder_index]
        print(folder_name)

        os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex6\ex6\email_data')

        os.chdir(folder_name)

        email_list = os.listdir()

        # Initialize array of features and their values
        X = np.zeros((len(email_list), vocab_df.shape[0]))
        y_arr = np.ones(len(email_list)) * y_val_list[folder_index]

        for email_index in range(len(email_list)):

            email_file_name = email_list[email_index]

            email_text_file = open(email_file_name, "r")
            email_str = email_text_file.read()
            email_text_file.close()
            word_list = preprocessEmail(email_str, header_Q=True)

            # Removing very short words
            word_list = [word for word in word_list if len(word)>1]

            x_arr = np.zeros(len(vocab_list))

            word_set = set(word_list)
            for i in range(len(vocab_list)):
                if vocab_list[i] in word_set:
                    x_arr[i] = 1
            X[email_index] = x_arr

        os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex6\ex6\email_data\processed_data')

        if not(os.path.isdir(folder_name)):
            os.mkdir(folder_name)

        os.chdir(folder_name)

        # Using sparse matrices for storage of matrix A
        X_csr = sparse.csr_matrix(X)
        sparse.save_npz('X.npz', X_csr)
        np.save('y.npy', y_arr)
        pd.Series(email_list).to_csv('f_names.txt', header=False)

#%%
def form_tr_set():
    '''Form the training set and the test set
    '''

    # Load the features + their output values
    os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex6\ex6\email_data\processed_data')

    # folders with the data
    folder_name_list = ['easy_ham', 'easy_ham_2', 'hard_ham', 'spam', 'spam_2']

    X_list = []
    y_list = []

    for folder_index in range(len(folder_name_list)):
        folder_name = folder_name_list[folder_index]

        os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex6\ex6\email_data\processed_data')
        os.chdir(folder_name)

        # Using sparse matrices for storage of matrix A
        X_csr = sparse.load_npz('X.npz')
        y_arr = np.load('y.npy')

        X_list.append(X_csr)
        y_list.append(y_arr)

    # Combine the feature arrays from the folders into one sparse matrix + one output vector
    X_csr = sparse.vstack(X_list)
    y_arr = np.hstack(y_list)

    # Randomly picking n_sample samples that will be used as the training data. The rest will be used as the training data.

    n_sample = 4000

    index_arr = np.arange(0, y_arr.shape[0])
    rand_index_arr = np.random.choice(index_arr, size=n_sample, replace=False)
    test_index_arr=np.array(list(set(index_arr).difference(set(rand_index_arr))))

    X_train_csr = X_csr[rand_index_arr]
    y_train_arr = y_arr[rand_index_arr]

    X_test_csr = X_csr[test_index_arr]
    y_test_arr = y_arr[test_index_arr]

    # Store the training and test data
    os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex6\ex6\email_data\processed_data')

    sparse.save_npz('X_test.npz', X_test_csr)
    sparse.save_npz('X_train.npz', X_train_csr)
    np.save('y_train.npy', y_train_arr)
    np.save('y_test.npy', y_test_arr)
#%%
# Load the training and test data
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex6\ex6\email_data\processed_data')

X_train_csr = sparse.load_npz('X_train.npz')
y_train_arr = np.load('y_train.npy')

X_test_csr = sparse.load_npz('X_test.npz')
y_test_arr = np.load('y_test.npy')
#%%
# Using linear kernel to train the SVM model
C_const = 0.1

clf = svm.SVC(C=C_const, kernel='linear', cache_size=200, tol=1E-3)
svc_fit = clf.fit(X=X_train_csr.toarray(), y=y_train_arr)
#%%
training_error = y_train_arr[clf.predict(X_train_csr.toarray()) == y_train_arr].shape[0] / y_train_arr.shape[0]

test_error = y_test_arr[clf.predict(X_test_csr.toarray()) == y_test_arr].shape[0] / y_test_arr.shape[0]
#%%
training_error
#%%
test_error
#%%
# Determining the most important words that are indicative of a spam message

# Load the vocabulary obtained from my analysis of the data
os.chdir(r'E:\Google Drive\Study\Coursera\Machine Learning\machine-learning-ex6\ex6')

vocab_df = pd.read_csv('vocab_mine.csv', index_col=0, header=0)

i_max = 15
most_likely_spam_words_str = ' '.join(np.flip(vocab_df.loc[np.argsort(clf.coef_[0])[clf.coef_[0].shape[0]-i_max:]].values[:,0]))

print(most_likely_spam_words_str)
