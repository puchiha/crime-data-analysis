from __future__ import division
import numpy as np
#import matplotlib.pyplot as pl
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.metrics import *
from sklearn.metrics import accuracy_score




import random
import sys
import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn.gaussian_process as gp

iris_dataset = datasets.load_iris()


RANDOM_SEED = 0xDEADBEEF
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def gp_clf_iris():
    # Follow the example from the sklearn docs, and only use the
    # first two features, so we can visualize the predicted
    # probabilities in 2D.
    X = iris_dataset.data[:, :2]
    y = iris_dataset.target
    y_names = iris_dataset.target_names
    print("Feature names: ", iris_dataset.feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED)

    # Make the RBF kernel anisotropic for maximum flexibility.
    kernel = gp.kernels.RBF(np.ones((X.shape[1], 1))) \
        * gp.kernels.ConstantKernel() \
        + gp.kernels.WhiteKernel()
    clf = gp.GaussianProcessClassifier(kernel, n_restarts_optimizer=0)
    print("Fitting Gaussian Process on input of shape {0}...".format(
        X_train.shape
    ))
    clf.fit(X_train, y_train)
    print("Learned kernel: {0}".format(str(clf.kernel_)))
    print("Fit complete.")
    
    y_pred = clf.predict(X_test)
    print(y_pred)
    
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: {0:.2f}%".format(acc * 100.0))
    
    # Plot class probabilities in 2D, with the coordinates being the
    # values of the first and second features (f0, f1, i.e., sepal
    # length and sepal width).
    f0_min = X[:, 0].min() - 1
    f0_max = X[:, 0].max() + 1
    f1_min = X[:, 1].min() - 1
    f1_max = X[:, 1].max() + 1
    step = 0.02
    f0, f1 = np.meshgrid(np.arange(f0_min, f0_max, step),
                         np.arange(f1_min, f1_max, step))
    grid_data = np.c_[f0.ravel(), f1.ravel()]
    print(X.shape)
    print(X_train.shape)
    print(grid_data.shape)
    prob_grid = clf.predict_proba(grid_data)
    prob_grid = prob_grid.reshape((f0.shape[0], f0.shape[1], 3))
    print( prob_grid.shape)#, '\n --- \n' ,prob_grid.squeeze()
    exit()
    plt.figure(figsize=(6, 6))
    plt.imshow(prob_grid, extent=(f0_min, f0_max, f1_min, f1_max),
               origin='lower')

    plt.scatter(X[y==0, 0], X[y==0, 1], s=30, c='red', edgecolors='black')
    plt.scatter(X[y==1, 0], X[y==1, 1], s=30, c='green', edgecolors='black')
    plt.scatter(X[y==2, 0], X[y==2, 1], s=30, c='blue', edgecolors='black')
    plt.show()
gp_clf_iris()


'''

""" This is code for simple GP regression. It assumes a zero mean GP Prior """


# This is the true unknown function we are trying to approximate
f = lambda x: np.sin(0.9*x).flatten()
#f = lambda x: (0.25*(x**2)).flatten()


# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 0.1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)


# Sample some input points and noisy versions of the function evaluated at
# these points. 
#X = np.random.uniform(-5, 5, size=(N,1))
#y = f(X) + s*np.random.randn(N)

data = pd.read_csv("../raw_data/crime_processed_neighbourhood.csv").head(10).as_matrix()
#data = data.sample(40000).as_matrix()
X = data[:, [0,1,2,3,4,5,6,7,9]]
y = data[:, 8]
xTr, xTe, yTr, yTe = train_test_split(X, y, test_size=0.9)

xTr = xTr.reshape(-1,1)
xTe = xTe.reshape(-1,1)
N = len(xTr)         # number of training points.
n = len(xTe)         # number of test points.
s = 0.00005    # noise variance.

K = kernel(xTr, xTr)
L = np.linalg.cholesky(K + s*np.eye(N))

# points we're going to make predictions at.
#Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# compute the mean at our test points.
Lk = np.linalg.solve(L, kernel(xTr, xTe))
mu = np.dot(Lk.T, np.linalg.solve(L, yTe))

# compute the variance at our test points.
K_ = kernel(xTe, xTe)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)


# PLOTS:
pl.figure(1)
pl.clf()
pl.plot(xTr, yTr, 'r+', ms=20)
pl.plot(xTe, yTe, 'b-')
pl.gca().fill_between(XTe.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(xTe, mu, 'r--', lw=2)
pl.savefig('predictive.png', bbox_inches='tight')
pl.title('Mean predictions plus 3 st.deviations')
#pl.axis([-5, 5, -3, 3])

# draw samples from the prior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n,10)))
pl.figure(2)
pl.clf()
pl.plot(xTe, f_prior)
pl.title('Ten samples from the GP prior')
pl.axis([-5, 5, -3, 3])
pl.savefig('prior.png', bbox_inches='tight')

# draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,10)))
pl.figure(3)
pl.clf()
pl.plot(xTe, f_post)
pl.title('Ten samples from the GP posterior')
pl.axis([-5, 5, -3, 3])
pl.savefig('post.png', bbox_inches='tight')

pl.show()
'''
