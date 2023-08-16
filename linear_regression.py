import math, copy
import numpy as np
import matplotlib.pyplot as plt

'''
Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
'''
    
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):  
        cost += (np.dot(X[i], w) + b - y[i]) ** 2
    return 1 / (2 * m) * cost


def compute_regularized_cost(X, y, w, b, lambda_=1):
    m, n = X.shape
    cost = compute_cost(X, y, w, b)
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/2) * reg_cost
    return (cost + reg_cost) / m


def compute_regularized_gradient(X, y, w, b, lambda_=1):
    m, n = X.shape
    gradients_w, gradient_b = compute_gradient(X, y, w, b)
    for i in range(n):
        gradients_w[i] += lambda_ * w[i] / m
    return gradients_w, gradient_b



def compute_gradient(X, y, w, b):
    m, n = X.shape
    gradients_w = np.zeros((n, ))
    gradient_b = 0
    for i in range(m):
        sigma = (np.dot(X[i], w) + b - y[i])
        gradient_b += sigma
        for j in range(n):
            gradients_w[j] += sigma * X[i, j]
    return gradients_w / m, gradient_b / m


def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    w, b = w_init, b_init   
    for i in range(num_iters):
        dw, db = compute_gradient(X, y, w, b)
        w -= alpha * dw
        b -= alpha * db
    return w, b


def predict(X, w, b):  
    return np.array([np.dot(x, w) + b for x in X])
   





