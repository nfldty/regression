import numpy as np
import matplotlib.pyplot as plt
import math


def sigmoid(z):
    
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        cost += - y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    return cost / m


def compute_regularized_cost(X, y, w, b, lambda_=1)
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
    
    
def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape   
    p = np.zeros(m)
   
    ### START CODE HERE ### 
    # Loop over each example
    for i in range(m):   
        f_wb = sigmoid(np.dot(X[i], w) + b)
        # Apply the threshold
        p[i] = 1 if f_wb >= 0.5 else 0
        
    ### END CODE HERE ### 
    return p


def compute_gradient(X, y, w, b):
    m, n = X.shape
    gradients_w = np.zeros((n, ))
    gradient_b = 0
    for i in range(m):
        sigma = sigmoid(np.dot(X[i], w) + b) - y[i]
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