import numpy as np
from sigmoid import sigmoid


def costFunctionReg(theta, reg, X, y):
    '''returns the cost in a regularized manner,
    input is theta,lambda as reg,X and y as inputs and predicted value respectively
    np.log(a)==> returns array of elementwise log of element'''
    m = y.size
    h = sigmoid(X.dot(theta))
    theta_J = theta[1:]
    regparameter = reg*(theta_J)/m                                                                                # the calue added to the cost function
    J = -1*(1/m)*((np.log(h).T).dot(y)+np.log(1-h).T.dot(1-y)) + regparameter
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])


def gradientReg(theta, reg, X, y):
    '''returns the gradient for input values of theta, reg, X and y'''
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1, 1)))

    grad = (1/m)*X.T.dot(h-y) + (reg/m)*np.r_[[[0]], theta[1:].reshape(-1, 1)]

    return(grad.flatten())
