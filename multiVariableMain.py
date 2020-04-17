# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:09:48 2020

@author: hp
"""

import numpy as np
import pandas as pd
data = pd.read_csv('ex1data2.txt', sep = ',', header = None)
X = data.iloc[:,0:2]                                                    # read first two columns into X
y = data.iloc[:,2]                                                      # read the third column into y
m = len(y)                                                              # no. of training samples
print(data.head())                                                      # view first few rows of the data
ones = np.ones((m,1))                                                   # initializing an array of 1s as value for intercept terms
X = np.hstack((ones, X))
alpha = 0.01
iterations = 500
theta = np.zeros((3,1))
y = y[:,np.newaxis]


'''Functions defined in other files will be imported here'''

from computeCost import computeCost
from gradientDescent import gradientDescent
from featureNormalization import featureNormalization

X=featureNormalization(X)                                   # calling function featureNormalization from featureNormalization.py file
J= computeCost(X,y,theta)                                   #calling function computeCost from computeCost.py file
print('Cost function J value :',J)
input("Press enter if you have completed gradient descent file, else Ctrl+C then enter to exit")
theta = gradientDescent(X, y, theta, alpha, iterations)     #calling function gradientDescent from gradientDescent.py file
J= computeCost(X,y,theta)
print('New Cost function value:',J)