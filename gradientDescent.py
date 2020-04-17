# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:10:58 2020

@author: hp
"""

import numpy as np
from computeCost import computeCost

def gradientDescent(X,y,theta,alpha,num_iters):
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
    with learning rate of alpha
    return theta updated
    """
    m=len(y)
    for i in range(num_iters):
        predictions = np.array(theta[0]+(theta[1])*(np.array(X)))
        error = np.sum(np.array(X)-np.array(y))
        descent=np.array(alpha * 1/m * error)
        theta= theta-descent
        if(i%100==0):
            print('Cost function after ',i,' iterations is ', computeCost(X,y,theta))
    
    return theta