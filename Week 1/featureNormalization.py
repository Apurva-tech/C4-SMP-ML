# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:11:38 2020

@author: hp
"""

import numpy as np
def featureNormalization(X):
    """
    Take in numpy array of X values and return normalize X values,
    the mean and standard deviation of each feature
    np.mean(A)==> returns mean of all the elements of A
    np.std(A)==> returns the standard deviation of the elements of A
    """
    mean=np.mean(X)
    std=np.std(X)
    
    X_norm = (X - mean)/std
    
    return X_norm