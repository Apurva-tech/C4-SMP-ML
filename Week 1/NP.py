# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:22:23 2020

@author: hp
"""

import numpy as np

A= np.array([[1,1],[2,2],[3,3]])        #one way to initialize numpy array
print(A)
print(A.shape)                          # print shape (dimension) of the matrix 
print(A.T)                              # print Transpose of A
print(np.sum(A))                        # print sum of all the elements of A
print(np.power(A,2))                    # square each element
print(np.mean(A))                       # average of all the elements in A
print(np.std(A))                        # standard deviation of all the elements in A


B=np.array([[1,2],[3,4],[5,6]])
print(B*A)                              # print elementwise multiplication of A and B
print(np.dot(A,B.T))                    # print dot product of A and B transpose
print(np.cross(A,B))                    # print cross product of A and B

print(np.eye(5))                        # print identity matrix of size 5,5

print(np.zeros((4,1)))                  # Initialize a 4x1 vector with all values 0

print(np.ones((4,1)))                   # Initialize a 4x1 vector with all values as 1