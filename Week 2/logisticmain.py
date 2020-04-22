import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from plotData import plotData
from loadData import loaddata

data = loaddata('ex2data1.txt', ',')

X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]                            # adds column of ones (first argument) to X data (second argument) AND creates X from data at the same time
y = np.c_[data[:,2]]                                                          # creates Y from data
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')    #plots the data#

plt.show()
input("proceed only if you've completed completed ALL the files, else press Ctrl+C then enter to exit the program")
from costFunction import  costFunction,gradient

initial_theta = np.zeros(X.shape[1])
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Cost: \n', cost)
print('Grad: \n', grad)


res = minimize(costFunction, initial_theta, args=(X,y), jac=gradient)         # using sklearn's  minimize function to optimize our constfunction for theta while taking in X,y and using gradient function
updated_theta=res.x
print(updated_theta)                                                         #printing updated theta to the screen

from sigmoid import sigmoid
plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
x1_min, x1_max = X[:,0].min(), X[:,0].max() ,                                               #taking min and max of marks in exam 1
x2_min, x2_max = X[:,1].min(), X[:,1].max()  ,                                              # taking min and max of marks in exam 2
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))            #creating a graph with min and max of exams 1 and 2 as boundaries
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))  # evaluating the sigmoid over the gird, returns (2500,1) style vector 
h = h.reshape(xx1.shape)                                                                    #converting 2500,1 to 50,50 to be able to plot
print(h.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')                                   #potting the function on the graph      
plt.show()
