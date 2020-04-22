import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from loadData import loaddata
from plotData import plotData
data = loaddata('ex2data1.txt', ',')

X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]                            # adds column of ones (first argument) to X data (second argument) AND creates X from data at the same time
y = np.c_[data[:,2]]                                                          # creates Y from data
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')    #plots the data
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)  #split X and y data into training and testing set. Model will train on train set and will check accuracy on test set
logreg = LogisticRegression()                                                             #Initializing the model
logreg.fit(X_train, y_train)                                                              #Feeding values in the model
print('Theta ',logreg.coef_)                                                              #theta value after the model is trained
y_pred = logreg.predict(X_test)                                                           # generating the predictions on the test set
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
