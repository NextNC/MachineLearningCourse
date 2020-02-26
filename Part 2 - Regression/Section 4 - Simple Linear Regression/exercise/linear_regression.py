#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:03:05 2020

@author: nunocarvalhao
"""

## Simple Linear Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Salary_Data.csv')

#matrix of features (independent variables)
X = dataset.iloc[:, :-1].values

# matrix of dependent variables
Y = dataset.iloc[:, 1].values

# Split dataset into training set and test set

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Feature scaling ist's needed beacause of the enormous diference betwen the absolute values of the features
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)"""

##Fitting Simple Linear Regression to the training set


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, Y_train)


##Predicting test set results 

Y_pred = regressor.predict(X_test)

## Plot training set results

plt.scatter(X_train, Y_train, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

## Plot test set results

plt.scatter(X_train, Y_train, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


















