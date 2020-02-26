#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:36:05 2020

@author: nunocarvalhao
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Data.csv')

#matrix of features (independent variables)
X = dataset.iloc[:, :-1].values

# matrix of dependent variables
Y = dataset.iloc[:, 3].values

# Taking care of missing data
# Updated Imputer
from sklearn.preprocessing import Imputer

#verbose/axis 1 rows 0 columns
imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)

# Upper bound is exluded that's why is 3 
imputer = imputer.fit(X[:, 1:3])
# replace nan by mean value                      
X[:, 1:3]=imputer.transform(X[:, 1:3])

#Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X = LabelEncoder() 
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

# Because we cant have different weights between countries we need to create a binary table to represent each one

oneHotEnconder = OneHotEncoder(categorical_features=[0])
X = oneHotEnconder.fit_transform(X).toarray()

# Encode categorical data from dependent varibles
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)

# Split dataset into training set and test set

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling ist's needed beacause of the enormous diference betwen the absolute values of the features
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)























