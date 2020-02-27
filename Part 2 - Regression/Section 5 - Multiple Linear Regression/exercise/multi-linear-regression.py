#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:38:53 2020

@author: nunocarvalhao
"""


# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('50_Startups.csv')

#matrix of features (independent variables)
X = dataset.iloc[:, :-1].values

# matrix of dependent variables
Y = dataset.iloc[:, 3].values

# Split dataset into training set and test set

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling ist's needed beacause of the enormous diference betwen the absolute values of the features
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)"""