# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importing libraries

import  numpy as np 

import matplotlib.pyplot as plt
import pandas as pd

### importing dataset
dataset = pd.read_csv('Data.csv')

## insdendant variables (matrix of features)
X = dataset.iloc[:, :-1].values

## dependant variables vector

Y = dataset.iloc[:, 3].values

# taking care missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(X[ : , 1:3])

X[ : , 1:3] = imputer.transform(X[ : , 1:3])

# processing categoric data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labenencoder_X = LabelEncoder()
X[ : ,0 ] = labenencoder_X.fit_transform(X[ : ,0 ])

oneHotEncoder = OneHotEncoder(categorical_features = [0])

X = oneHotEncoder.fit_transform(X).toarray()

labenencoder_Y = LabelEncoder()
Y = labenencoder_Y.fit_transform(Y)








