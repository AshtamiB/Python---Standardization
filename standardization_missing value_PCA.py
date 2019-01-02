# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 13:08:20 2018

@author: Ashtami
"""
#Axis 0 = columns, Axis 1 = Rows

#STANDARDIZATION (Standardized_X = (X – Average) / Std_Deviation)

import os
os.chdir("C:/Users/Ashtami/Documents/Python/")

import sklearn.preprocessing as skp
from sklearn.datasets import load_iris
iris = load_iris() #load the Iris dataset
X = iris.data
y = iris.target
scaler = skp.StandardScaler().fit(X)


standardized_X = scaler.transform(X)
#inverse_X = scaler.inverse_transform(standardized_X)
print(standardized_X)


#NORMALIZATION  (Normalized_X = (X – min)/(max-min) )

import sklearn.preprocessing as skp
Normalized_Dataset = skp.normalize(Dataset, norm='l2')

#BINARIZATION values below threshold are 0, above it are 1
import sklearn.preprocessing as skp
binarized_Dataset = skp.binarize(Dataset,threshold=0.0)


#MISSING VALUE IMPUTATION
import sklearn.preprocessing as skp
imp = skp.Imputer(missing_values=0,strategy='mean',axis=0)
data_imputed = imp.fit_transform(Dataset)
print(data_imputed)

#2
import pandas as pd
import numpy as np
import sklearn.preprocessing as skp
DataFrame = pd.read_csv('C:/Users/Ashtami/Documents/Python/Data_Missing_values.csv',header=None)
DataMatrix = DataFrame.as_matrix()
input_matrix = np.array(DataMatrix)
imp = skp.Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
data_imputed = imp.fit_transform(input_matrix)
print(data_imputed)

#strategy could be 'mean', 'median', 'most_frequent'.
#Missing values could be 0, 'NaN'.
#Axis = 0 for columns, 1 for rows

#MISSING VALUE SUBSTITUTE
data.fillna(0)

#PRINCIPAL COMPUTATIONAL ANALYSIS
import numpy as np
import sklearn.decomposition as skd

Dataset = np.array([[0.387,4878, 5.42],[0.723,12104,5.25],
[1,12756,5.52],[1.524,6787,3.94],])
pca = skd.PCA(n_components=2, whiten=False)
pca.fit(Dataset)
Dataset_Reduced_Dim = pca.transform(Dataset)
print(Dataset_Reduced_Dim )
print(Dataset_Reduced_Dim.shape)
U, S, V = np.linalg.svd(Dataset)


#----------


