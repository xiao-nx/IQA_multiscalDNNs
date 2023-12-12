#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 06, 2023

@author: Xiao Ning
"""

import os
import sys
import joblib
import numpy as np

from tqdm import tqdm 


from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, '../')
from dataset.to_nx import get_dataset


def fit_svr_model(dataset_splits):
    
	# creat model
	# svr_model = SVR(kernel='linear', C=1.0, epsilon=0.2)
	model_predictions= []
 
	for i, dataset_split in tqdm(enumerate(dataset_splits)):
		print(i)
		X_train, y_train = dataset_split['X_train'], dataset_split['y_train']
		X_test, y_test = dataset_split['X_test'], dataset_split['y_test']

		# fit
		# svr_model.fit(X_train, y_train)
  
		# Create a pipeline with PCA and SVR
		pipeline = Pipeline([
			('scaler', StandardScaler()),  # Standardize features
			# ('pca', PCA(n_components=20)),   # Apply PCA, adjust n_components as needed
			('svr', SVR(kernel='linear', C=1.0, epsilon=0.1))
		])
  
		# Train the model
		pipeline.fit(X_train, y_train)
  
		# test
		y_pred = pipeline.predict(X_test)

		# save results
		model_predictions.append({'y_pred': y_pred, 'y_test': y_test})

		# print metrics
		performance_matrix = evaluate_metrics(y_test, y_pred)
		print(f'Performance for Fold {i + 1}: {performance_matrix}')
		
	# save results
	joblib.dump(model_predictions, '../results/svr_model_predictions.pkl')

	return 


def evaluate_metrics(y_gt, y_pred):
	
	mse = mean_squared_error(y_gt, y_pred)
	mae = mean_absolute_error(y_gt, y_pred)
	
	performance_matrix = {'mse': mse,
                          'mae': mae}
    
	return performance_matrix



if __name__ == '__main__':
	# load data
	dataset_splits = joblib.load('../data/dataset_splits.pkl')
	
	fit_svr_model(dataset_splits)