#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 06, 2023

@author: Xiao Ning
"""

import os
import sys
import numpy as np

import joblib

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, '../')
from dataset.to_nx import get_dataset

# random seed
SEED = 42
np.random.seed(SEED)

# model


model_predictions= []

# model


def fit_dt_model(dataset_splits):
    
	# creat model
	tree_model = DecisionTreeRegressor(max_depth=25, 
                                       min_samples_split = 5,
                                       criterion='absolute_error',
                                       random_state=SEED) 

	for i, dataset_split in enumerate(dataset_splits):
		X_train, y_train = dataset_split['X_train'], dataset_split['y_train']
		X_test, y_test = dataset_split['X_test'], dataset_split['y_test']

		# fite
		tree_model.fit(X_train, y_train)
	
		# test
		y_pred = tree_model.predict(X_test)

		# save results
		model_predictions.append({'y_pred': y_pred, 'y_test': y_test})

		# print metrics
		performance_matrix = evaluate_metrics(y_test, y_pred)
		print(f'Performance for Fold {i + 1}: {performance_matrix}')
		
	# save results
	joblib.dump(model_predictions, '../results/dt_model_predictions.pkl')

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
	
	fit_dt_model(dataset_splits)