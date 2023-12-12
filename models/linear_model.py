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

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, '../')
from dataset.to_nx import get_dataset

# random seed
SEED = 42
np.random.seed(SEED)

# model


model_predictions= []


def normalize_data(data):
    """
    将数据归一化到 0 到 1 区间

    Parameters:
    - data: 原始数据，一个二维数组或矩阵

    Returns:
    - normalized_data: 归一化后的数据
    - min_vals: 每列的最小值
    - max_vals: 每列的最大值
    """
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data, min_vals, max_vals

def inverse_normalize_data(normalized_data, min_vals, max_vals):
    """
    将归一化后的数据映射回原始值

    Parameters:
    - normalized_data: 归一化后的数据
    - min_vals: 每列的最小值
    - max_vals: 每列的最大值

    Returns:
    - original_data: 映射回原始值的数据
    """
    original_data = normalized_data * (max_vals - min_vals) + min_vals
    return original_data


def fit_linear_model(dataset_splits):
    
	# creat model
	linear_model = LinearRegression(fit_intercept=True)    

	for i, dataset_split in enumerate(dataset_splits):
		X_train, y_train = dataset_split['X_train'], dataset_split['y_train']
		X_test, y_test = dataset_split['X_test'], dataset_split['y_test']
		print("X_train:", X_train)
  
		# normaliztion
		# X_train, min_vals, max_vals = normalize_data(X_train)
		min_vals = min(np.append(y_train, y_test))
		max_vals = max(np.append(y_train, y_test))
		y_train, min_vals, max_vals = normalize_data(y_train)
		y_test, min_vals, max_vals = normalize_data(y_test)

		# fite
		linear_model.fit(X_train, y_train)
	
		# test
		y_pred = linear_model.predict(X_test)

		# save results
		model_predictions.append({'y_pred': y_pred, 'y_test': y_test})

		# print metrics
		performance_matrix = evaluate_metrics(y_test, y_pred)
		print(f'Performance for Fold {i + 1}: {performance_matrix}')
		
	# save results
	joblib.dump(model_predictions, '../results/linear_model_predictions.pkl')

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
	
	fit_linear_model(dataset_splits)