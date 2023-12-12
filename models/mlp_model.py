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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, '../')
from dataset.to_nx import get_dataset

# random seed
SEED = 42
np.random.seed(SEED)


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        # 定义隐藏层
        self.hidden_layers = nn.ModuleList([
            nn.Linear(input_size, hidden_sizes[0]),
            # nn.ReLU()
            # nn.Tanh()
        ])
        
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.extend([
                nn.Linear(hidden_sizes[i-1], hidden_sizes[i]),
                # nn.ReLU()
                # nn.Tanh()
            ])

        # 输出层
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # 前向传播
        for layer in self.hidden_layers:
            x = layer(x)

        x = self.output_layer(x)

        return x

    
    
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
	for epoch in range(num_epochs + 1):
		model.train()
		for inputs, targets in train_loader:
			# print('inputs: ', inputs.shape)
			# print('targets: ', targets.shape)
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			loss.backward()
			optimizer.step()
		if epoch % 100 == 0:
			print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 验证函数
def validate_model(model, val_loader):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
    
    performance_matrix = evaluate_metrics(targets, outputs)

    return performance_matrix

def evaluate_metrics(y_gt, y_pred):
	
	mse = mean_squared_error(y_gt, y_pred)
	mae = mean_absolute_error(y_gt, y_pred)
	
	performance_matrix = {'mse': mse,
                          'mae': mae}
    
	return performance_matrix


# data processing: normalization
import numpy as np

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



def main():
	input_size = 27
	hidden_sizes = [64, 64]
	output_size = 1

	model = MLP(input_size, hidden_sizes, output_size)
	# criterion = nn.MSELoss()
	criterion = nn.L1Loss()
 
	initial_lr = 0.002
	optimizer = optim.Adam(model.parameters(), lr=initial_lr)
	scheduler = StepLR(optimizer, step_size=200, gamma=0.90)
 
	# save learning rate
	lr_history = []

	num_epochs = 2000
	dataset_splits = joblib.load('../data/dataset_splits.pkl')
    
	for i, dataset_split in enumerate(dataset_splits):
		print(i)
		X_train, y_train = dataset_split['X_train'], dataset_split['y_train']
		X_test, y_test = dataset_split['X_test'], dataset_split['y_test']
		
		# normaliztion
		# X_train, min_vals, max_vals = normalize_data(X_train)
		min_vals = min(np.append(y_train, y_test))
		max_vals = max(np.append(y_train, y_test))
		y_train, min_vals, max_vals = normalize_data(y_train)
		y_test, min_vals, max_vals = normalize_data(y_test)
  
		# numpy to tensor
		# X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
		# X_val, Y_val = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
  
		# Dataloader
		# train_dataset = TensorDataset(X_train, y_train)
		# train_loader = DataLoader(train_dataset, batch_size=len(X_train), shuffle=True) # len(X_train)

		# val_dataset = TensorDataset(X_val, Y_val)
		# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

		#
		train_dataset = MyDataset(X_train, y_train)
		train_loader = DataLoader(train_dataset, batch_size=len(X_train), shuffle=True)

		val_dataset = MyDataset(X_test, y_test)
		val_loader = DataLoader(val_dataset, batch_size=len(X_train), shuffle=True)
  
		# for each splited dataset
		train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
		performance_matrix = validate_model(model, val_loader)
		print(f'Performance for Fold {i + 1}: {performance_matrix}')



if __name__ == '__main__':
	# load data
	# dataset_splits = joblib.load('../data/dataset_splits.pkl')
	
	main()
	# model = MLP(input_size=10, hidden_sizes=[64, 128, 64], output_size=1)
	# print(model)
