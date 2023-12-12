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


class CustomDataset(Dataset):
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
            nn.Tanh()
        ])
        
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.extend([
                nn.Linear(hidden_sizes[i-1], hidden_sizes[i]),
                # nn.ReLU()
                nn.Tanh()
            ])

        # 输出层
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # 前向传播
        for layer in self.hidden_layers:
            x = layer(x)

        x = self.output_layer(x)

        return x
    
    
# 定义一个大的神经网络，包含两个子网络的结构
class CombinedNet(nn.Module):
	def __init__(self, sub_net1, sub_net2, sub_net3):
		super(CombinedNet, self).__init__()
		self.sub_net1 = sub_net1
		self.sub_net2 = sub_net2
		self.sub_net3 = sub_net3

	def forward(self, x1, x2, x3):
		out1 = self.sub_net1(x1)
		out2 = self.sub_net2(x2)
		out3 = self.sub_net3(x3)
        # Define the structure of the large network as needed
		out = out1 + out2 + out3
		return out

    
    
def train_model(model, train_loader_0, train_loader_1, train_loader_2, criterion, optimizer, scheduler, num_epochs=10):
	for epoch in range(num_epochs + 1):
		model.train()
		for (inputs1, targets), (inputs2, _), (inputs3, _) in zip(train_loader_0, train_loader_1, train_loader_2):
			# 
			optimizer.zero_grad()
			# feed data
			outputs = model(inputs1, inputs2, inputs3)
			loss = criterion(outputs, targets)
			
			# bp 
			loss.backward()
			optimizer.step()
		# Update the learning rate at the end of each epoch
		scheduler.step()
   
		if epoch % 100 == 0:
			print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 验证函数
def validate_model(model, val_loader_0, val_loader_1, val_loader_2):
    model.eval()
    # total_loss = 0.0
    
    with torch.no_grad():
        for (inputs1, targets), (inputs2, _), (inputs3, _) in zip(val_loader_0, val_loader_1, val_loader_2):
            outputs = model(inputs1, inputs2, inputs3)
    
    performance_matrix = evaluate_metrics(targets, outputs)

    return performance_matrix

def evaluate_metrics(y_gt, y_pred):
	
	mse = mean_squared_error(y_gt, y_pred)
	mae = mean_absolute_error(y_gt, y_pred)
	
	performance_matrix = {'mse': mse,
                          'mae': mae}
    
	return performance_matrix


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
    
	group_index = [
     				[5, 6, 7, 8 ,9, 20], # 6
                	[1, 11, 21, 22, 23, 24, 25], # 7
                 	[0, 2, 3, 4, 10, 12, 13, 14, 15, 16, 17, 18, 19, 26] # 14
                  ]
	
	input_size_0 = len(group_index[0])
	input_size_1 = len(group_index[1])
	input_size_2 = len(group_index[2])
	hidden_sizes = [64, 64]
	output_size = 1

	sub_net1 = MLP(input_size_0, hidden_sizes, output_size)
	sub_net2 = MLP(input_size_1, hidden_sizes, output_size)
	sub_net3 = MLP(input_size_2, hidden_sizes, output_size)

	# # 创建一个大网络实例
	combined_net = CombinedNet(sub_net1, sub_net2, sub_net3)
 
	criterion = nn.MSELoss()
	# criterion = nn.L1Loss()
 
	initial_lr = 0.0025
	optimizer = optim.Adam(combined_net.parameters(), lr=initial_lr)

	scheduler = StepLR(optimizer, step_size=250, gamma=0.90)
 
	# save learning rate
	# lr_history = []

	num_epochs = 2000

	dataset_splits = joblib.load('../data/dataset_splits.pkl')
    
	for i, dataset_split in enumerate(dataset_splits):
		# print(i)
		X_train, y_train = dataset_split['X_train'], dataset_split['y_train']
		X_test, y_test = dataset_split['X_test'], dataset_split['y_test']
		
		## normaliztion
		# x

		# y
		min_vals = min(np.append(y_train, y_test))
		max_vals = max(np.append(y_train, y_test))
		y_train, min_vals, max_vals = normalize_data(y_train)
		y_test, min_vals, max_vals = normalize_data(y_test)
  
		train_dataset_0 = CustomDataset(X_train[:, group_index[0]], y_train)
		train_dataset_1 = CustomDataset(X_train[:, group_index[1]], y_train)
		train_dataset_2 = CustomDataset(X_train[:, group_index[2]], y_train)
  
		train_loader_0 = DataLoader(train_dataset_0, batch_size=len(X_train), shuffle=True) # 
		train_loader_1 = DataLoader(train_dataset_1, batch_size=len(X_train), shuffle=True)
		train_loader_2 = DataLoader(train_dataset_2, batch_size=len(X_train), shuffle=True)

		val_dataset_0 = CustomDataset(X_test[:, group_index[0]], y_test)
		val_dataset_1 = CustomDataset(X_test[:, group_index[1]], y_test)
		val_dataset_2 = CustomDataset(X_test[:, group_index[2]], y_test)

		val_loader_0 = DataLoader(val_dataset_0, batch_size=1, shuffle=True)
		val_loader_1 = DataLoader(val_dataset_1, batch_size=1, shuffle=True)
		val_loader_2 = DataLoader(val_dataset_2, batch_size=1, shuffle=True)

  
		# for each splited dataset
		train_model(combined_net, train_loader_0, train_loader_1, train_loader_2, criterion, optimizer, scheduler, num_epochs=num_epochs)
		performance_matrix = validate_model(combined_net, val_loader_0, val_loader_1, val_loader_2)
		print(f'Performance for Fold {i + 1}: {performance_matrix}')



if __name__ == '__main__':
	# load data
	# dataset_splits = joblib.load('../data/dataset_splits.pkl')
	
	main()
	# model = MLP(input_size=10, hidden_sizes=[64, 128, 64], output_size=1)
	# print(model)
