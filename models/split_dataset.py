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

sys.path.insert(0, '../')
from dataset.to_nx import get_dataset


if __name__ == '__main__':
    # load data
    for descriptor in ['KENEEE_Atom_Sum_NN']:
        dataset = get_dataset('QM7', descriptor,
                        remove_sig_errs=True, from_file=True)

    X = dataset['X']
    y = np.array(dataset['targets'])

    # KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # save the aplited data
    dataset_splits = []

    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # save
        dataset_split = {'X_train': X_train,
                        'y_train': y_train, 
                        'X_test': X_test, 
                        'y_test': y_test}
        dataset_splits.append(dataset_split)
        
    # save
    joblib.dump(dataset_splits, '../data/dataset_splits.pkl')