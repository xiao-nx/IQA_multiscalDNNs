#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:07:10 2021

@author: ljia
"""
import sys
sys.path.insert(0, '../')
from dataset.to_nx import get_qm7_eigen_vectors


reconst = get_qm7_eigen_vectors('reconst_infos', verbose=0)

sig_errs = {}
for idx, info in enumerate(reconst):
	if isinstance(info, dict) and info['sig']:
		sig_errs[idx + 1] = info