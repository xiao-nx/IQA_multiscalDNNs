#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:36:48 2021

@author: ljia
"""

import sys
sys.path.insert(0, '../')
from dataset.to_nx import get_qm7_eigen_vectors
from dataset.retrieve_data import get_molecule_energies

reconst = get_qm7_eigen_vectors('reconst_infos', verbose=0)
Emol_ref_all = get_molecule_energies(dataset='QM7', DFT_method='pbeqidh')


Emol_gau = []
Emol_ref = []
for idx, info in enumerate(reconst):
	mol_idx = idx + 1
	if isinstance(info, dict):
		Emol_gau.append(info['Emol_Gaussian'])
		Emol_ref.append(Emol_ref_all[idx])


from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(Emol_gau, Emol_ref, squared=False)
print('rmse is ' + str(rmse))


import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

coef = np.polyfit(Emol_gau, Emol_ref, 1)
poly1d_fn = np.poly1d(coef)

plt.plot(Emol_gau, Emol_ref, '+', Emol_gau, poly1d_fn(Emol_gau), '--', markersize=0.1, linewidth=0.1)
plt.xlabel('Emol_gau')
plt.ylabel('Emol_ref')
plt.savefig('error_energies.pdf', format='pdf', dpi=300, transparent=False, bbox_inches='tight')
plt.show()
plt.clf()
plt.close()