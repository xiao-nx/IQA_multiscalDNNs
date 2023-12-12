#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:23:24 2021

@author: ljia
"""
import os
import sys
sys.path.insert(0, '../')
from dataset.to_nx import get_qm7_eigen_vectors
from dataset.retrieve_data import get_molecule_energies
import numpy as np


def get_vectors():
	import pickle

	fn_save = 'eigenvectors_saved.pkl'
	if os.path.isfile(fn_save) and os.path.getsize(fn_save) != 0:
		with open(fn_save, 'rb') as f:
			vecs = pickle.load(f)
		self_iqa = vecs['self_iqa']
		additive_iqa = vecs['additive_iqa']
		iqa = vecs['iqa']
		asympt_iqa = vecs['asympt_iqa']

	else:
		self_iqa = get_qm7_eigen_vectors('self_iqa_eigenvector', verbose=0)
		additive_iqa = get_qm7_eigen_vectors('additive_iqa_eigenvector', verbose=0)
		iqa = get_qm7_eigen_vectors('iqa_eigenvector', verbose=0)
		asympt_iqa = get_qm7_eigen_vectors('asympt_iqa_eigenvector', verbose=0)
		with open(fn_save, 'wb') as f:
			pickle.dump({'self_iqa': self_iqa,
				         'additive_iqa': additive_iqa,
						 'iqa': iqa,
						 'asympt_iqa': asympt_iqa}, f)

	return self_iqa, additive_iqa, iqa, asympt_iqa


def check_total_error():
	self_iqa, additive_iqa, iqa, asympt_iqa = get_vectors()

	from sklearn.metrics import mean_squared_error

	rmse_self = [mean_squared_error(self_iqa[i], additive_iqa[i], squared=False) for i in range(0, len(additive_iqa))]
	rmse_iqa = [mean_squared_error(iqa[i], additive_iqa[i], squared=False) for i in range(0, len(additive_iqa))]
	rmse_asy = [mean_squared_error(asympt_iqa[i], additive_iqa[i], squared=False) for i in range(0, len(additive_iqa))]


	mean_self_iqa = np.mean(self_iqa[self_iqa != 0])
	mean_iqa = np.mean(iqa[iqa != 0])
	mean_asy_iqa = np.mean(asympt_iqa[asympt_iqa != 0])

	max_rmse_self = np.max(rmse_self)
	max_rmse_iqa = np.max(rmse_iqa)
	max_rmse_asy = np.max(rmse_asy)

	mean_rmse_self = np.mean(rmse_self[rmse_self != 0])
	mean_rmse_iqa = np.mean(rmse_iqa[rmse_iqa != 0])
	mean_rmse_asy = np.mean(rmse_asy[rmse_asy != 0])


	### Compute reletive errors.

	import matplotlib.pyplot as plt

	plt.bar(range(0, len(additive_iqa)), rmse_asy)
	# 	bars, width=barWidth, color=palette(i),
	# 	edgecolor='black', linewidth=0.2,
	# 	yerr=y_err, error_kw=dict(lw=0.5, capsize=3, capthick=0.5),
	# 	label=xp)

	nm = 'rmse_asy'
	plt.savefig(nm + '.png', format='png', dpi=300, transparent=False, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close()


def check_error_of_each_atom_type():
	self_iqa, additive_iqa, iqa, asympt_iqa = get_vectors()
	self_iqa_by_atypes = {'C': [], 'O': [], 'N': [], 'H': [], 'S': []}
	additive_iqa_by_atypes = {'C': [], 'O': [], 'N': [], 'H': [], 'S': []}
	iqa_by_atypes = {'C': [], 'O': [], 'N': [], 'H': [], 'S': []}
	asympt_iqa_by_atypes = {'C': [], 'O': [], 'N': [], 'H': [], 'S': []}

	### Get values for each type of atom. Find unexpected values.
	for i_mol, vec in enumerate(additive_iqa):
		for i_eigen, eigenv in enumerate(vec):
			if eigenv == 0:
				continue
			elif -40 < eigenv < -30: # (37., C)
				atype = 'C'
			elif -80 < eigenv < -70: # (73., O)
				atype = 'O'
			elif -60 < eigenv < -50: # (53., N)
				atype = 'N'
			elif -0.6 < eigenv < -0.29: # (0.59., H)
				atype = 'H'
			elif -400 < eigenv < -390: # (392., S)
				atype = 'S'
			else:
				print('%d, %d: %f' % (i_mol, i_eigen, eigenv))
				continue

			self_iqa_by_atypes[atype].append(self_iqa[i_mol, i_eigen])
			additive_iqa_by_atypes[atype].append(eigenv)
			iqa_by_atypes[atype].append(iqa[i_mol, i_eigen])
			asympt_iqa_by_atypes[atype].append(asympt_iqa[i_mol, i_eigen])


	### Compute stats.
	from sklearn.metrics import mean_squared_error


	print('errors between additive IQA and self IQA:')
	for atype in additive_iqa_by_atypes.keys():
		rmse_self = mean_squared_error(self_iqa_by_atypes[atype], additive_iqa_by_atypes[atype], squared=False)
		relative_err_self = _relative_rmse(additive_iqa_by_atypes[atype], self_iqa_by_atypes[atype])
		print('%s: rmse: %f, rmse (%%): %.2f%%.' % (atype, rmse_self, relative_err_self * 100))

	print('errors between additive IQA and IQA:')
	for atype in additive_iqa_by_atypes.keys():
		rmse_iqa = mean_squared_error(iqa_by_atypes[atype], additive_iqa_by_atypes[atype], squared=False)
		relative_err_iqa = _relative_rmse(additive_iqa_by_atypes[atype], iqa_by_atypes[atype])
		print('%s: rmse: %f, rmse (%%): %.2f%%.' % (atype, rmse_iqa, relative_err_iqa * 100))

	print('errors between additive IQA and asympt IQA:')
	for atype in additive_iqa_by_atypes.keys():
		rmse_asympt = mean_squared_error(asympt_iqa_by_atypes[atype], additive_iqa_by_atypes[atype], squared=False)
		relative_err_asympt = _relative_rmse(additive_iqa_by_atypes[atype], asympt_iqa_by_atypes[atype])
		print('%s: rmse: %f, rmse (%%): %.2f%%.' % (atype, rmse_asympt, relative_err_asympt * 100))


def _relative_rmse(base, compared):
	return np.sqrt(np.sum(np.divide(np.subtract(compared, base), base) ** 2) / len(base))




if __name__ == '__main__':
# 	check_total_error()
	check_error_of_each_atom_type()