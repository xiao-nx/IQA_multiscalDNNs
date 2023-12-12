#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 17:12:38 2021

@author: ljia
"""

import os
import sys
from tqdm import tqdm
from load_dataset import gjf_to_molecule_formula


def get_gaussian_unnormal_list(path, itr_mode='serialnumber', verbose=True):

	def _get_mol_formula(fn_full):
		fn_gjf = fn_full.replace('outputs/', 'datasets/', 1)[:-3] + 'gjf'
		fn_gjf = fn_gjf.split('/')
		fn_gjf = '/'.join(fn_gjf[:-3]) + '/gjf.' + fn_gjf[-3] + '/' + fn_gjf[-1]
		return gjf_to_molecule_formula(fn_gjf)

	# Notice idx_unnormal starts from 1.
	idx_unnormal = []
	mol_sym_list = []

	if itr_mode == 'serialnumber':
		nb_total = len([f for f in os.listdir(path) if f.endswith('.out')])
		for i in tqdm(range(1, nb_total + 1), desc='Checking unnormal terminations', file=sys.stdout):
			fname = 'molecule' + str(i) + '.out'
			fn_full = os.path.join(path, fname)
			if _get_unnormal_termination_info_from_file(fn_full, idx_unnormal, idx_mol=i):
				mol_sym_list.append(_get_mol_formula(fn_full))

	elif itr_mode == 'filename':
		fn_list = [f for f in os.listdir(path) if f.endswith('.out')]
		nb_total = len(fn_list)
		for fname in tqdm(fn_list, desc='Checking unnormal terminations', file=sys.stdout):
			mol_name = os.path.splitext(fname)[0]
			fn_full = os.path.join(path, fname)
			if _get_unnormal_termination_info_from_file(fn_full, idx_unnormal, idx_mol=mol_name):
				mol_sym_list.append(_get_mol_formula(fn_full))

	else:
		import warnings
		warnings.warn('Sorry, the "itr_mode" ' + itr_mode + ' cannot be recognized. Supporting "serialnumber" and "filename".')

	if verbose:
		print('Mols with unnormal terminations: ', idx_unnormal)
		print('{0:.2f}'.format(len(idx_unnormal) / nb_total * 100) + '% of Gaussian computation terminated unnormally.')

	return idx_unnormal, mol_sym_list, nb_total


def _get_unnormal_termination_info_from_file(fname, idx_unnormal, idx_mol=None):
	if not os.path.isfile(fname):
		print('File "' + fname + '" does not exist. Please check.')
		return False

	with open(fname, 'r') as f:
		lines = f.readlines()
		line_termination = [l for l in lines if 'Normal termination of Gaussian 16 at' in l]
		if len(line_termination) < 1:
			print('The gaussian computation of mol ' + str(idx_mol) + ' terminate unnormally.')
			idx_unnormal.append(idx_mol)
			return True

	return False


def get_gaussian_unnormal_list_from_csv(path):
	import pandas as pd

	df = pd.read_csv(path)
	df = pd.to_numeric(df.iloc[range(0, len(df) - 1), 0]).tolist()

	return df


def save_gaussian_termination_stats(dataset, method_s, itr_mode='serialnumber', symbol_path=None):
	# Set paths.
	path = '../outputs/' + dataset + '/' + method_s + '/gaussian/'
	save_path = '../outputs/' + dataset + '/' + method_s + '/stats_gaussian_termination_' + dataset.replace('/', '_') + '.csv'
	print('data path: ', path)
	print('save path: ', save_path)

	# Notice idx_unnormal starts from 1.
	idx_unnormal, mol_sym_list, nb_total = get_gaussian_unnormal_list(path, itr_mode=itr_mode, verbose=True)

	# Get the according molecule symbol.
	import pandas as pd
	import numpy as np

	if symbol_path is not None:

		df = pd.read_csv(symbol_path)
		df = df.iloc[np.array(idx_unnormal) - 1, [0, -1]]

	else:

		df = pd.DataFrame()
		if itr_mode == 'serialnumber':
			df['# mol'] = idx_unnormal
		else:
			df['# mol'] = range(1, len(idx_unnormal) + 1)
		df['Formula'] = mol_sym_list

	# Add summary.
	### Sum up each column.
	col_sum = ['total mols: ' + str(len(idx_unnormal)),
			'percentage: ' + '{0:.2f}'.format(len(idx_unnormal) / nb_total * 100) + '%']
	df.loc['sum'] = col_sum

	# Change index to the first column.
	df.set_index('# mol', inplace=True)


	df.to_csv(save_path)


if __name__ == '__main__':
	#%%
#	#####################################################################
#	# Save gaussian unnormal termination for QM7, svwn sto-3g.
#	dataset = 'QM7'
#	method_s = 'svwn'
#	itr_mode = 'serialnumber'
#	symbol_path = '../datasets/' + dataset + '/stats_' + dataset + '.csv'


# 	#####################################################################
# 	# Save gaussian unnormal termination for QM7, pbeqidh.
# 	dataset = 'QM7'
# 	method_s = 'pbeqidh'
# 	itr_mode = 'serialnumber'
# 	symbol_path = '../datasets/' + dataset + '/stats_' + dataset + '.csv'


# 	#####################################################################
# 	# Save gaussian unnormal termination for QM7, svwn.6-31Gd.
# 	dataset = 'QM7'
# 	method_s = 'svwn.6-31Gd'
# 	itr_mode = 'filename'
# 	symbol_path = None


	#%%
	#####################################################################
	# Save gaussian unnormal termination for QMrxn20, svwn sto-3g.
	#dataset = 'QMrxn20/reactant-conformers'
	dataset = 'QMrxn20/transition-states/e2'
	#dataset = 'QMrxn20/transition-states/sn2'
	method_s = 'svwn'
	itr_mode = 'serialnumber'
	symbol_path = None


	#%%
# 	#####################################################################
#	# Save gaussian unnormal termination for Diatomic, pbeqidh.
#	dataset = 'Diatomic'
#	method_s = 'pbeqidh'
#	itr_mode = 'filename'
#	symbol_path = None



	#%%
	save_gaussian_termination_stats(dataset, method_s, itr_mode=itr_mode, symbol_path=symbol_path)

#	# Get gaussian unnormal termination list from .csv statistics file.
#	path = '../outputs/QM7/svwn/stats_nnacp_qm7.csv'
#	idx_unnormal = get_nnacp_list_from_csv(path)
#	# Check not computed nnacp mols.
#	missed = []
#	for i in idx_unnormal:
#		fname = '../outputs/QM7/svwn.6-31Gd/wfx_tmp/molecule' + str(i) + '.wfx'
#		if not os.path.isfile(fname):
#			missed.append(i)
#	print(missed)


#	#%%
#	# Save nnacp statistics for Diatomic using pbeqidh.
#	path = '../outputs/Diatomic/pbeqidh/wfx_tmp/'
#	save_path = '../outputs/Diatomic/pbeqidh/stats_nnacp_diatomic.csv'
#	save_nnacp_stats(path, save_path, itr_mode='filename')