#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:30:37 2021

@author: ljia
"""
import os
import sys
from tqdm import tqdm
from load_dataset import gjf_to_molecule_formula


def get_nnacp_list(path, itr_mode='serialnumber', nb_mols=None, verbose=True):

	def _get_mol_formula(fn_full):
		fn_gjf = fn_full.replace('outputs/', 'datasets/', 1)[:-3] + 'gjf'
		fn_gjf = fn_gjf.split('/')
		fn_gjf = '/'.join(fn_gjf[:-3]) + '/gjf.' + fn_gjf[-3] + '/' + fn_gjf[-1]
		return gjf_to_molecule_formula(fn_gjf)


	# Notice idx_nnacp starts from 1.
	idx_nnacp = []
	num_list = []
	mol_sym_list = []

	if itr_mode == 'serialnumber':
		for i in tqdm(range(1, nb_mols + 1), desc='Checking nnacps', file=sys.stdout):
			fname = 'molecule' + str(i) + '.sum'
			fn_full = os.path.join(path, fname)
			if _get_nnacp_info_from_file(fn_full, idx_nnacp, num_list, idx_mol=i):
				mol_sym_list.append(_get_mol_formula(fn_full))

	elif itr_mode == 'filename':
		fn_list = [f for f in os.listdir(path) if f.endswith('.sum')]
		for fname in tqdm(fn_list, desc='Checking nnacps', file=sys.stdout):
			mol_name = os.path.splitext(fname)[0]
			fn_full = os.path.join(path, fname)
			if _get_nnacp_info_from_file(fn_full, idx_nnacp, num_list, idx_mol=mol_name):
				mol_sym_list.append(_get_mol_formula(fn_full))


	else:
		import warnings
		warnings.warn('Sorry, the "itr_mode" ' + itr_mode + ' cannot be recognized. Supporting "serialnumber" and "filename".')

	if verbose:
		print('Mols with NNACPs: ', idx_nnacp)
		nb_total = (nb_mols if itr_mode == 'serialnumber' else len(fn_list))
		print('{0:.2f}'.format(len(idx_nnacp) / nb_total * 100) + '% of the files contain psudo atom(s).')

	return idx_nnacp, num_list, nb_total, mol_sym_list


def _get_nnacp_info_from_file(fname, idx_nnacp, num_list, idx_mol=None):

	if not os.path.isfile(fname):
		print(fname + ' does not exist.')
		return False

	with open(fname, 'r', encoding='charmap') as f:
		try:
			lines = f.readlines()
		except UnicodeDecodeError:
			# The default encoding format "charmap" may encounter some unrecognized characters: "UnicodeDecodeError: 'utf-8' codec can't decode byte 0x91 in position 6618: invalid start byte".
			with open(fname, 'r', encoding='windows-1252') as f1:
				lines = f1.readlines()

	line_nnacp = [l for l in lines if 'Number of NNACPs' in l]
	if not len(line_nnacp) == 1:
		print('more than one line in mol ', str(idx_mol), '.')
	else:
		num = int(line_nnacp[0].split()[-1])
		if num != 0:
			print('There are NNACPs in mol ' + str(idx_mol) + '.')
			idx_nnacp.append(idx_mol)
			num_list.append(num)
			return True

	return False


def get_nnacp_list_from_csv(path):
	import pandas as pd

	df = pd.read_csv(path)
	df = pd.to_numeric(df.iloc[range(0, len(df) - 1), 0]).tolist()

	return df


def save_nnacp_stats(path, save_path, itr_mode='serialnumber', symbol_path=None):
	# Notice idx_nnacp starts from 1.
	idx_nnacp, num_list, nb_total, mol_sym_list = get_nnacp_list(path, itr_mode=itr_mode, nb_mols=7165, verbose=True)

	# Get the according molecule symbol.
	import pandas as pd
	import numpy as np

	if symbol_path is not None:

		df = pd.read_csv(symbol_path)
		df = df.iloc[np.array(idx_nnacp) - 1, [0, -1]]

	else:

		df = pd.DataFrame()
		if itr_mode == 'serialnumber':
			df['# mol'] = idx_nnacp
		else:
			df['# mol'] = range(1, len(idx_nnacp) + 1)
		df['Formula'] = mol_sym_list

	df['# NNACPs'] = num_list

	# Add summary.
	### Sum up each column.
	col_sum = ['Summary:', 'total mols: ' + str(len(idx_nnacp)),
			'percentage: ' + '{0:.2f}'.format(len(idx_nnacp) / nb_total * 100) + '%']
	df.loc['sum'] = col_sum

	# Change index to the first column.
	df.set_index('# mol', inplace=True)


	df.to_csv(save_path)


if __name__ == '__main__':
	#%%

# 	### Save nnacp statistics for QM7, svwn sto-3g.
# 	path = '../outputs/QM7/svwn/wfx_tmp/'
# 	save_path = '../outputs/QM7/svwn/stats_nnacp_qm7.csv'
# 	symbol_path = '../datasets/QM7/stats_QM7.csv'
# 	save_nnacp_stats(path, save_path, itr_mode='serialnumber', symbol_path=symbol_path)

# 	# Get nnacp list from .csv statistics file.
# 	path = '../outputs/QM7/svwn/stats_nnacp_qm7.csv'
# 	idx_nnacp = get_nnacp_list_from_csv(path)
# 	# Check not computed nnacp mols.
# 	missed = []
# 	for i in idx_nnacp:
# 		fname = '../outputs/QM7/svwn.6-31Gd/wfx_tmp/molecule' + str(i) + '.wfx'
# 		if not os.path.isfile(fname):
# 			missed.append(i)
# 	print(missed)


	### Save nnacp statistics for QM7, svwn 6-31G(d).
	path = '../outputs/QM7/svwn.6-31Gd/wfx_tmp/'
	save_path = '../outputs/QM7/svwn.6-31Gd/stats_nnacp_QM7.csv'
# 	symbol_path = '../datasets/QM7/stats_QM7.csv'
	save_nnacp_stats(path, save_path, itr_mode='filename')



#	#%%
#	# Save nnacp statistics for Diatomic using pbeqidh.
#	path = '../outputs/Diatomic/pbeqidh/wfx_tmp/'
#	save_path = '../outputs/Diatomic/pbeqidh/stats_nnacp_diatomic.csv'
#	save_nnacp_stats(path, save_path, itr_mode='filename')