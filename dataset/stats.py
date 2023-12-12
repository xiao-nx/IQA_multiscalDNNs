#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:50:49 2021

@author: ljia

This script computes statistics of each dataset and save it to file.
"""

import sys
import time
from collections import Counter
# import numpy as np
from tqdm import tqdm
from load_dataset import load_dataset
import pandas as pd


def qmrxn20_stats(path, **kwargs):
	"""Compute statistics of the QMrxn20 datasets and save it to .csv file,
	including number of each atom type in each molecule and the sum of it,
	the formula of each molecule.

	Parameters
	----------
	path : string, optional
		The path to save the .csv file. The default is '../datasets/QMrxn20/'.

	**kwargs : keyword arguments
		Auxilary arguments for loading the dataset.

	Returns
	-------
	None.

	"""
	stime = time.time()

	## Mapping from atomic charges to atom symbols.
# 	charge2sym = {1: 'H', 2: 'he', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca'}
	sym_list = ['C', 'H', 'N', 'O', 'Cl', 'Br', 'F']

	### Get dataset.
	dataset = load_dataset('qmrxn20')

	### Create panda data frame.
	df = pd.DataFrame(columns=['# mol'] + sym_list + ['Formula', 'States'])

	i_mol = 0

	### For each subset:
	for sset_name, mols in dataset.items():

		print('Subset "' + sset_name + '":')

		### For each molecule:
		for _, mol in tqdm(enumerate(mols), desc='Gathering statistics of molecules', file=sys.stdout):
# 		# name of the molecule.
# 		mol_nf = 'molecule' + str(i_mol + 1) + '.gjf'

			### Count atoms.
			cnt = Counter([atom[0] for atom in mol])

			stats_cur = [i_mol + 1] # statistics of the current molecule.
			formu = '' # the formula of the molecule.
			# For each atom type.
			for sym in sym_list:
				stats_cur.append(cnt[sym])

				# Create the formula.
				if cnt[sym] > 0:
					formu += sym
					if cnt[sym] > 1:
						formu += str(int(cnt[sym]))

			stats_cur.append(formu)
			stats_cur.append(sset_name)

			# Add to table.
			df.loc[int(i_mol + 1)] = stats_cur

			# Write to file in the progress in case of breaking.
			if (i_mol + 1) % 10000 == 0 or (time.time() - stime) % 60 == 0:
				df.to_csv(path + '/stats_QMrxn20.csv', index=True)

			i_mol += 1


	### Sum up each column.
	col_sum = ['Sum']
	nb_tail = 2
	sums = df.iloc[:, 1:-nb_tail].sum(axis=0)
	col_sum += list(sums) + [''] * nb_tail

	df.loc['sum'] = col_sum

	# Change index to the first column.
	df.set_index('# mol', inplace=True)


	### Save to file.
	df.to_csv(path + '/stats_QMrxn20.csv', index=True)


	runtime = time.time() - stime
	print('Statistics of ' + str(i_mol) + ' molecules saved in ' + str(runtime) + ' s.')


def get_all_symbols_qmrxn20(data, verbose=True):
	"""Find out all atom symbols in the QMrxn20 dataset.

	Parameters
	----------
	data : dict
		The dataset in the form of the output of the `load_qmrxn20` function.

	verbose : boolean, optional
		Where to print out the atom symbol list. The default is True.

	Returns
	-------
	sym_list : list
		The atom symbol list.

	"""
	sym_list = []

	### For each subset:
	for sset_name, mols in data.items():

		print('Subset "' + sset_name + '":')

		### For each molecule:
		for _, mol in tqdm(enumerate(mols), desc='Checking atom symbols in molecules', file=sys.stdout):

			### For each atom:
			for atom in mol:
				if atom[0] not in sym_list:
					sym_list.append(atom[0])

	if verbose:
		print('The dataset contains the following atoms: ', sym_list)

	return sym_list





if __name__ == '__main__':
	if len(sys.argv) > 1:
		path = sys.argv[1]
	else:
		path = '../datasets/QMrxn20/'

# 	dataset = load_dataset('qmrxn20')
# 	sym_list = get_all_symbols_qmrxn20(dataset)

	qmrxn20_stats(path)

	print('Done.')

