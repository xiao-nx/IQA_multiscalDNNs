#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:58:56 2021

@author: ljia
This script retrieve all kinds of data from the QM7 dataset.
"""
import sys
import os
# from tqdm import tqdm
from gklearn.utils.iters import get_iters
import numpy as np
# from collections import Counter
from .load_dataset import load_dataset


#%%


def get_coulomb_matrices(path=None, dataset='QM7'):
	"""Retrieve the Coulomb matrices of a dataset.

	Parameters
	----------
	path : string
		path to the dataset.

	Returns
	-------
	The coulomb matrices.

	"""
	if dataset.lower() == 'qm7':
		kwargs = ({'path': path}	 if path is not None else {})
		clb_mats = get_coulomb_mat_qm7(**kwargs)

	return clb_mats


def get_coulomb_mat_qm7(path='../datasets/QM7/'):
	"""Retrieve the Coulomb matrices from the QM7 dataset.

	Parameters
	----------
	path : string
		path to the dataset.

	Returns
	-------
	The coulomb matrices.

	"""

	## Mapping from atomic charges to atom symbols.
# 	charge2sym = {1: 'H', 2: 'he', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca'}
# 	charge2sym = {6: 'C', 1: 'H', 7: 'N', 8: 'O', 16: 'S'}

	### Extract data infos.
	dataset = load_dataset('qm7', path=path)
	charges = dataset['Z']
# 	coords_B = dataset['R']
	coulombs = dataset['X']
	clb_mats = []

	### For each molecule:
	for i_mol, charge in get_iters(enumerate(charges), desc='Retrieving Coulomb matrices', file=sys.stdout, length=len(charges)):

		# Get the number of atoms.
		nb_of_atoms = np.count_nonzero(charge)
		# Extract Coulomb matrix.
		clb_mat = coulombs[i_mol][0:nb_of_atoms, :][:, 0:nb_of_atoms]
		clb_mats.append(clb_mat)

	return clb_mats


#%%


def compute_CME(clb_mats):
	"""Compute eigenvalues for a list of metrices.

	Parameters
	----------
	clb_mats : list
		A list of array.

	Returns
	-------
	cmes : list
		The computed eigenvalues.

	"""
	cmes = []
	for m in clb_mats:
		val, _ = np.linalg.eig(m)
		val[::-1].sort()
		cmes.append(val)

	return cmes


#%%


def get_IQA_matrix():
	pass


def approx_IQA_matrix():
	pass


#%%


def get_molecule_energies(path=None, dataset='QM7', DFT_method='pbeqidh'):
	path_root = (path if path is not None
			  else os.path.join(os.path.dirname(os.path.abspath(__file__)), '../outputs/'))
	path_root += dataset + '/' + DFT_method + '/gaussian/'

	if dataset.lower() == 'qm7':
		energies = get_molecule_energies_qm7(path_root, DFT_method)

	return energies


def get_molecule_energies_qm7(path_root, DFT_method):
	energies = []

	for i_mol in get_iters(range(1, 7165 + 1), desc='Retrieving molecule energies', file=sys.stdout):
		fname = os.path.join(path_root, 'molecule' + str(i_mol) + '.out')
		energies.append(get_molecule_energy_from_out_file(fname, DFT_method))

# 		import random
# 		energies.append(-random.random())# @todo: to change back.
# 		import warnings
# 		warnings.warn('Using random number for test. Please change it back.')

	return energies


def get_molecule_energy_from_out_file(fname, DFT_method):
	with open(fname, 'r') as f:
		lines = f.readlines()

	if DFT_method == 'pbeqidh':

		energy_line = [l for l in lines if 'E2(PBEQIDH)' in l]

		if len(energy_line) > 1:
			raise Exception('There are more than 1 lines containing "E2(PBEQIDH)".')
		if len(energy_line) < 1:
			raise Exception('There is no line containing "E2(PBEQIDH)".')

		energy = float(energy_line[0].split()[-1].replace('D', 'e'))

	return energy


#%%


def get_atomization_energies(path=None, dataset='QM7'):
	"""Retrieve atomization energies of molecules in a dataset.

	Parameters
	----------
	path : string
		path to the dataset.

	Returns
	-------
	Atomization energies.

	"""
	if dataset.lower() == 'qm7':
		kwargs = ({'path': path} if path is not None else {})
		targets = get_atomization_energies_qm7(**kwargs)

	return targets


def get_atomization_energies_qm7(path='../datasets/QM7/'):
	"""Retrieve atomization energies of molecules in the QM7 dataset.

	Parameters
	----------
	path : string
		path to the dataset.

	Returns
	-------
	Targets.

	"""
	dataset = load_dataset('qm7', path=path)
	targets = dataset['T'][0].tolist()

	return targets


#%%


def get_input_data(data_path):
	c_mat = get_coulomb_matrices(data_path)
	cmes = compute_CME(c_mat)
	iqa_mat = get_IQA_matrix()
	iqa_mat_approx = approx_IQA_matrix()

	return c_mat, cmes, iqa_mat, iqa_mat_approx


if __name__ == '__main__':
	if len(sys.argv) > 1:
		data_path = sys.argv[1]
	else:
		data_path = '../datasets'

	X = get_input_data(data_path)

	print('Done.')

