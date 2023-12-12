#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:05:28 2021

@author: ljia

This script contains methods to save all kinds of datasets.
"""
import time
import os
import sys
from tqdm import tqdm


def save_qmrxn20_to_gjf(data, save_path, method='m062x', out='wfx', ranges_=None):
	"""Save the QMrxn20 data to gjf files.

	Parameters
	----------
	data : dict
		The dataset in the form of the output of the `load_qmrxn20` function.

	save_path : string
		Path to save the .gjf files.

	ranges_ : list of iterable objects, optional
		Designates molecules to be considered, such as lists of indices of molecules. Notice the indices starts from 1 rather than 0. the The default is None.

	Returns
	-------
	None.

	"""
	if method == 'pbeqidh' and out == 'wfx':
		import warnings
		warnings.warn('Are you sure you want to compute .wfx file using "pbeqidh"? It may be quite slow...')

	### For each subset:
	for i_set, (sset_name, mols) in enumerate(data.items()):

		print('Subset "' + sset_name + '":')

		stime = time.time()

		# Create a directory for the subset.
		str_method = '.' + method
#		str_method = ('' if method == 'm062x' else ('.' + method))
		path_gjf = save_path + '/' + sset_name + '/gjf' + str_method + '/'
		os.makedirs(path_gjf, exist_ok=True)


		### useful string for .gjf files.
		str_method = get_str_method(method)
		str_out = ('out=wfx' if out == 'wfx' else '')


		# Get the range of mols.
		if ranges_ is not None and ranges_[i_set] is not None:
#			# Notice the range is sorted as strings rather than numbers to be consistent with load_qmrxn20() method (see _load_qmrxn20_iter_folder()).
#			range_ = [int(i_o) for i_o in sorted([str(i_i) for i_i in ranges_[i_set]])]
			mols_iter = [mols[idx - 1] for idx in ranges_[i_set]]
		else:
			mols_iter = mols

		### For each molecule:
		for i_mol, mol in tqdm(enumerate(mols_iter), desc='Saving molecules to .gjf', file=sys.stdout):

			if ranges_ is not None and ranges_[i_set] is not None:
				idx_mol = ranges_[i_set][i_mol]
			else:
				idx_mol = i_mol + 1

			# name of the molecule.
			mol_nf = 'molecule' + str(idx_mol) + '.gjf'


			### Create strings to save.
			# prefix info.
			str_pre = '%chk=molecule' + str(idx_mol) + '.chk\n%mem=12000mb\n%nprocs=16\n#p ' + str_method + ' ' + str_out + ' \n\nmolecule' + str(idx_mol) + '\n\n'
			if sset_name == 'reactant-conformers':
				str_pre += '0 1\n'
			elif sset_name.startswith('transition-states'):
				str_pre += '-1 1\n'

			# suffix.
			if out == 'wfx':
				str_suf = '\nmolecule' + str(idx_mol) + '.wfx\n\n\n'
			else:
				str_suf = '\n\n\n'


			### Get atom symbols and coordinates.
			str_atoms = ''
			### For each atom:
			for i_atom, atom in enumerate(mol):

				# Add the symbol and the coordinates.
				str_atoms += ' ' + atom[0] + '               ' \
					+ ('' if atom[1].startswith('-') else ' ') + '   ' + atom[1] \
					+ ('' if atom[2].startswith('-') else ' ') + '   ' + atom[2] \
					+ ('' if atom[3].startswith('-') else ' ') + '   ' + atom[3] + '\n'
#				str_atoms += ' ' + atom[0] + '               ' \
#					+ ('' if atom[1].startswith('-') else ' ') + '   {:.8f}'.format(atom[1]) \
#					+ ('' if atom[2].startswith('-') else ' ') + '   {:.8f}'.format(atom[2]) \
#					+ ('' if atom[3].startswith('-') else ' ') + '   {:.8f}'.format(atom[3]) + '\n'


			# Create saving string.
			str_mol = str_pre + str_atoms + str_suf

			### Save file.
#			# path to save the gjf file.
#			path_gjf = os.path.join(path_dir, 'gjf/')
#			os.makedirs(path_gjf, exist_ok=True)
#			# Save.
			with open(os.path.join(path_gjf, mol_nf), 'w') as f:
				f.write(str_mol)

		runtime = time.time() - stime
		len_mols = (len(ranges_[i_set]) if (ranges_ is not None and ranges_[i_set] is not None) else len(mols))
		print(str(len_mols) + ' molecules saved in ' + str(runtime) + ' s.')


def save_qm7_to_gjf(data, save_path, method='m062x', out='wfx', range_=None):
	"""Save the QM7 mat data to gjf files.

	Parameters
	----------
	data : TYPE
		DESCRIPTION.

	save_path : string
		path to the dataset.

	method : TYPE, optional
		DESCRIPTION. The default is 'm062x'.

	out : TYPE, optional
		DESCRIPTION. The default is 'wfx'.

	range_ : iterable object, optional
		Designates molecules to be considered, such as a list of indices of molecules. Notice the indices starts from 1 rather than 0. the The default is None.

	Returns
	-------
	None.

	"""
	if method == 'pbeqidh' and out == 'wfx':
		import warnings
		warnings.warn('Are you sure you want to compute .wfx file using "pbeqidh"? It may be quite slow...')

	stime = time.time()

	## Mapping from atomic charges to atom symbols.
#	charge2sym = {1: 'H', 2: 'he', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca'}
	charge2sym = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S'}

	### Extract data infos.
	charges = data['Z']
	coords_B = data['R']

	# Create save directory.
	str_method = '.' + method
	path_gjf = save_path + '/gjf' + str_method + '/'
	os.makedirs(path_gjf, exist_ok=True)


	### useful string for .gjf files.
	str_method = get_str_method(method)
	str_out = ('out=wfx' if out == 'wfx' else '')


	### For each molecule:
	iterator = (range_ if range_ is not None else range(1, 7165 + 1))
	for i_mol in tqdm(iterator, desc='Converting molecules', file=sys.stdout):
		# name of the molecule.
		mol_nf = 'molecule' + str(i_mol) + '.gjf'

		charge = charges[i_mol - 1]

		### Create strings to save.
		# prefix info.
		str_pre = '%chk=molecule' + str(i_mol) + '.chk\n%mem=12000mb\n%nprocs=16\n#p ' + str_method + ' ' + str_out + ' units=au\n\nmolecule' + str(i_mol) + '\n\n0 1\n'

		# suffix.
		if out == 'wfx':
			str_suf = '\nmolecule' + str(i_mol) + '.wfx\n\n\n'
		else:
			str_suf = '\n\n\n'


		### Get atom symbols and coordinates.
		str_atoms = ''
		# For each atom:
		for i_atom, char in enumerate(charge):
			# Meet filled numbers.
			if char == 0:
				break

			# Get the atom symbol.
			if int(char) not in charge2sym:
				print('Molecule #' + str(i_mol) + ', charge #' + str(int(char)) + ': The charge value' + str(int(char)) + 'is not considered in our program.')

			# Get the coordinates.
			cors = coords_B[i_mol - 1][i_atom]
			str_atoms += ' ' + charge2sym[int(char)] + '               ' \
				+ ('' if str(cors[0]).startswith('-') else ' ') + '   {:.8f}'.format(cors[0]) \
				+ ('' if str(cors[1]).startswith('-') else ' ') + '   {:.8f}'.format(cors[1]) \
				+ ('' if str(cors[2]).startswith('-') else ' ') + '   {:.8f}'.format(cors[2]) + '\n'


		# Create saving string.
		str_mol = str_pre + str_atoms + str_suf

		### Save file.
		with open(os.path.join(path_gjf, mol_nf), 'w') as f:
			f.write(str_mol)

	runtime = time.time() - stime
	print(str(charges.shape[0]) + ' molecules saved in ' + str(runtime) + ' s.')


def get_str_method(method):
	if method == 'm062x':
		str_method = 'm062x 6-31++G(d,p)' # very high accuracy?
	elif method == 'svwn':
		str_method = 'svwn sto-3g' # fast and not accurate.
	elif method == 'svwn.6-31Gd':
		str_method = 'svwn 6-31G(d)' # may solve some nnacp problem caused by "svwn sto-3g".
	elif method == 'svwn.sto-3g.scf=qc': # more robust than svwn 6-31G(d), but much more time consuming.
		str_method = 'svwn sto-3g scf=qc'
	elif method == 'pbeqidh':
		str_method = 'pbeqidh def2tzvp' # more accurate than "svwn sto-3g".
	else:
		import warnings
		warnings.warn('The method cannot be recognized, better be sure it is correct! :)')
		str_method = method

	return str_method


if __name__ == '__main__':
	from load_dataset import load_dataset

	if len(sys.argv) > 1:
		path = sys.argv[1]
	else:
		path = '../datasets/QMrxn20/'


	#%%


#	### Save QMrxn20: svwn sto-3g.
#	ds = load_dataset('QM7') # {'haha': 'hahaha'} #
#	path_save = '../datasets/QM7/'
#	save_qm7_to_gjf(ds, path_save, method='svwn', out='wfx')


	### Save QM7: svwn.6-31Gd
	from check_nnacp import get_nnacp_list_from_csv
	path_nnacp = '../outputs/QM7/svwn/stats_nnacp_qm7.csv'
	idx_nnacp = get_nnacp_list_from_csv(path_nnacp)

	ds = load_dataset('qm7')
	path_save = '../datasets/QM7/'
	save_qm7_to_gjf(ds, path_save, method='svwn.6-31Gd', out='wfx', range_=idx_nnacp)


# 	### Save QM7: pbeqidh def2tzvp.
# 	ds = load_dataset('QM7')
# 	path_save = '../datasets/QM7/'
# 	save_qm7_to_gjf(ds, path_save, method='pbeqidh', out='')


	#%%

# 	### Save QMrxn20: svwn sto-3g.
# 	ds = load_dataset('qmrxn20') # {'haha': 'hahaha'} #
#	path_save = '../datasets/QMrxn20/'
# 	save_qmrxn20_to_gjf(ds, path_save, method='svwn', out='wfx')


# 	### Save QMrxn20: pbeqidh def2tzvp.
# 	ds = load_dataset('qmrxn20')
# 	path_save = '../datasets/QMrxn20/'
# 	save_qmrxn20_to_gjf(ds, path_save, method='pbeqidh', out='')


#	### Save QMrxn20: svwn sto-3g scf=qc.
#	from check_gaussian_termination import get_gaussian_unnormal_list_from_csv

#	#dataset = 'QMrxn20/reactant-conformers'
#	dataset = 'QMrxn20/transition-states/e2'
#	#dataset = 'QMrxn20/transition-states/sn2'
#	method_s = 'svwn.sto-3g.scf=qc'
#	path_unnormal = '../outputs/' + dataset + '/svwn/stats_gaussian_termination_' + dataset.replace('/', '_') + '.csv'
#	idx_unnormal = get_gaussian_unnormal_list_from_csv(path_unnormal)

#	ds = load_dataset('qmrxn20', subtypes=['transition-states/e2'], from_gjf=True) # Make sure to use from_gjf=True, as the raw data were not sorted the first time that I tried to retrieve it. Failing to do so may cause an inconsistence with ranges_. The latest code use sorted when retrieveing the raw data, it will make sure that the order will be always the same, however it is ordered alphabetly rather than numerically.
#	path_save = '../datasets/QMrxn20/'
#	save_qmrxn20_to_gjf(ds, path_save, method=method_s, out='wfx', ranges_=[idx_unnormal])
