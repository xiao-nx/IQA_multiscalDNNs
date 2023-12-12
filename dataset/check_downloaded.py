#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:38:30 2021

@author: ljia
"""
import os
import sys
from tqdm import tqdm


charge2sym = {1: 'h', 6: 'c', 7: 'n', 8: 'o', 16: 's'}


def check_downloaded_QM7_sum(path):
	"""Check if all the summary files computed by Aimall for QM7 are downloaded.

	Parameters
	----------
	path : string
		Path to the downloaded data.

	Returns
	-------
	None.

	"""
	nb_missing = []
	for i in tqdm(range(1, 7166), desc='Checking summary files', file=sys.stdout):
		for ext in ['extout', 'mgpviz', 'sumviz', 'mgp', 'sum', 'wfx']:
			fname = 'molecule' + str(i) + '.' + ext
			if not os.path.isfile(os.path.join(path, fname)):
				print('file "' + fname + '" is not downloaded.')
				nb_missing.append(i)

	print('# mols with missing files: ', set(nb_missing))


# @todo: nna not considered.
def check_downloaded_QM7_atomicfiles(path, ds_path):
	"""Check if all the atomic files computed by Aimall for QM7 are downloaded.

	Parameters
	----------
	path : string
		Path to the downloaded data.

	Returns
	-------
	None.

	"""
	from itertools import product
	import re
	from load_dataset import load_qm7

	### Extract data infos.
	dataset = load_qm7(path=ds_path)
	charges = dataset['Z']

	nb_missing_folder = []
	nb_missing_files = []

	### For each molecule:
	for i_mol, charge in tqdm(enumerate(charges), desc='Checking summary files', file=sys.stdout):
		dir_name = 'molecule' + str(i_mol + 1) + '_atomicfiles/'

		### The whole folder is missing
		if not os.path.isdir(os.path.join(path, dir_name)):
			print('Directory "' + dir_name + '" is not downloaded.')
			nb_missing_folder.append(i_mol + 1)

		### Some files in the folder is missing.
		else:
			# Get number of atoms.
			nb_atoms = len(charge[charge != 0])

			# Get the list of files.
			fn_list = [f for f in os.listdir(os.path.join(path, dir_name)) if os.path.isfile(os.path.join(path, dir_name, f))]

			# Check the missing files:
			prod = product(range(1, nb_atoms + 1), ['int', 'inp'])
			for cnt, ext in prod:

				# Self.
				r = re.compile('[a-z]' + str(cnt) + '.' + ext)
				ex_list = list(filter(r.match, fn_list))

				if len(ex_list) == 0:
					print('file "*'+ str(cnt) + '.' + ext + '" in mol # ' + str(i_mol) + ' is not downloaded.')
					nb_missing_files.append(i_mol)
					break

				# with others.
				for cnt2 in range(cnt + 1, nb_atoms + 1):
					r = re.compile('[a-z]' + str(cnt) + '_[a-z]' + str(cnt2) + '.' + ext)
					ex_list = list(filter(r.match, fn_list))

					if len(ex_list) == 0:
						print('file "*'+ str(cnt) + '_*' + str(cnt2) + '.' + ext + '" in mol # ' + str(i_mol) + ' is not downloaded.')
						nb_missing_files.append(i_mol)
						break


# 			prod = product(range(1, nb_atoms + 1), ['int', 'inp'])
# 			for cnt, ext in prod:

# 				# Self.
# 				is_exist = []
# 				for sym in charge2sym.values():
# 					fname = sym + str(cnt) + '.' + ext
# 					is_exist.append(os.path.isfile(os.path.join(path, dir_name, fname)))
# 				if sum(is_exist) == 0:
# 					print('file "*'+ str(cnt) + '.' + ext + '" in mol # ' + str(i_mol) + ' is not downloaded.')
# 					nb_missing_files.append(i_mol)
# 					break

# 				# with others.
# 				for cnt2 in range(cnt + 1, nb_atoms + 1):
# 					is_exist = []
# 					for sym1 in charge2sym.values():
# 						for sym2 in charge2sym.values():
# 							fname = sym1 + str(cnt) + '_' + sym2 + str(cnt2) + '.' + ext
# 							is_exist.append(os.path.isfile(os.path.join(path, dir_name, fname)))
# 					if sum(is_exist) == 0:
# 						print('file "*'+ str(cnt) + '_*' + str(cnt2) + '.' + ext + '" in mol # ' + str(i_mol) + ' is not downloaded.')
# 						nb_missing_files.append(i_mol)
# 						break



# 			# Sum up the atom symbol.
# 			atom_cnt = {}

# 			# For each atom:
# 			for i_atom, charg in enumerate(charge):
# 				# Meet filled numbers.
# 				if charg == 0:
# 					break

# 				sym = charge2sym[charg]
# 				if sym in atom_cnt:
# 					atom_cnt[sym] += 1
# 				else:
# 					atom_cnt[sym] = 1

# 			# Get all possible files:
# 			fn_list = []
# 			for i_a1, atom in enumerate(atom_cnt.keys()):
# 				# Self.
# 				fn_list += [atom + str(i) for i in range(1, atom_cnt[atom] + 1)]

# 				# Same atom symbol.
# 				comb_a1 = combinations(range(1, atom_cnt[atom] + 1), 2)
# 				for i1, i2 in comb_a1:
# 					fn_list.append(atom + str(i1) + '_' + atom + str(i2))

# 				# With other atoms:
# 				for i_a2, atom2 in enumerate(atom_cnt.keys()):
# 					if i_a2 > i_a1:
# 						prod_a1_a2 = product(range(1, atom_cnt[atom] + 1), range(1, atom_cnt[atom2] + 1))
# 						for i1, i2 in prod_a1_a2:
# 							fn_list.append(atom + str(i1) + '_' + atom2 + str(i2))

# 			# Check the missing files:
# 			for ext in ['int', 'inp']:
# 				for fn in fn_list:
# 					fname = fn + '.' + ext
# 					if not os.path.isfile(os.path.join(path, dir_name, fname)):
# 						print('file "' + fname + '" is not downloaded.')
# 						nb_missing_files.append(i_mol)


	print('# mols missing all files/folder: ', set(nb_missing_folder))
	print('\n# mols missing a part of files: ', set(nb_missing_files))


if __name__ == '__main__':
# 	path = '../outputs/QM7/svmn/wfx_tmp/'
# 	check_downloaded_QM7_sum(path)


	path = '../outputs/QM7/svmn/wfx_tmp/'
	path_ds = '../datasets/QM7/'
	check_downloaded_QM7_atomicfiles(path, path_ds)
