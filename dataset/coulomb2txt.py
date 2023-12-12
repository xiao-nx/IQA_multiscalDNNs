#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:24:34 2021

@author: ljia
"""

import os
import sys
import time
import numpy as np
from collections import Counter
from tqdm import tqdm
from load_dataset import load_qm7


def coulomb2txt(path):
	"""Save the Coulomb matrices in the QM7 dataset to .txt file.

	Parameters
	----------
	path : string
		path to the dataset.

	Returns
	-------
	None.

	"""
	stime = time.time()

	## Mapping from atomic charges to atom symbols.
# 	charge2sym = {1: 'H', 2: 'he', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca'}
	charge2sym = {6: 'C', 1: 'H', 7: 'N', 8: 'O', 16: 'S'}

	### Extract data infos.
	dataset = load_qm7(path=path)
	charges = dataset['Z']
# 	coords_B = dataset['R']
	coulombs = dataset['X']

	# Create string to save.
	str_save = 'The Coulomb matrix of each molecule in the QM7 dataset.\n\n'


	### For each molecule:
	for i_mol, charge in tqdm(enumerate(charges), desc='Converting molecules', file=sys.stdout):

		### Create the formula.
		# Count atoms.
		cnt = Counter(charge)

# 		# In case the atom type is not considered.
# 		for item in cnt:
# 			if int(item) != 0 and int(item) not in charge2sym:
# 				print('Molecule #' + str(i_mol) + ', atom with charge ' + str(int(item)) + 'is not considered in our program.')

		formu = '' # the formula of the molecule.
		# For each atom type.
		for charg in charge2sym:
			# Create the formula.
			if cnt[charg] > 0:
				formu += charge2sym[charg]
			if cnt[charg] > 1:
				formu += str(int(cnt[charg]))


		### Append the string to save.
		# prefix info.
		str_pre = 'molecule' + str(i_mol + 1) + ': ' + formu + '\n\n'
		# suffix.
		str_suf = '\n\n\n'


		# Get the number of atoms.
		nb_of_atoms = np.count_nonzero(charge)
		# Extract Coulomb matrix.
		clb_mat = coulombs[i_mol][0:nb_of_atoms, :][:, 0:nb_of_atoms]
		# Convert to string.
		str_clb = np.array2string(clb_mat, max_line_width=10e8, threshold=10e8)

# 			# Get the atom symbol.
# 			if int(char) not in charge2sym:
# 				print('Molecule #' + str(i_mol) + ', charge #' + str(int(char)) + ': The charge value' + str(int(char)) + 'is not considered in our program.')

# 			# Get the coordinates.
# 			cors = coords_B[i_mol][i_atom]
# 			str_atoms += ' ' + charge2sym[int(char)] + '               ' \
# 				+ ('' if str(cors[0]).startswith('-') else ' ') + '   {:.8f}'.format(cors[0]) \
# 				+ ('' if str(cors[1]).startswith('-') else ' ') + '   {:.8f}'.format(cors[1]) \
# 				+ ('' if str(cors[2]).startswith('-') else ' ') + '   {:.8f}'.format(cors[2]) + '\n'


		# Create saving string.
		str_save += str_pre + str_clb + str_suf


	### Save file.
# 		# path to save the gjf file.
# 		path_gjf = path
# 		os.makedirs(path_gjf, exist_ok=True)
	# Save.
	with open(os.path.join(path, 'Coulomb_QM7.txt'), 'w') as f:
		f.write(str_save)

	runtime = time.time() - stime
	print('Coulomb matrices of ' + str(charges.shape[0]) + ' molecules extracted in ' + str(runtime) + ' s.')


if __name__ == '__main__':
	if len(sys.argv) > 1:
		path = sys.argv[1]
	else:
		path = '../datasets'

	coulomb2txt(path)

	print('Done.')