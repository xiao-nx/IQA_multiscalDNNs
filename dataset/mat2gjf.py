#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:13:35 2021

@author: ljia
"""
import os
import sys
import time
# import numpy as np
from tqdm import tqdm
from load_dataset import load_qm7


def mat2gjf(path):
	"""Save the QM7 mat data to gjf files.

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
	charge2sym = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S'}

	### Extract data infos.
	dataset = load_qm7(path=path)
	charges = dataset['Z']
	coords_B = dataset['R']


	### For each molecule:
	for i_mol, charge in tqdm(enumerate(charges), desc='Converting molecules', file=sys.stdout):
		# name of the molecule.
		mol_nf = 'molecule' + str(i_mol + 1) + '.gjf'


		### Create strings to save.
		# prefix info.
		str_pre = '%chk=molecule' + str(i_mol + 1) + '.chk\n%mem=12000mb\n%nprocs=16\n#p m062x 6-31++G(d,p) out=wfx units=au\n\nmolecule' + str(i_mol + 1) + '\n\n0 1\n'
		# suffix.
		str_suf = '\nmolecule' + str(i_mol + 1) + '.wfx\n\n\n'


		### Get atom symbols and coordinates.
		str_atoms = ''
		# For each atom:
		for i_atom, char in enumerate(charge):
			# Meet filled numbers.
			if char == 0:
				break

			# Get the atom symbol.
			if int(char) not in charge2sym:
				print('Molecule #' + str(i_mol + 1) + ', charge #' + str(int(char)) + ': The charge value' + str(int(char)) + 'is not considered in our program.')

			# Get the coordinates.
			cors = coords_B[i_mol][i_atom]
			str_atoms += ' ' + charge2sym[int(char)] + '               ' \
				+ ('' if str(cors[0]).startswith('-') else ' ') + '   {:.8f}'.format(cors[0]) \
				+ ('' if str(cors[1]).startswith('-') else ' ') + '   {:.8f}'.format(cors[1]) \
				+ ('' if str(cors[2]).startswith('-') else ' ') + '   {:.8f}'.format(cors[2]) + '\n'


		# Create saving string.
		str_mol = str_pre + str_atoms + str_suf

		### Save file.
		# path to save the gjf file.
		path_gjf = path + '/gjf/'
		os.makedirs(path_gjf, exist_ok=True)
		# Save.
		with open(os.path.join(path_gjf, mol_nf), 'w') as f:
			f.write(str_mol)

	runtime = time.time() - stime
	print(str(charges.shape[0]) + ' molecules saved in ' + str(runtime) + ' s.')


if __name__ == '__main__':
	if len(sys.argv) > 1:
		path = sys.argv[1]
	else:
		path = '../datasets/QM7/'

	mat2gjf(path)

	print('Done.')