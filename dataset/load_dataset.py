#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:05:19 2021

@author: ljia
"""
import os
import sys
# from tqdm import tqdm
from gklearn.utils.iters import get_iters
from collections import Counter


#%%


def load_dataset(ds_name, **kwargs):
	"""Load pre-defined datasets.

	Parameters
	----------
	ds_name : string
		The name of the dataset. Case insensitive.

	**kwargs : keyword arguments
		Auxilary arguments.

	Returns
	-------
	data : uncertain (depends on the dataset)
		The loaded molecules.

	"""
	if ds_name.lower() == 'qm7':
		data = load_qm7(**kwargs)

	elif ds_name.lower() == 'qmrxn20':
		data = load_qmrxn20(**kwargs)

	return data



# --------------------------------------------
# Load dataset QM7
# --------------------------------------------


def load_qm7(path=None):
	"""Load QM7 dataset.

	QM7 is a subset of GDB-13 (a database of nearly 1 billion
	stable and synthetically accessible organic molecules)
	containing up to 7 heavy atoms C, N, O, and S. The 3D
	Cartesian coordinates of the most stable conformations and
	their atomization energies were determined using ab-initio
	density functional theory (PBE0/tier2 basis set). This dataset
	also provided Coulomb matrices as calculated in [Rupp et al.
	PRL, 2012]:
	Stratified splitting is recommended for this dataset.
	The data file (.mat format, we recommend using `scipy.io.loadmat`
	for python users to load this original data) contains five arrays:
	- "X" - (7165 x 23 x 23), Coulomb matrices
	- "T" - (7165), atomization energies (unit: kcal/mol)
	- "P" - (5 x 1433), cross-validation splits as used in [Montavon et al.
	  NIPS, 2012]
	- "Z" - (7165 x 23), atomic charges (nuclear charge)
	- "R" - (7165 x 23 x 3), cartesian coordinate (unit: Bohr) of each
	 atom in the molecules.


	Parameters
	----------
	path : string, optional
		The path to the qm7.mat file. The default is '../datasets/QM7/'.

	Returns
	-------
	dataset : dict
		The QM7 dataset.


	@Reference:
	.. [1] http://quantum-machine.org/code/nn-qm7.tar.gz.

	"""
	import scipy, scipy.io

	if path is None:
		dir_ = os.path.dirname(os.path.abspath(__file__))
		path = os.path.join(dir_, '../datasets/QM7/')
	elif path.startswith('../'):
		dir_ = os.path.dirname(os.path.abspath(__file__))
		path = os.path.join(dir_, path)

	# --------------------------------------------
	# Load data
	# --------------------------------------------
	if not os.path.exists(path + '/qm7.mat'):
		os.system('wget http://www.quantum-machine.org/data/qm7.mat -P ' + path + '/')
	dataset = scipy.io.loadmat(path + '/qm7.mat')

	return dataset


# --------------------------------------------
# Load dataset QMrxn20
# --------------------------------------------


def load_qmrxn20(path='../datasets/QMrxn20/', subtypes=['reactant-conformers', 'transition-states'], from_gjf=False):
	"""Load QMrxn20 dataset.

	For competing E2 and SN2 reactions, 4'400 validated transition state geometries and 143'200 reactant complex geometries including conformers obtained at MP2/6-311G(d) and DF-LCCSD/cc-pVTZ//MP2/6-311G(d) level of theory are reported. The data covers the chemical compound space spanned by the substituents NO2, CN, CH3, and NH2 and early halogens (F, Cl, Br) as nucleophiles and leaving groups based on an ethane skeleton. Ready-to-use activation energies are given for the different levels of theory where in some cases relaxation effects have been treated with machine learning surrogate models.


	Parameters
	----------
	path : string, optional
		The path to the QMrxn20 dataset. The default is '../datasets/QMrxn20/'.

	subtypes: list
		The sub-types included in the return data. Possible candidates: 'reactant-conformers', 'transition-states'. The default is ['reactant-conformers', 'transition-states'].

	dataset : dict
		The QMrxn20 dataset, where the keys are the subset names.


	@Reference:
	.. [1] https://archive.materialscloud.org/record/2020.55.

	"""
	# --------------------------------------------
	# Download data if needed.
	# --------------------------------------------
	if not os.path.exists(path + '/geometries/'):
		if not os.path.exists(path + '/geometries.tgz'):
			os.system('wget "https://archive.materialscloud.org/record/file?filename=geometries.tgz&record_id=414" -P ' + path + '/')
			os.rename(os.path.join(path + 'file?filename=geometries.tgz&record_id=414'), os.path.join(path + '/geometries.tgz'))
		# Extract.
		extract(os.path.join(path + '/geometries.tgz'))


	dataset = {}

	### Load reactant conformers if required.
	if 'reactant-conformers' in subtypes:
		dataset['reactant-conformers'] = _load_qmrxn20_iter_folder(os.path.join(path, 'geometries/reactant-conformers'), from_gjf=from_gjf)
	### Load transition states if required.
	if 'transition-states' in subtypes or 'transition-states/e2' in subtypes:
		dataset['transition-states/e2'] = _load_qmrxn20_iter_folder(os.path.join(path, 'geometries/transition-states/e2'), from_gjf=from_gjf)
	if 'transition-states' in subtypes or 'transition-states/sn2' in subtypes:
		dataset['transition-states/sn2'] = _load_qmrxn20_iter_folder(os.path.join(path, 'geometries/transition-states/sn2'), from_gjf=from_gjf)


	return dataset


def _load_qmrxn20_iter_folder(root_dir, from_gjf=False):
	"""Traverse all sub-folders in the QMrxn20 subset directory to load the dataset.

	Parameters
	----------
	root_dir : string
		The root directory of QMrxn20.

	Returns
	-------
	ds_list : list
		A list of all molecules in the dataset.

	"""

	ds_list = []

	if not from_gjf:
		for subdir, _, files in get_iters(os.walk(root_dir), desc='Traversing dirs', file=sys.stdout):
			for name in sorted(files):
# 				print(name)
				if name.endswith('.xyz'):
					fname = os.path.join(subdir, name)
					data_cur = _load_qmrxn20_from_xyz_files(fname)
					ds_list.append(data_cur)

	else:
		dir_gjf = root_dir + '/gjf.svwn'
		dir_gjf = dir_gjf.replace('geometries/', '', 1)
		nb_mols = len([name for name in os.listdir(dir_gjf) if name.endswith('.gjf')])
		for i in range(1, nb_mols + 1):
			fn_mol = dir_gjf + '/molecule' + str(i) + '.gjf'
# 			print(fn_mol)
			data_cur = _load_qmrxn20_from_gjf_files(fn_mol)
			ds_list.append(data_cur)


	return ds_list


def _load_qmrxn20_from_xyz_files(fname):
	"""Load a QMrxn20 molecule from the .xyz file.

	Parameters
	----------
	fname : string
		The name of the .xyz file.

	Raises
	------
	Warning
		If the file structure may not be correct. Raised if string "Coordinates from ORCA-job run" is not in the file.

	Returns
	-------
	atom_list : list
		A list of atoms, each item is a list of atom symbol and its x, y, z coordinates.

	"""
	# INTERNAL VARIABLES:
	atom_list = []

	# OPEN FILE:
	with open(fname, 'r') as f:
		lines = f.readlines()  # Convert file into a array of lines

	# ERRORS:
	if 'Coordinates from ORCA-job run' not in lines[1]:
		raise Warning('File structure may not be correct.')  # Checks if atomic list exist inside file

	# GET ATOM LIST:
	for line in lines[2:]:
		split_line = line.split()
		atom_list.append(split_line)

	return atom_list


def _load_qmrxn20_from_gjf_files(fname):
	"""Load a QMrxn20 molecule from the .gjf file.

	Parameters
	----------
	fname : string
		The name of the .gjf file.

	Returns
	-------
	atom_list : list
		A list of atoms, each item is a list of atom symbol and its x, y, z coordinates.

	"""
	# INTERNAL VARIABLES:
	atom_list = []

	# OPEN FILE:
	with open(fname, 'r') as f:
		lines = f.readlines()  # Convert file into a array of lines

	# GET ATOM LIST:
	for line in lines[8:]:
		if line.strip() == '':
			break
		split_line = line.split()
		atom_list.append(split_line)

	return atom_list



#%%


def gjf_to_molecule_formula(filename, system='Hill'):
	atom_list = []

	### Read .gjf file.
	with open(filename) as f:
		content = f.readlines()

		# Read the counts line.
		for l_cur in content[8:]:
			if l_cur.strip() == '':
				break
			atom_list.append(l_cur.split()[0])


	### Convert atom list to molecule formula.
	formu = '' # the formula of the molecule.
	cnt = Counter(atom_list)

	# Order atoms. Use the Hill system.
	sym_list = list(cnt.keys())
	sym_ordered = []
	for sym in ['C', 'H']:
		if sym in sym_list:
			sym_ordered.append(sym)
			sym_list.remove(sym)
	sym_ordered += sorted(sym_list)

	# Create the formula.
	# For each atom type.
	for sym in sym_ordered:
		formu += sym
		if cnt[sym] > 1:
			formu += str(int(cnt[sym]))


	return formu


#%%


def extract(tar_url, extract_path=None):
	"""Unzip a .tgz file.

	Parameters
	----------
	tar_url : string
		The full path of the .tgz file.

	extract_path : string, optional
		Where to extract. The default is the same directory as the zipped file.

	Returns
	-------
	None.

	References
	----------
	.. [1] https://www.codegrepper.com/code-examples/python/extract+tgz+files+in+python.

	"""
	import tarfile

	tar = tarfile.open(tar_url, 'r')
	if extract_path is None:
		extract_path = os.path.splitext(tar_url)[0]

	for item in get_iters(tar, desc='Unzipping file', file=sys.stdout):
		tar.extract(item, extract_path)
		if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
			extract(item.name, './' + item.name[:item.name.rfind('/')])


if __name__ == '__main__':
	#%%

	ds = load_dataset('qmrxn20')



# 	# --------------------------------------------
# 	# Extract training data
# 	# --------------------------------------------
# 	P = dataset['P'][range(0, split) + range(split + 1, 5)].flatten()
# 	X = dataset['X'][P]
# 	T = dataset['T'][0,P]