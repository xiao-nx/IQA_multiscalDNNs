#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:36:06 2021

@author: ljia
"""
import sys
import os
import pickle
import numpy as np
import networkx as nx
# from tqdm import tqdm
from gklearn.utils.iters import get_iters


def get_dataset(ds_name, descriptor, remove_sig_errs=True, from_file=True):

	# Read descriptors from .pkl file.
	if from_file:
		path = os.path.dirname(os.path.abspath(__file__)) + '/../datasets/' + ds_name + '/'
		os.makedirs(path, exist_ok=True)
		fn_ds = os.path.join(path, 'ds.' + ds_name + '.' + descriptor + '.pkl')
		if os.path.isfile(fn_ds) and os.path.getsize(fn_ds) != 0:
			with open(fn_ds, 'rb') as f:
				dataset = pickle.load(f)
				return dataset

	if ds_name.lower() == 'qm7':
		dataset = get_dataset_qm7(descriptor, remove_sig_errs=remove_sig_errs)

	else:
		raise Exception('The given dataset name "' + ds_name + '" is not recognized. Possible choices include "QM7".')


	# Save data if required.
	if from_file:
		with open(fn_ds, 'wb') as f:
			pickle.dump(dataset, f)

	return dataset


#%%

def get_dataset_qm7(descriptor, remove_sig_errs=True):

	dataset = {}

	if descriptor.lower() == 'coulomb_matrix':
		dataset['graphs'] = get_qm7_graphs_coulomb_matrix()

	else:
		desc_eigenvecs = ['coulomb_eigenvector', 'self_iqa_eigenvector', 'additive_iqa_eigenvector', 'iqa_eigenvector', 'asympt_iqa_eigenvector', # 0-4
					  # -------------------------- 5-10
					  'eiqa_self_add_sc', 'eiqa_self_add_cs',
					  'eiqa_self_int_vector_sc', 'eiqa_self_int_vector_cs',
					  'eiqa_add_int_vector_sc', 'eiqa_add_int_vector_cs',
					  # -------------------------- 11-15
					  'additive_ke', 'additive_nn', 'additive_ne', 'additive_ee',
					  'additive_knne',
					  # -------------- 16-20, one-body operator decompositions
					  # sum version
					  'ke_atom_sum_nne', 'ne_atom_sum_kne', 'kene_atom_sum_ne',
					  'keneee_atom_sum_nn',
					  # atomic version
					  'kenergy_atom',
					  # ------------------------- 21
					  'emol_gaussian'] # @todo: check name consistency.
		if descriptor.lower() in desc_eigenvecs:
			dataset['X'] = get_qm7_iqa_vectors(descriptor.lower(),
										remove_sig_err=remove_sig_errs)
		else:
			raise ValueError('Descriptor "' + descriptor + '" cannot be recognized. Possible candadites include: "' + '", "'.join(desc_eigenvecs) + '".')


	from .retrieve_data import get_molecule_energies
# 	try:
	dataset['targets'] = get_molecule_energies(dataset='QM7', DFT_method='pbeqidh')
# 	except Exception:
# 		import warnings
# 		warnings.warn('Some molecule energies cannot be retrieved from .out files generated using the PBEQIDH method, temporarily using zeros instead. Please check and fix it.')
# 		dataset['targets'] = [0] * len(dataset['X'])
	# @todo: to change back.
# 	from .retrieve_data import get_atomization_energies
# 	dataset['targets'] = get_atomization_energies(dataset='QM7', DFT_method='pbeqidh')


	if 'desc_eigenvecs' in locals() and descriptor.lower() in desc_eigenvecs:
		### Remove 0 vectors from X.
# 		idx_nonzeros = np.where(dataset['X'].any(axis=1))[0]
		idx_nonzeros = np.where(~np.isnan(dataset['X']).any(axis=1))[0]
		nb_removed = len(dataset['X']) - len(idx_nonzeros)
		dataset['X'] = dataset['X'][idx_nonzeros]
		dataset['targets'] = [dataset['targets'][i] for i in idx_nonzeros]

		print(str(nb_removed) + ' molecules are removed due to various reasons. See exceptions in function get_qm7_coulomb_vector in to_nx.py for more detail.')

	return dataset


def get_qm7_graphs_coulomb_matrix():
	from .retrieve_data import get_coulomb_matrices

	### Get Coulomb matrices.
	clmb_mats = get_coulomb_matrices(dataset='QM7')

	### Create NetworkX graphs.
	graphs = []
	for i_mol, clmb in get_iters(enumerate(clmb_mats), desc='Creating NetworkX graphs', file=sys.stdout, length=len(clmb_mats)):
		# Create graph.
		graph = nx.Graph(name=str(i_mol))

		# @todo: Here we convert matrix values to string to be consistent with the gedlibpy library implementation. It is not a good practice. Modify gedlibpy and remove this.
		# Add nodes.
		for i_atom, row in enumerate(clmb):
			graph.add_node(i_atom, coulomb=str(row[i_atom]))

		# Add edges.
#		# @todo: to change back.
		for i_atom in range(clmb.shape[0]):
			for j_atom in range(i_atom + 1, clmb.shape[0]):
				graph.add_edge(i_atom, j_atom, coulomb=str(clmb[i_atom, j_atom]))

		graphs.append(graph)

	return graphs


def get_qm7_iqa_vectors(descriptor, remove_sig_err=True, verbose=1): # @todo: to change back.
	### Get the maxinum number of atoms in a molecule.
	from .retrieve_data import get_coulomb_matrices
	clmb_mats = get_coulomb_matrices(dataset='QM7')
	nb_atom_max = max([i.shape[0] for i in clmb_mats])
	nb_mols = len(clmb_mats)


	### Get the key string of the descriptor.
	key_str = get_key_string_qm7(descriptor)


	### initialize IQA vectors.
	iqa_vecs, len_vector = initialize_iqa_vectors(descriptor, nb_mols, nb_atom_max)


	### Get vectors from files.
	fpath_sum = '../outputs/QM7/svwn/wfx_tmp/'
	if descriptor.lower() in ['additive_ke', 'additive_nn', 'additive_ne',
						   'additive_ee', 'additive_knne']:
		sfile_path = '../outputs/QM7/svwn/gaussian/'
		sfile_suf = 'out'
		get_iqa_vectors_from_files_qm7(key_str, nb_mols, nb_atom_max, remove_sig_err, verbose, iqa_vecs, len_vector, fpath_sum, sfile_path, sfile_suf)

	# ----------------------------------- one-body operator decompositions

	# sum versions
	elif descriptor.lower() == 'ke_atom_sum_nne':
		# atomic descriptors (energies + number of atoms for each type of atom).
		sfile_path = fpath_sum
		sfile_suf = 'sum'
		atom_type_list = ['C', 'H', 'N', 'O', 'S']
		get_iqa_vectors_from_files_qm7('KE_Atom_Sum', nb_mols, nb_atom_max, remove_sig_err, verbose, iqa_vecs[0], len_vector[0], fpath_sum, sfile_path, sfile_suf,
								  atom_type_list=atom_type_list)

		# additive_KNNE.
		sfile_path = '../outputs/QM7/svwn/gaussian/'
		sfile_suf = 'out'
		get_iqa_vectors_from_files_qm7('additive_KNNE', nb_mols, nb_atom_max, remove_sig_err, verbose, iqa_vecs[1], len_vector[1], fpath_sum, sfile_path, sfile_suf)

		# concatenate two sets of vectors.
		iqa_vecs = np.concatenate((iqa_vecs[0], iqa_vecs[1][:, 1:]), axis=1)

	elif descriptor.lower() == 'ne_atom_sum_kne':
		# atomic descriptors (energies + number of atoms for each type of atom).
		sfile_path = fpath_sum
		sfile_suf = 'sum'
		atom_type_list = ['C', 'H', 'N', 'O', 'S']
		get_iqa_vectors_from_files_qm7('NE_Atom_Sum', nb_mols, nb_atom_max, remove_sig_err, verbose, iqa_vecs[0], len_vector[0], fpath_sum, sfile_path, sfile_suf,
								  atom_type_list=atom_type_list)

		# additive_KNNE.
		sfile_path = '../outputs/QM7/svwn/gaussian/'
		sfile_suf = 'out'
		get_iqa_vectors_from_files_qm7('additive_KNNE', nb_mols, nb_atom_max, remove_sig_err, verbose, iqa_vecs[1], len_vector[1], fpath_sum, sfile_path, sfile_suf)

		# concatenate two sets of vectors.
		iqa_vecs = np.concatenate((iqa_vecs[0], iqa_vecs[1][:, [0, 1, 3]]), axis=1)


	elif descriptor.lower() == 'kene_atom_sum_ne':
		# atomic descriptors (energies + number of atoms for each type of atom).
		sfile_path = fpath_sum
		sfile_suf = 'sum'
		atom_type_list = ['C', 'H', 'N', 'O', 'S']
		get_iqa_vectors_from_files_qm7('KE_Atom_Sum', nb_mols, nb_atom_max, remove_sig_err, verbose, iqa_vecs[0], len_vector[0], fpath_sum, sfile_path, sfile_suf,
								  atom_type_list=atom_type_list)

		# atomic descriptors (energies + number of atoms for each type of atom).
		sfile_path = fpath_sum
		sfile_suf = 'sum'
		atom_type_list = ['C', 'H', 'N', 'O', 'S']
		get_iqa_vectors_from_files_qm7('NE_Atom_Sum', nb_mols, nb_atom_max, remove_sig_err, verbose, iqa_vecs[1], len_vector[1], fpath_sum, sfile_path, sfile_suf,
								  atom_type_list=atom_type_list)

		# additive_KNNE.
		sfile_path = '../outputs/QM7/svwn/gaussian/'
		sfile_suf = 'out'
		get_iqa_vectors_from_files_qm7('additive_KNNE', nb_mols, nb_atom_max, remove_sig_err, verbose, iqa_vecs[2], len_vector[2], fpath_sum, sfile_path, sfile_suf)

		# concatenate two sets of vectors.
		iqa_vecs = np.concatenate((iqa_vecs[0][:, 0:5], iqa_vecs[1],
							 iqa_vecs[2][:, [1, 3]]), axis=1)


	elif descriptor.lower() == 'keneee_atom_sum_nn':
		# descriptors got from the AIMALL .sum file.
		sfile_path = fpath_sum
		sfile_suf = 'sum'
		atom_type_list = ['C', 'H', 'N', 'O', 'S']
		get_iqa_vectors_from_files_qm7('KENEEE_Atom_Sum_NN_sum', nb_mols, nb_atom_max, remove_sig_err, verbose, iqa_vecs[0], len_vector[0], fpath_sum, sfile_path, sfile_suf,
								  atom_type_list=atom_type_list)

		# additive_KNNE.
		sfile_path = '../outputs/QM7/svwn/gaussian/'
		sfile_suf = 'out'
		get_iqa_vectors_from_files_qm7('additive_KNNE', nb_mols, nb_atom_max, remove_sig_err, verbose, iqa_vecs[1], len_vector[1], fpath_sum, sfile_path, sfile_suf)

		E_xc = iqa_vecs[1][:, [3]] - (iqa_vecs[0][:, [-6]] + iqa_vecs[0][:, [-7]])

		# concatenate two sets of vectors.
		iqa_vecs = np.concatenate((iqa_vecs[0][:, 0:20], # 4 sets of decompositions
							 E_xc , # E_{xc}
							 iqa_vecs[0][:, -5:], # nb_atom_types
							 iqa_vecs[1][:, [1]]), # E_{NN}
							axis=1)


	# atomic versions
	elif descriptor.lower() == 'kenergy_atom':
		# atomic descriptors (energy for each atom).
		sfile_path = fpath_sum
		sfile_suf = 'sum'
		get_iqa_vectors_from_files_qm7('KEnergy_Atom', nb_mols, nb_atom_max, remove_sig_err, verbose, iqa_vecs, len_vector, fpath_sum, sfile_path, sfile_suf)

	# ----------------------------------- else

	else:
		sfile_path = fpath_sum
		sfile_suf = 'sum'
		get_iqa_vectors_from_files_qm7(key_str, nb_mols, nb_atom_max, remove_sig_err, verbose, iqa_vecs, len_vector, fpath_sum, sfile_path, sfile_suf)


	return iqa_vecs


def get_key_string_qm7(descriptor):
    ### Get the key string of the descriptor.
	if descriptor == 'coulomb_eigenvector': # @todo: check name consistency.
		key_str = 'Coulomb_vector'
	elif descriptor == 'self_iqa_eigenvector':
		key_str = 'Eiqa_self'
	elif descriptor == 'additive_iqa_eigenvector':
		key_str = 'Eiqa_add'
	elif descriptor == 'iqa_eigenvector':
		key_str = 'IQA_vector'
	elif descriptor == 'asympt_iqa_eigenvector':
		key_str = 'IQA_inf_vector'
	elif descriptor == 'reconst_infos':
		key_str = 'reconst_infos'
	# -----------------------------------
	elif descriptor.lower() == 'eiqa_self_add_sc':
		key_str = 'Eiqa_self_add_sc'
	elif descriptor.lower() == 'eiqa_self_add_cs':
		key_str = 'Eiqa_self_add_cs'
	elif descriptor.lower() == 'eiqa_self_int_vector_sc':
		key_str = 'Eiqa_self_int_vector_sc'
	elif descriptor.lower() == 'eiqa_self_int_vector_cs':
		key_str = 'Eiqa_self_int_vector_cs'
	elif descriptor.lower() == 'eiqa_add_int_vector_sc':
		key_str = 'Eiqa_add_int_vector_sc'
	elif descriptor.lower() == 'eiqa_add_int_vector_cs':
		key_str = 'Eiqa_add_int_vector_cs'
	# -----------------------------------
	elif descriptor.lower() == 'additive_ke':
		key_str = 'additive_KE'
	elif descriptor.lower() == 'additive_nn':
		key_str = 'additive_NN'
	elif descriptor.lower() == 'additive_ne':
		key_str = 'additive_NE'
	elif descriptor.lower() == 'additive_ee':
		key_str = 'additive_EE'
	elif descriptor.lower() == 'additive_knne':
		key_str = 'additive_KNNE'
	# ----------------------------------- one-body operator decomposition.
	# sum verions
	elif descriptor.lower() == 'ke_atom_sum_nne':
		key_str = 'KE_Atom_Sum_NNE'
	elif descriptor.lower() == 'ne_atom_sum_kne':
		key_str = 'NE_Atom_Sum_KNE'
	elif descriptor.lower() == 'kene_atom_sum_ne':
		key_str = 'KENE_Atom_Sum_NE'
	elif descriptor.lower() == 'keneee_atom_sum_nn':
		key_str = 'KENEEE_Atom_Sum_NN'
	# atonic versions
	elif descriptor.lower() == 'kenergy_atom':
		key_str = 'KEnergy_Atom'
	# -----------------------------------
	elif descriptor == 'emol_gaussian':
		key_str = 'Emol_Gaussian'
	else:
		raise ValueError('The given descriptor "' + descriptor + '" cannot be recognized. See available candidiates in the "get_qm7_eigen_vectors" function.')

	return key_str


def initialize_iqa_vectors(descriptor, nb_mols, nb_atom_max):
	### Set the size of returned vectors.
	if descriptor.lower() == 'reconst_infos': # @todo: check name consistency.
		iqa_vecs = [0] * nb_mols
		len_vector = nb_atom_max
	elif descriptor.lower() == 'emol_gaussian':
		iqa_vecs = np.empty((nb_mols, 1))
		len_vector = 1
	elif descriptor.lower() in ['eiqa_self_add_sc', 'eiqa_self_add_cs',
							 'eiqa_self_int_vector_sc',
							 'eiqa_self_int_vector_cs',
							 'eiqa_add_int_vector_sc',
							 'eiqa_add_int_vector_cs']:
		iqa_vecs = np.empty((nb_mols, nb_atom_max * 2))
		len_vector = nb_atom_max * 2
	elif descriptor.lower() in ['additive_ke', 'additive_nn', 'additive_ne',
						   'additive_ee']:
		iqa_vecs = np.empty((nb_mols, 1))
		len_vector = 1
	elif descriptor.lower() == 'additive_knne':
		iqa_vecs = np.empty((nb_mols, 4))
		len_vector = 4

	# ----------------------------------- one-body operator decompositions
	# sum versions
	elif descriptor.lower() in ['ke_atom_sum_nne', 'ne_atom_sum_kne']:
		# 'ke_atom_sum_nne': [Sums of the kinetic energy of each type of atom,
		# numbers of each type of atom, NN, NE, EE];
		# 'ne_atom_sum_kne': [Sums of the nucleus-electron attractive energy of
		# each type of atom, numbers of each type of atom, KE, NN, EE];
		# types of atoms (for QM7): C, H, N, O, S, in this order.
		len_vector = [10, 4]
		iqa_vecs = [np.empty((nb_mols, len_vector[0])),
			  np.empty((nb_mols, len_vector[1]))]

	elif descriptor.lower() == 'kene_atom_sum_ne':
		# [Sums of the kinetic energy of each type of atom,
		# Sums of the nucleus-electron attractive energy of each type of atom,
		# numbers of each type of atom, NN, EE];
		# types of atoms (for QM7): C, H, N, O, S, in this order.
		len_vector = [10, 10, 4]
		iqa_vecs = [np.empty((nb_mols, len_vector[0])),
			  np.empty((nb_mols, len_vector[1])),
			  np.empty((nb_mols, len_vector[2]))]

	elif descriptor.lower() == 'keneee_atom_sum_nn':
		# [ 1) Sums of the kinetic energy of each type of atom,
		# 2) Sums of the nucleus-electron attractive energy of each type of atom,
		# 3) Sums of the Coulombic bi-electronic repulsion inside each type of
		# atom (E_{H,self}),
		# 4) Sums of the half (to prevent double counting) of the sum of all
		# classical repulsions between electrons in each atom and those in all
		# other basins (E_{H,inter}),
		# 5) Remaining term unknown in analytical form (E_{xc} = EE -
		# (\sum{E_{H,self}} + \sum{E_{H,inter}})),
		# 6) numbers of each type of atom,
		# 7) NN];
		# types of atoms (for QM7): C, H, N, O, S, in this order.
		len_vector = [27, 4]
		iqa_vecs = [np.empty((nb_mols, len_vector[0])),
			  np.empty((nb_mols, len_vector[1]))]

	# atomic versions
	elif descriptor.lower() == 'kenergy_atom':
		iqa_vecs = np.empty((nb_mols, nb_atom_max))
		len_vector = nb_atom_max

	else:
		iqa_vecs = np.empty((nb_mols, nb_atom_max))
		len_vector = nb_atom_max

	return iqa_vecs, len_vector


def get_iqa_vectors_from_files_qm7(key_str, nb_mols, nb_atom_max, remove_sig_err, verbose, iqa_vecs, len_vector, fpath_sum, sfile_path, sfile_suf, **kwargs):
	### Get vectors from files.
	from .IQAvectors import IQAvectors

	nb_err = {'NNACP': 0, 'Some Atomic': 0, 'Eint_matrix': 0,
		   'IndexError': 0, 'diff': 0}
# 	for i_mol in get_iters(range(1, 100), desc='Retrieving input descriptors', file=sys.stdout):
# 	for i_mol in get_iters(range(5000, nb_mols + 1), desc='Retrieving input descriptors', file=sys.stdout):
	for i_mol in get_iters(range(1, nb_mols + 1), desc='Retrieving input descriptors', file=sys.stdout): # @todo: to change back.
		fname = os.path.join(sfile_path, 'molecule' + str(i_mol) + '.' + sfile_suf)

		try:
			# Ignore the molecule if the molecular energies computed by Gaussian software and IQA have a significant difference.
			if remove_sig_err:
				fn_sum = os.path.join(fpath_sum, 'molecule' + str(i_mol) + '.sum')
				if IQAvectors(fn_sum, n_max=nb_atom_max, descriptor='reconst_infos', verbose=verbose)['reconst_infos']['sig']:
					iqa_vecs[i_mol - 1] = [np.nan] * len_vector
					nb_err['diff'] += 1
					continue

			iqa_vecs[i_mol - 1] = IQAvectors(fname, n_max=nb_atom_max, descriptor=key_str, verbose=verbose, **kwargs)[key_str]

		except ValueError as e:
			if verbose:
				print('ValueError when reading "molecule' + str(i_mol) + '.' + sfile_suf + '": ' + repr(e) + '\n')
			if 'Charge_nucleus' in repr(e):
				# For now, caused by NNACPs (IQAvectors, lines around 164) (e.g., mol 4 of QM7, 1019 mols in total).
				iqa_vecs[i_mol - 1] = [np.nan] * len_vector
				nb_err['NNACP'] += 1
			elif 'Some Atomic Properties' in repr(e):
				# For now, caused by the error of missing the "Some Atomic Properties" part in the .sum file (IQAvectors, lines around 68) (e.g., mol 4473 of QM7, 13 mols in total).
				iqa_vecs[i_mol - 1] = [np.nan] * len_vector
				nb_err['Some Atomic'] += 1
			elif 'Eint_matrix' in repr(e):
			# For now, caused by the error that Eint_matrix not correctly extracted (missing some pairs) (IQAvectors, lines around 116) (e.g., mol 4998 of QM7, 2 mols in total). (3 mols?)
				iqa_vecs[i_mol - 1] = [np.nan] * len_vector
				nb_err['Eint_matrix'] += 1
			else:
				raise

		except UnboundLocalError as e:
			if verbose:
				print('UnboundLocalError when reading "molecule' + str(i_mol) + '.' + sfile_suf + '": ' + repr(e) + '\n')
			iqa_vecs[i_mol - 1] = [np.nan] * len_vector
			raise

		except IndexError as e:
			# For now, caused by the error that Eint_matrix not correctely extracted (missing some pairs) (IQAvectors, lines around 116) (e.g., mol 4999 of QM7, 2 mols in total). No?
			if verbose:
				print('IndexError when reading "molecule' + str(i_mol) + '.' + sfile_suf + '": ' + repr(e) + '\n')
			iqa_vecs[i_mol - 1] = [np.nan] * len_vector
			nb_err['Eint_matrix'] += 1

		except FileNotFoundError as e:
			if verbose:
				print('FileNotFoundError when reading "molecule' + str(i_mol) + '.' + sfile_suf + '": ' + repr(e) + '\n')
			iqa_vecs[i_mol - 1] = [np.nan] * len_vector
			raise


			#------------------------------------------------------------
# 			fname_6_31Gd = fname.replace('/svwn/', '/svwn.6-31Gd/')
# 			iqa_vecs[i_mol - 1] = IQAvectors(fname_6_31Gd, n_max=nb_atom_max, verbose=False)['Coulomb_vector']
			#------------------------------------------------------------
# 			import warnings
# 			warnings.warn('The Coulomb vector of molecule ' + str(i_mol) + ' cannot be retrieved from .sum file, temporarily using a zero vector instead. Please check to correct it.')

	if verbose:
		print('Number molecules with errors:')
		print(nb_err)

#%%