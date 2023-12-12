#!/usr/bin/env python3
# Computes Coulomb vectors and various IQA vectors for molecular energy
#   representation by machine learning.
# V. Tognetti, November 22nd 2021
# Linlin Jia, November, 25th, 2021
import os
import math
import numpy as np
from numpy import linalg
import pickle
from IPython import get_ipython


def IQAvectors(file_name, n_max=None, descriptor='all', verbose=False, **kwargs):

	# Use absolute path to avoid potential problems.
	if file_name.startswith('../'):
		dir_ = os.path.dirname(os.path.abspath(__file__))
		file_name = os.path.join(dir_, file_name)


	##### descriptors from .out file.
	if descriptor in ['additive_KE', 'additive_NN', 'additive_NE',
				   'additive_EE', 'additive_KNNE']:
		# Get descriptors from .out file.
		try:
			results = IQAvectors_from_out_file(file_name, verbose=verbose)
		except Exception as e:
			raise e

		return results


	##### Get descriptors from .sum file.
	#### ----------------------------------- one-body operator decompositions
	#### sum versions
	### atomic electronic kinetic energy descriptor from .sum file (sumed for
	### each type of atom).
	elif descriptor == 'KE_Atom_Sum':
		try:
			results = IQAvectors_from_sum_file(file_name, verbose=verbose)
		except Exception as e:
			raise e

		results = {'KE_Atom_Sum':
			 _atomic_energy_sum_descriptor_from_dict(results['KEnergy_atom'],
									   kwargs['atom_type_list'])}

		return results

	### atomic nucleus-electron attractive energy descriptor from .sum file
	### (sumed for each type of atom).
	elif descriptor == 'NE_Atom_Sum':
		try:
			results = IQAvectors_from_sum_file(file_name, verbose=verbose)
		except Exception as e:
			raise e

		results = {'NE_Atom_Sum':
			 _atomic_energy_sum_descriptor_from_dict(results['NEnergy_atom'],
									   kwargs['atom_type_list'])}

		return results

	### atomic nucleus-electron attractive energy descriptor from .sum file
	### (sumed for each type of atom).
	elif descriptor == 'NE_Atom_Sum':
		try:
			results = IQAvectors_from_sum_file(file_name, verbose=verbose)
		except Exception as e:
			raise e

		results = {'NE_Atom_Sum':
			 _atomic_energy_sum_descriptor_from_dict(results['NEnergy_atom'],
									   kwargs['atom_type_list'])}

		return results

	#### atomic versions
	### the KENEEE_Atom_Sum_NN descriptor from .sum file (a part, seperated
	### for each atom).
	elif descriptor == 'KENEEE_Atom_Sum_NN_sum':
		try:
			results = IQAvectors_from_sum_file(file_name, verbose=verbose)
		except Exception as e:
			raise e

		KE_atom = \
			_atomic_energy_sum_descriptor_from_dict(results['KEnergy_atom'],
										   kwargs['atom_type_list'])
		NE_atom = \
			_atomic_energy_sum_descriptor_from_dict(results['NEnergy_atom'],
										   kwargs['atom_type_list'])
		EH_self = \
			_atomic_energy_sum_descriptor_from_dict(results['EH_self_atom'],
										   kwargs['atom_type_list'])
		EH_inter = \
			_atomic_energy_sum_descriptor_from_dict(results['EH_inter_atom'],
										   kwargs['atom_type_list'])
		nb_atom_types = len(kwargs['atom_type_list'])
		results = {'KENEEE_Atom_Sum_NN_sum':
			 KE_atom[0:nb_atom_types] + NE_atom[0:nb_atom_types] \
			+ EH_self[0:nb_atom_types] + EH_inter[0:nb_atom_types] \
			+ [results['EH_self_total'], results['EH_inter_total']] \
			+ KE_atom[nb_atom_types:]
			}

		return results


	### other descriptors from .sum file.
	else:
		path, fname = os.path.split(os.path.abspath(file_name))
		path_pkl = path + '/results_pkl'
		fn_pkl = os.path.join(path_pkl, '.'.join(fname.split('.')[:-1]) + '.pkl')

		# Read descriptors from .pkl file.
		# @todo: dosn't work if descriptor is a mixed one.
		if os.path.isfile(fn_pkl) and os.path.getsize(fn_pkl) != 0:
			with open(fn_pkl, 'rb') as f:
				results = pickle.load(f)

			# Return if the required descriptor exists.
			if descriptor == 'all' or descriptor in results:
				_pad_vectors(results, n_max, results['n_atoms'], descriptor=descriptor)
				return results

		# Get descriptors from .sum file.
		try:
			results = IQAvectors_from_sum_file(file_name, verbose=verbose)
		except Exception as e:
			raise e
		# Save descriptors to .pkl file.
		os.makedirs(path_pkl, exist_ok=True)
		with open(fn_pkl, 'wb') as f:
			pickle.dump(results, f)

		# Pad vectors with 0 values.
		_pad_vectors(results, n_max, results['n_atoms'], descriptor=descriptor)
		return results


def _atomic_energy_sum_descriptor_from_dict(ke_atoms, atom_type_list, with_nb_atoms=True):
	nb_atom_type = len(atom_type_list)
	ke_atom_sum_vec = [0] * (nb_atom_type * 2)
	for key, val in ke_atoms.items(): # @todo: maybe sum up and then assign will be faster.
		a_type = ''.join([i for i in key if not i.isdigit()])
		idx_type = atom_type_list.index(a_type)
		ke_atom_sum_vec[idx_type] += val
		ke_atom_sum_vec[idx_type + nb_atom_type] += 1

	return ke_atom_sum_vec


def _atomic_energy_descriptor_from_dict(ke_atoms, n_max):
	ke_atom_vec = [0] * n_max
	for i, (key, val) in enumerate(ke_atoms.items()): # @todo: maybe sum up and then assign will be faster.
		ke_atom_vec[i] = val
	return ke_atom_vec


def IQAvectors_from_out_file(file_name, verbose=False):
	with open(file_name, 'r') as f:
		lines = f.readlines()

	# additive_KE, additive_EE.
	for idx, line in enumerate(lines):
		if line.strip().startswith('KE='):
			l_KE = line.strip()
			break
	idx_PE = l_KE.index('PE=')
	idx_EE = l_KE.index('EE=')
	additive_KE = [float(l_KE[3:idx_PE].strip().replace('D', 'E'))]
	additive_EE = [float(l_KE[idx_EE+3:].strip().replace('D', 'E'))]

	# additive_NN, additive_NE.
	for idx, line in enumerate(lines[idx+1:]):
		if line.strip().startswith('N-N='):
			l_NN = line.strip()
			break
	idx_EN = l_NN.index('E-N=')
	idx_KE = l_NN.index('KE=')
	additive_NN = [float(l_NN[4:idx_EN].strip().replace('D', 'E'))]
	additive_NE = [float(l_NN[idx_EN+4:idx_KE].strip().replace('D', 'E'))]

	additive_KNNE = additive_KE + additive_NN + additive_NE + additive_EE

	outputs = {'additive_KE': additive_KE,
			'additive_NN': additive_NN,
			'additive_NE': additive_NE,
			'additive_EE': additive_EE,
			'additive_KNNE': additive_KNNE}
	return outputs


def _pad_vectors(results, n_max, n_atoms, descriptor='all'):

	##########################################################
	### Completing vectors with 0 values #####################
	##########################################################

	if n_max is not None:
# 		if descriptor == 'all' or 'Eiqa_self_add_sc' in descriptor:
# 			results['Eiqa_self_add_sc'] = np.concatenate((
# 				results['Eiqa_self_add_sc'][0:len(results['Eiqa_self'])],
# 				np.zeros((n_max - len(results['Eiqa_self']),)),
# 				results['Eiqa_self_add_sc'][len(results['Eiqa_self']):],
# 				np.zeros((n_max - len(results['Eiqa_self']),)),
# 				))
# 		if descriptor == 'all' or 'Eiqa_self_add_cs' in descriptor:
# 			results['Eiqa_self_add_cs'] = sorted(np.concatenate((
# 				results['Eiqa_self_add_sc'],
# 				np.zeros((n_max * 2 - len(results['Eiqa_self_add_sc']),)),
# 				)))
		for i in range(n_max - n_atoms):
			results['Coulomb_vector'] = np.append(results['Coulomb_vector'], 0.0)
			results['Eiqa_self'] = np.append(results['Eiqa_self'], 0.0)
			results['Eiqa_add'] = np.append(results['Eiqa_add'], 0.0)
			results['IQA_vector'] = np.append(results['IQA_vector'], 0.0)
			results['IQA_int_vector'] = np.append(results['IQA_int_vector'], 0.0)
			results['IQA_inf_vector'] = np.append(results['IQA_inf_vector'], 0.0)

		# @todo: check name consistency.
		if descriptor == 'all' or 'Eiqa_self_add_sc' in descriptor:
			results['Eiqa_self_add_sc'] = np.concatenate((results['Eiqa_self'], results['Eiqa_add']))

		elif descriptor == 'all' or 'Eiqa_self_add_cs' in descriptor:
			results['Eiqa_self_add_cs'] = np.sort(np.concatenate((results['Eiqa_self'], results['Eiqa_add'])))

		elif descriptor == 'all' or 'Eiqa_self_int_vector_sc' in descriptor:
			results['Eiqa_self_int_vector_sc'] = np.concatenate((results['Eiqa_self'], results['IQA_int_vector']))

		elif descriptor == 'all' or 'Eiqa_self_int_vector_cs' in descriptor:
			results['Eiqa_self_int_vector_cs'] = np.sort(np.concatenate((results['Eiqa_self'], results['IQA_int_vector'])))

		elif descriptor == 'all' or 'Eiqa_add_int_vector_sc' in descriptor:
			results['Eiqa_add_int_vector_sc'] = np.concatenate((results['Eiqa_add'], results['IQA_int_vector']))

		elif descriptor == 'all' or 'Eiqa_add_int_vector_cs' in descriptor:
			results['Eiqa_add_int_vector_cs'] = np.sort(np.concatenate((results['Eiqa_add'], results['IQA_int_vector'])))


#%%


def IQAvectors_from_sum_file(file_name, verbose=False):

##########################################################
### Reading user's specifications ########################
##########################################################

# file_name = '/home/ljia/Téléchargements/example_ts/transition_state/molecule1.sum'
# n_max = 12

	##########################################################
	### Extracting data from AIMAll output ###################
	##########################################################

	# Initializing lists
	Charge_nucleus = []
	X_nucleus = []; Y_nucleus = []; Z_nucleus = []
	Charge_atom = []; KEnergy_atom = {}; C6_atom = []
	Eiqa_self = []; Eiqa_add = []
	EH_self_atom = {}; EH_inter_atom = {}

	# Getting the number of atoms in the molecule
	sum_file = open(file_name,'r')
	for num, i in enumerate(sum_file):
		if 'Nuclear Charges' in i:
			numline1 = num
		# Line with "Some Atomic Properties" may not exist.
# 		if 'Some Atomic Properties' in i:
# 			numline2 = num
			break
	for i in sum_file:
		num += 1
		if i.strip() == '':
			numline2 = num
			break
# 	n_atoms = numline2 - numline1 - 5
	n_atoms = numline2 - numline1 - 4
	n_pairs = int(n_atoms * (n_atoms - 1) / 2)

	# Reading nuclear charges and Cartesian coordinates in bohr
	sum_file.seek(0)
	for i in range(numline1 + 4):
		sum_file.readline()
	for i in range(n_atoms):
		t = sum_file.readline(); lineDecomp = t.split()
		Charge_nucleus.append(lineDecomp[1])
		X_nucleus.append(float(lineDecomp[2]))
		Y_nucleus.append(float(lineDecomp[3]))
		Z_nucleus.append(float(lineDecomp[4]))

	# Reading atomic charges and kinetic energy.
	for i in range(11):
		sum_file.readline()
	try:
		for i in range(n_atoms):
			t = sum_file.readline(); lineDecomp = t.split()
			KEnergy_atom[lineDecomp[0]] = float(lineDecomp[3])
			Charge_atom.append(float(lineDecomp[1]))
	except ValueError as e:
# 		print('The "Some Atomic Properties" part is not shown in the .sum file: ' + repr(e))
		raise ValueError('The "Some Atomic Properties" part is not shown in the .sum file: ' + repr(e))


	# Reading Gaussian molecular energy.
	sum_file.seek(0)
	for num, i in enumerate(sum_file):
		if 'Molecular energy' in i:
			numline3 = num
			break
	sum_file.seek(0)
	for i in range(numline3):
		sum_file.readline()
	t = sum_file.readline(); lineDecomp = t.split()
	Emol_Gaussian = float(lineDecomp[7])


	# Reading IQA additive energy.
	sum_file.seek(0)
	for num, i in enumerate(sum_file):
		if 'IQA Additive' in i:
			break
	for i in range(14):
		sum_file.readline()
	for i in range(n_atoms):
		t = sum_file.readline(); lineDecomp = t.split()
		Eiqa_add.append(float(lineDecomp[1]))
	Emol_IQA = sum(Eiqa_add)
	Reconst_error = abs(Emol_IQA - Emol_Gaussian)


	# Reading IQA self energy and Coulomb Part of Vee(A,A) (Two-electron
	# Interaction Energy of Atom A With Itself).
	sum_file.seek(0)
	for num, i in enumerate(sum_file):
		if 'More IQA Intraatomic' in i:
			numline5 = num
			break
	sum_file.seek(0)
	for i in range(numline5 - n_atoms - 3):
		sum_file.readline()
	for i in range(n_atoms):
		t = sum_file.readline(); lineDecomp = t.split()
		Eiqa_self.append(float(lineDecomp[1]))
		EH_self_atom[lineDecomp[0]] = float(lineDecomp[5])
	# Reading the sum of EH_self_atom.
	sum_file.readline()
	EH_self_total = float(sum_file.readline().split()[5])


	# Reading Coulomb Part of Vee(A,A')/2 (Half of Two-electron Interaction
	# Energy Between Atom A and Other Atoms of Molecule).
	sum_file.seek(0)
	for num, i in enumerate(sum_file):
		if 'More IQA Atomic Contributions to Interatomic' in i:
			numline_Hinter = num
			break
	sum_file.seek(0)
	for i in range(numline_Hinter - n_atoms - 3):
		sum_file.readline()
	for i in range(n_atoms):
		t = sum_file.readline(); lineDecomp = t.split()
		EH_inter_atom[lineDecomp[0]] = float(lineDecomp[6])
	# Reading the sum of EH_inter_atom.
	sum_file.readline()
	EH_inter_total = float(sum_file.readline().split()[6])


	# Reading Eij interatomic energies
	Eint_matrix = np.zeros((n_atoms, n_atoms))
	sum_file.seek(0)
	for num, i in enumerate(sum_file):
		if 'More IQA Diatomic' in i:
			numline6 = num
			break
	sum_file.seek(0)
	for i in range(numline6 - n_pairs - 2):
		sum_file.readline()
	try:
		for j in range(1, n_atoms):
			for i in range(0, j):
				t = sum_file.readline(); lineDecomp = t.split()
				Eint_matrix[i, j] = lineDecomp[2]
	except IndexError as e:
		raise IndexError('Eint_matrix not correctly extracted: ' + repr(e))
	except ValueError as e:
		raise ValueError('Eint_matrix not correctly extracted: ' + repr(e))
	Eint_matrix = Eint_matrix + Eint_matrix.T


	# Reading delocalization indices
	Deloc_matrix = np.zeros((n_atoms, n_atoms))
	sum_file.seek(0)
	for num, i in enumerate(sum_file):
		if 'Diatomic Electron Pair' in i:
			break
	for i in range(14):
		sum_file.readline()
	for j in range(1, n_atoms):
		for i in range(0, j):
			t = sum_file.readline(); lineDecomp = t.split()
			Deloc_matrix[i, j] = lineDecomp[3]
	Deloc_matrix = Deloc_matrix + Deloc_matrix.T


	# Reading nucleus-electron attractive energies.
	NEnergy_atom = {}
	sum_file.seek(0)
	for i, l in enumerate(sum_file):
		if 'Virial-Based Atomic Energy Components' in l:
			break
	for i in range(20):
		sum_file.readline()
	for i in range(n_atoms):
		t = sum_file.readline(); lineDecomp = t.split()
		NEnergy_atom[lineDecomp[0]] = float(lineDecomp[3])

	# Close file.
	sum_file.close()


	##########################################################
	### Calculating interaction matrices #####################
	##########################################################

	Coulomb_matrix = np.zeros((n_atoms, n_atoms))
	Edisp_matrix = np.zeros((n_atoms, n_atoms))
	IQA_matrix = np.zeros((n_atoms, n_atoms))
	IQA_inf_matrix = np.zeros((n_atoms, n_atoms))
	IQA_inf_woDisp_matrix = np.zeros((n_atoms, n_atoms))

	# Evaluating C6 coefficients. Initial values are in J*mol^-1*nm^6.
	# A scaling factor is applied to convert them in Hartree*bohr^6.
	# Values from J. Comput. Chem. 27 (2006) 1787.
	scale = 0.001 / 2625.51 * pow(18.897, 6)
	for i in range(n_atoms):
		if (Charge_nucleus[i] == '1.0'): C6_atom.append(0.14 * scale)
		elif (Charge_nucleus[i] == '6.0'): C6_atom.append(1.75 * scale)
		elif (Charge_nucleus[i] == '7.0'): C6_atom.append(1.23 * scale)
		elif (Charge_nucleus[i] == '8.0'): C6_atom.append(0.70 * scale)
		elif (Charge_nucleus[i] == '9.0'): C6_atom.append(0.75 * scale)
		elif (Charge_nucleus[i] == '16.0'): C6_atom.append(5.57 * scale)
		elif (Charge_nucleus[i] == '17.0'): C6_atom.append(5.07 * scale)
		elif (Charge_nucleus[i] == '35.0'): C6_atom.append(12.47 * scale)
		else:
			raise ValueError('Charge_nucleus of atom ' + str(i) + ' is not expected: ' + str(Charge_nucleus[i]) + '. Please check if NNACP is introduced.')

	# Computing asymptotic interaction energies according to:
	#	2IQA_inf[i,j] = Qi*Qj/Rij - DIij/(2*Rij) - C6ij/Rij^6.
	# C6ij is chosen as the geometry mean of the atomic C6.
	# IQA_inf_woDisp_matrix does not incude dispersion and is
	#	used to calculate the asymptotic error.
	# For the Coulomb matrix: Coulomb[i,j]=-Zi*Zj/Rij
	# The IQA diagonal elements are the self energies.
	# The Coulomb diagonal elements are -0.5*Z^2.4.
	Edisp = 0.0
	Asympt_error_matrix = 0.0
	for i in range(n_atoms):
		Coulomb_matrix[i, i] = -0.5 * pow(float(Charge_nucleus[i]), 2.4)
		IQA_matrix[i, i] = Eiqa_self[i]
		IQA_inf_matrix[i, i] = Eiqa_self[i]
		for j in range(n_atoms):
			if (j != i):
				IQA_matrix[i, j] = 0.5 * Eint_matrix[i, j]
				Rij = math.sqrt(pow(X_nucleus[i] - X_nucleus[j], 2) + \
								pow(Y_nucleus[i] - Y_nucleus[j], 2) + \
								pow(Z_nucleus[i] - Z_nucleus[j], 2))
				C6ij = math.sqrt(C6_atom[i] * C6_atom[j])
				Coulomb_matrix[i, j] = -float(Charge_nucleus[i]) * \
									  float(Charge_nucleus[j]) / Rij
				IQA_inf_woDisp_matrix[i, j] = 0.5 * ( \
									  Charge_atom[i] * Charge_atom[j] / Rij - \
									  Deloc_matrix[i, j] / (2.0 * Rij))
				IQA_inf_matrix[i, j] = IQA_inf_woDisp_matrix[i,j ] - \
									  0.5 * C6ij / pow(Rij, 6)
				Edisp += -0.5 * C6ij / pow(Rij, 6)
	Asympt_error_matrix = abs(IQA_inf_matrix - IQA_matrix)
	Asympt_error_matrix_value = np.sum(Asympt_error_matrix) / (2*n_pairs)

	##########################################################
	### Computing energy vectors #############################
	##########################################################

	Eiqa_self.sort()
	Eiqa_add.sort()
	Coulomb_vector = linalg.eigvalsh(Coulomb_matrix)
	Coulomb_vector.sort()
	IQA_vector = linalg.eigvalsh(IQA_matrix)
	IQA_vector.sort()
	IQA_int_vector = linalg.eigvalsh(Eint_matrix / 2) # @todo: to write in function comment: notice that this matrix is divided by 2.
	IQA_int_vector.sort()
	IQA_inf_vector = linalg.eigvalsh(IQA_inf_matrix)
	IQA_inf_vector.sort()
	Asympt_error_eig1 = abs(IQA_vector-IQA_inf_vector)
	Asympt_error_eig = sum(Asympt_error_eig1)/n_atoms

	##########################################################
	### Displaying results ###################################
	##########################################################

	if verbose:
		if (Reconst_error > 0.001): print('	=> Significant reconstruction error')
# 		if (Reconst_error <= 0.001): print('	=> Accurate energy reconstruction')
		if verbose >= 2:
			get_ipython().magic('clear')
			float_formatter = lambda x: "%.6f" % x
			np.set_printoptions(formatter={'float_kind':float_formatter})
			print()
			print(); print()
			print('_______________________________________________________________')
			print('****************** All values in atomic units *****************')
			print(); print('File:', file_name); print()
			print('Gaussian molecular energy = ', round(Emol_Gaussian, 6))
			print('IQA molecular energy	  = ', round(Emol_IQA, 6))
			print('Reconstruction error	  = ', round(Reconst_error, 6))

			print('Grimme dispersion energy  = ', round(Edisp, 6))
			print(); print('Mean asymptotic error on IQA matrix	  = ', \
						   round(Asympt_error_matrix_value, 6))
			print('Mean asymptotic error on IQA eigenvalues = ', \
						   round(Asympt_error_eig, 6))
			print(); print('	Sorted Coulomb vector:')
			print(Coulomb_vector); print()
			print('	Sorted self IQA vector:')
			print(np.round(Eiqa_self, 6)); print()
			print('	Sorted additive IQA vector:')
			print(np.round(Eiqa_add, 6)); print()
			print('	Sorted IQA eigenvector:')
			print(np.round(IQA_vector, 6)); print()
			print('	Sorted asympt. IQA eigenvector:')
			print(np.round(IQA_inf_vector, 6)); print()
			print('_______________________________________________________________')
			print(); print()


# 	##########################################################
# 	### Completing vectors with 0 values #####################
# 	##########################################################

# 	if n_max is not None:
# 		for i in range(n_max - n_atoms):
# 			Coulomb_vector = np.append(Coulomb_vector, 0.0)
# 			Eiqa_self = np.append(Eiqa_self, 0.0)
# 			Eiqa_add = np.append(Eiqa_add, 0.0)
# 			IQA_vector = np.append(IQA_vector, 0.0)
# 			IQA_inf_vector = np.append(IQA_inf_vector, 0.0)


	##########################################################
	### Return vectors #####################
	##########################################################

	outputs = {'Coulomb_vector': Coulomb_vector, 'Eiqa_self': Eiqa_self,
		 'Eiqa_add': Eiqa_add, 'IQA_vector': IQA_vector,
		 'IQA_int_vector': IQA_int_vector,
		 'IQA_inf_vector': IQA_inf_vector,
		 ### ------------------------------
		 'reconst_infos': {'sig': Reconst_error > 0.001, 'error': Reconst_error, 'Emol_IQA': Emol_IQA, 'Emol_Gaussian': Emol_Gaussian},
		 'Emol_Gaussian': Emol_Gaussian, 'n_atoms': n_atoms,
		 ### ------------------------------ one-body operator decompositions.
		 'KEnergy_atom': KEnergy_atom,
		 'NEnergy_atom': NEnergy_atom,
		 'EH_self_atom': EH_self_atom, 'EH_self_total': EH_self_total,
		 'EH_inter_atom': EH_inter_atom, 'EH_inter_total': EH_inter_total,
		 } # @todo: check name consistency.

	return outputs
