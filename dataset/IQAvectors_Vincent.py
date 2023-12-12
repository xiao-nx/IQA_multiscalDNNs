#!/usr/bin/env python3
# Computes Coulomb vectors and various IQA vectors for molecular energy
#   representation by machine learning.
# V. Tognetti, November 22nd 2021

import math
import numpy as np
from numpy import linalg
from IPython import get_ipython

##########################################################
### Reading user's specifications ########################
##########################################################

file_name = '../outputs/QM7/svwn/wfx_tmp/molecule1.sum'
n_max = 12

##########################################################
### Extracting data from AIMAll output ###################
##########################################################

# Initializing lists
Charge_nucleus = []
X_nucleus = []; Y_nucleus = []; Z_nucleus = []
Charge_atom = []; C6_atom = []
Eiqa_self = []; Eiqa_add = []

# Getting the number of atoms in the molecule
sum_file = open(file_name,'r')
for num, i in enumerate(sum_file):
    if 'Nuclear Charges' in i: numline1 = num
    if 'Some Atomic Properties' in i: numline2 = num
n_atoms = numline2-numline1-5
n_pairs = int(n_atoms*(n_atoms-1)/2)

# Reading nuclear charges and Cartesian coordinates in bohr
sum_file.seek(0)
for i in range(numline1+4):
    sum_file.readline()
for i in range(n_atoms):
    t = sum_file.readline(); lineDecomp = t.split()
    Charge_nucleus.append(lineDecomp[1])
    X_nucleus.append(float(lineDecomp[2]))
    Y_nucleus.append(float(lineDecomp[3]))
    Z_nucleus.append(float(lineDecomp[4]))

# Reading atomic charges
for i in range(11):
    sum_file.readline()
for i in range(n_atoms):
    t = sum_file.readline(); lineDecomp = t.split()
    Charge_atom.append(float(lineDecomp[1]))

# Reading Gaussian molecular energy
sum_file.seek(0)
for num, i in enumerate(sum_file):
    if 'Molecular energy' in i: numline3 = num
sum_file.seek(0)
for i in range(numline3):
    sum_file.readline()
t = sum_file.readline(); lineDecomp = t.split()
Emol_Gaussian = float(lineDecomp[7])

# Reading IQA additive and self energies
sum_file.seek(0)
for num, i in enumerate(sum_file):
    if 'IQA Additive' in i: numline4 = num
    if 'IQA Intraatomic' in i: numline5 = num
sum_file.seek(0)
for i in range(numline4+15):
    sum_file.readline()
for i in range(n_atoms):
    t = sum_file.readline(); lineDecomp = t.split()
    Eiqa_add.append(float(lineDecomp[1]))
Emol_IQA = sum(Eiqa_add)
Reconst_error = abs(Emol_IQA-Emol_Gaussian)
sum_file.seek(0)
for i in range(numline5-n_atoms-3):
    sum_file.readline()
for i in range(n_atoms):
    t = sum_file.readline(); lineDecomp = t.split()
    Eiqa_self.append(float(lineDecomp[1]))

# Reading Eij interatomic energies
Eint_matrix = np.zeros((n_atoms,n_atoms))
sum_file.seek(0)
for num, i in enumerate(sum_file):
    if 'More IQA Diatomic' in i: numline6 = num
sum_file.seek(0)
for i in range(numline6-n_pairs-2):
    sum_file.readline()
for j in range(1,n_atoms):
    for i in range(0,j):
        t = sum_file.readline(); lineDecomp = t.split()
        Eint_matrix[i,j] = lineDecomp[2]
Eint_matrix = Eint_matrix + Eint_matrix.T

# Reading delocalization indices
Deloc_matrix = np.zeros((n_atoms,n_atoms))
sum_file.seek(0)
for num, i in enumerate(sum_file):
    if 'Diatomic Electron Pair' in i: numline7 = num
sum_file.seek(0)
for i in range(numline7+15):
    sum_file.readline()
for j in range(1,n_atoms):
    for i in range(0,j):
        t = sum_file.readline(); lineDecomp = t.split()
        Deloc_matrix[i,j] = lineDecomp[3]
Deloc_matrix = Deloc_matrix + Deloc_matrix.T
sum_file.close()

##########################################################
### Calculating interaction matrices #####################
##########################################################

Coulomb_matrix = np.zeros((n_atoms,n_atoms))
Edisp_matrix = np.zeros((n_atoms,n_atoms))
IQA_matrix = np.zeros((n_atoms,n_atoms))
IQA_inf_matrix = np.zeros((n_atoms,n_atoms))
IQA_inf_woDisp_matrix = np.zeros((n_atoms,n_atoms))

# Evaluating C6 coefficients. Initial values are in J*mol^-1*nm^6.
# A scaling factor is applied to convert them in Hartree*bohr^6.
# Values from J. Comput. Chem. 27 (2006) 1787.
scale = 0.001/2625.51*pow(18.897,6)
for i in range(n_atoms):
    if(Charge_nucleus[i]=='1.0'): C6_atom.append(0.14*scale)
    if(Charge_nucleus[i]=='6.0'): C6_atom.append(1.75*scale)
    if(Charge_nucleus[i]=='7.0'): C6_atom.append(1.23*scale)
    if(Charge_nucleus[i]=='8.0'): C6_atom.append(0.70*scale)
    if(Charge_nucleus[i]=='9.0'): C6_atom.append(0.75*scale)
    if(Charge_nucleus[i]=='16.0'): C6_atom.append(5.57*scale)
    if(Charge_nucleus[i]=='17.0'): C6_atom.append(5.07*scale)
    if(Charge_nucleus[i]=='35.0'): C6_atom.append(12.47*scale)

# Computing asymptotic interaction energies according to:
#    2IQA_inf[i,j] = Qi*Qj/Rij - DIij/(2*Rij) - C6ij/Rij^6.
# C6ij is chosen as the geometry mean of the atomic C6.
# IQA_inf_woDisp_matrix does not incude dispersion and is
#    used to calculate the asymptotic error.
# For the Coulomb matrix: Coulomb[i,j]=-Zi*Zj/Rij
# The IQA diagonal elements are the self energies.
# The Coulomb diagonal elements are -0.5*Z^2.4.
Edisp = 0.0
Asympt_error_matrix = 0.0
for i in range(n_atoms):
    Coulomb_matrix[i,i] = -0.5*pow(float(Charge_nucleus[i]),2.4)
    IQA_matrix[i,i] = Eiqa_self[i]
    IQA_inf_matrix[i,i] = Eiqa_self[i]
    for j in range(n_atoms):
        if (j!=i):
            IQA_matrix[i,j] = 0.5*Eint_matrix[i,j]
            Rij = math.sqrt(pow(X_nucleus[i]-X_nucleus[j],2) + \
                            pow(Y_nucleus[i]-Y_nucleus[j],2) + \
                            pow(Z_nucleus[i]-Z_nucleus[j],2))
            C6ij = math.sqrt(C6_atom[i]*C6_atom[j])
            Coulomb_matrix[i,j] = -float(Charge_nucleus[i])* \
                                  float(Charge_nucleus[j])/Rij
            IQA_inf_woDisp_matrix[i,j] = 0.5*( \
                                  Charge_atom[i]*Charge_atom[j]/Rij - \
                                  Deloc_matrix[i,j]/(2.0*Rij))
            IQA_inf_matrix[i,j] = IQA_inf_woDisp_matrix[i,j] - \
                                  0.5*C6ij/pow(Rij,6)
            Edisp += -0.5*C6ij/pow(Rij,6)
Asympt_error_matrix = abs(IQA_inf_matrix-IQA_matrix)
Asympt_error_matrix_value = np.sum(Asympt_error_matrix)/(2*n_pairs)

##########################################################
### Computing energy vectors #############################
##########################################################

Eiqa_self.sort()
Eiqa_add.sort()
Coulomb_vector = linalg.eigvalsh(Coulomb_matrix)
Coulomb_vector.sort()
IQA_vector = linalg.eigvalsh(IQA_matrix)
IQA_vector.sort()
IQA_inf_vector = linalg.eigvalsh(IQA_inf_matrix)
IQA_inf_vector.sort()
Asympt_error_eig1 = abs(IQA_vector-IQA_inf_vector)
Asympt_error_eig = sum(Asympt_error_eig1)/n_atoms

##########################################################
### Displaying results ###################################
##########################################################

get_ipython().magic('clear')
float_formatter = lambda x: "%.6f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
print()
print(); print()
print('_______________________________________________________________')
print('****************** All values in atomic units *****************')
print(); print('File:',file_name); print()
print('Gaussian molecular energy = ',round(Emol_Gaussian,6))
print('IQA molecular energy      = ',round(Emol_IQA,6))
print('Reconstruction error      = ',round(Reconst_error,6))
if(Reconst_error>0.001): print('    => Significant reconstruction error')
if(Reconst_error<=0.001): print('    => Accurate energy reconstruction')
print('Grimme dispersion energy  = ',round(Edisp,6))
print(); print('Mean asymptotic error on IQA matrix      = ', \
               round(Asympt_error_matrix_value,6))
print('Mean asymptotic error on IQA eigenvalues = ', \
               round(Asympt_error_eig,6))
print(); print('    Sorted Coulomb vector:')
print(Coulomb_vector); print()
print('    Sorted self IQA vector:')
print(np.round(Eiqa_self,6)); print()
print('    Sorted additive IQA vector:')
print(np.round(Eiqa_add,6)); print()
print('    Sorted IQA eigenvector:')
print(np.round(IQA_vector,6)); print()
print('    Sorted asympt. IQA eigenvector:')
print(np.round(IQA_inf_vector,6)); print()
print('_______________________________________________________________')
print(); print()

##########################################################
### Completing vectors with 0 values #####################
##########################################################

for i in range(n_max-n_atoms):
    Coulomb_vector=np.append(Coulomb_vector,0.0)
    Eiqa_self=np.append(Eiqa_self,0.0)
    Eiqa_add=np.append(Eiqa_add,0.0)
    IQA_vector=np.append(IQA_vector,0.0)
    IQA_inf_vector=np.append(IQA_inf_vector,0.0)
