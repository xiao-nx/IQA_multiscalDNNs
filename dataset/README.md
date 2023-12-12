# QM7TOGJF

This program transfers the QM7 dataset in MATLAB format into .gjf file using Python.


QM7 is a subset of GDB-13 (a database of nearly 1 billion stable and synthetically accessible organic molecules) containing up to 7 heavy atoms C, N, O, and S. The 3D Cartesian coordinates of the most stable conformations and their atomization energies were determined using ab-initio density functional theory (PBE0/tier2 basis set). This dataset also provided Coulomb matrices as calculated in [Rupp et al. PRL, 2012]:
Stratified splitting is recommended for this dataset.
The data file (.mat format, we recommend using `scipy.io.loadmat` for python users to load this original data) contains five arrays:
- "X" - (7165 x 23 x 23), Coulomb matrices
- "T" - (7165), atomization energies (unit: kcal/mol)
- "P" - (5 x 1433), cross-validation splits as used in [Montavon et al. NIPS, 2012]
- "Z" - (7165 x 23), atomic charges
- "R" - (7165 x 23 x 3), cartesian coordinate (unit: Bohr) of each atom in the molecules.


## Requirements.
- python >= 3.6.9
- scipy
- tdqm

## How to use

```
python mat2gif.py [PATH_TO_DATASET]
```

## Notice

* The cartesian coordinate of the atoms in QM7 is in Bohr.
* In each molecule, all properties have 23 values corresponding to 23 atoms, where the shortage is filled with 0. 
* The program uses atomic charges ("Z") to determine the atom symbols and the number of atoms in a molecule.
* In [DeepChem](https://github.com/deepchem/deepchem) library, QM7 is loaded by [_QM7Loader](https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/qm7_datasets.py), where the comments says:
```
DeepChem 2.4.0 has turned on sanitization for this dataset by default.  For the QM7 dataset, this means that calling this function will return 6838 compounds instead of 7160 in the source dataset file.  This appears to be due to valence specification mismatches in the dataset that weren't caught in earlier more lax versions of RDKit. Note that this may subtly affect benchmarking results on this dataset.
```
This seems like a coding problem rather than a problem with the dataset itself, so in our program it is not considered.
