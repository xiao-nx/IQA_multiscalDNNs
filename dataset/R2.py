#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 14:56:23 2022

@author: ljia
"""
import sys
import os
import pickle
sys.path.insert(1, '../')
import numpy as np


def compute_r2():

	### Load dataset.
	from dataset.to_nx import get_qm7_eigen_vectors

	ds_fname = 'x_for_R2.pkl'
	if os.path.isfile(ds_fname) and os.path.getsize(ds_fname) != 0:
		with open(ds_fname, 'rb') as f:
			X = pickle.load(f)
	else:
		ds_name = 'QM7'
		descriptor = 'Eiqa_self_add_sc'
	# 	dataset = get_dataset(ds_name, descriptor)
	# 	X = dataset['X']
		X = get_qm7_eigen_vectors(descriptor.lower(), remove_sig_err=True)
		with open(ds_fname, 'wb') as f:
			pickle.dump(X, f)

	size = int(X.shape[1] / 2)
	X_self = X[:, 0:size]
	X_add = X[:, size:]


	### Compute R2.
	from sklearn.metrics import r2_score
	r2_add_self = r2_score(X_add, X_self)
	print(r2_add_self)


	### Plot errors.
	err = np.empty((len(X_self), 1))
	for i in range(len(X_self)):
		err[i] = 0


def get_correlation(descriptor, reg_model='linear'):

	### Load dataset.
	from dataset.to_nx import get_dataset

	ds_fname = 'X_for_R2.' + descriptor + '.pkl'
	if os.path.isfile(ds_fname) and os.path.getsize(ds_fname) != 0:
		with open(ds_fname, 'rb') as f:
			dataset = pickle.load(f)
			X = dataset['X']
			targets = dataset['targets']

	else:
		ds_name = 'QM7'
		remove_sig_errs=True
		dataset = get_dataset(ds_name, descriptor, remove_sig_errs=remove_sig_errs)
		X = dataset['X']
		targets = dataset['targets']
		with open(ds_fname, 'wb') as f:
			pickle.dump(dataset, f)


	### Perform linear regression.
	if reg_model is None:
		targets_pred = X
	elif reg_model == 'negtive':
		targets_pred = -X
	elif reg_model == 'linear':
		from sklearn import linear_model
		clf = linear_model.LinearRegression()
		clf.fit(X, targets)
		targets_pred = clf.predict(X)
		print('coefs:')
		print(clf.coef_)
		print('intercept = %f' % clf.intercept_)
		print('score = %f' % clf.score(X, targets))

# 	coef = np.polyfit(X, targets, 1)
# 	poly1d_fn = np.poly1d(coef)
# 	targets_pred = poly1d_fn(X)


	### Compute RMSE, MAE, and R2.
	from sklearn.metrics import mean_squared_error
	rmse = mean_squared_error(targets_pred, targets, squared=False)
	print('rmse = ' + str(rmse))

	from sklearn.metrics import mean_absolute_error
	mae = mean_absolute_error(targets_pred, targets)
	print('mae = ' + str(mae))


	from sklearn.metrics import r2_score
	r2 = r2_score(targets_pred, targets)
	print('r2 = %f' % r2)


	# Plot correlation.
	import matplotlib
	matplotlib.use('TkAgg')
	import matplotlib.pyplot as plt
# 	plt.plot(X, targets_pred, '+', X, targets_pred, '--', markersize=0.1, linewidth=0.1)
# 	plt.xlabel(descriptor)
# 	plt.ylabel('Emol_ref')
# 	plt.title(descriptor + ' vs. Emol_ref')
	plt.plot(targets, targets, '--', targets, targets_pred, 'x', markersize=0.2, linewidth=0.2)
	plt.xlabel('Emol_ref')
	plt.ylabel('Emol_predicted')
	plt.title(descriptor)
	fn_fig_pref = '../figures/correlation_linear_reg.' + descriptor
	plt.savefig(fn_fig_pref + '.eps', format='eps', dpi=300, transparent=False, bbox_inches='tight')
	plt.savefig(fn_fig_pref + '.pdf', format='pdf', dpi=300, transparent=False, bbox_inches='tight')
	plt.savefig(fn_fig_pref + '.png', format='png', dpi=300, transparent=False, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close()


if __name__ == '__main__':
	###
# 	compute_r2()

	### exp 2.
	for descriptor in ['KENEEE_Atom_Sum_NN',
					'KENE_Atom_Sum_NE', 'NE_Atom_Sum_KNE', 'KE_Atom_Sum_NNE',
					'Emol_Gaussian', 'additive_KE', 'additive_NN',
					'additive_NE', 'additive_EE', 'additive_KNNE']:
		print('--------- %s ---------' % descriptor)
		get_correlation(descriptor)
