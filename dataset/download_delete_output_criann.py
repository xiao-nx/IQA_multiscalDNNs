#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:45:20 2021

@author: ljia
"""
import os
import sys
from tqdm import tqdm


def run_command(command, output=True):
	if output:
		stream = os.popen(command)
		return stream.readlines()
	else:
		os.popen(command)
		return 0


def get_paramiko_tran(server_name, username):
	import paramiko
	tran = paramiko.Transport(server_name)
	private = paramiko.RSAKey.from_private_key_file(os.path.expanduser('~/.ssh/id_rsa'))
	tran.connect(username=username, pkey=private)
	sftp = paramiko.SFTPClient.from_transport(tran)
	return tran, sftp
#	client = paramiko.SSHClient()
#	client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#	client.connect(hostname='myria.criann.fr', username='ljia', pkey=private)



def get_nb_files(path_remote, ext=None):
	if ext is None:
		command = 'ssh ljia03@myria.criann.fr "find \\"' + path_remote + '\\" -type f | wc -l"'
	else:
		command = 'ssh ljia03@myria.criann.fr "find \\"' + path_remote + '\\" -type f -name \\"*.' + ext + '\\" | wc -l"'

	output = run_command(command)
	nb_of_files = output[0].strip()
	return nb_of_files


def save_paths_to_log(fn_log, path_remote, ext=None):
	if os.path.isfile(fn_log):
		print('Using already existing log file.')
		return True

	if ext is None:
		command = 'ssh ljia03@myria.criann.fr "find \\"' + path_remote + '\\" -type f" > "' + fn_log + '"'
	else:
		command = 'ssh ljia03@myria.criann.fr "find \\"' + path_remote + '\\" -type f -name \\"*.' + ext + '\\"" > "' + fn_log + '"'

	output = run_command(command)
	return output


def download_files(fn_log, path_local, path_remote, batch_size=100, delete_otf=False):
	save_interval = 5000

	# backup log file.
	fn_log_bp = fn_log + '.backup'
	if not os.path.isfile(fn_log_bp):
		from shutil import copyfile
		copyfile(fn_log, fn_log_bp)
		log_bp_exist = False
	else:
		log_bp_exist = True
		print('Using already existing log backup file.')


	with open(fn_log_bp, 'r') as f_log_bp:
		lines_log = f_log_bp.readlines()


#	### Check log backup file, remove possible redundant lines due to the on-the-fly deletion.
#	if log_bp_exist:
#		for i in tqdm(range(0, save_interval), desc='Checking redundant lines in log backup file', file=sys.stdout):
#			line_c = lines_log[i].strip().replace(path_remote, '').strip('/')
#			path_local_c = os.path.join(path_local, line_c)
#			if not os.path.exists(path_local_c):
#				break

#		with open(fn_log_bp, 'w') as f_log_bp:
#			f_log_bp.write(''.join(lines_log[i:]))


	####################################################################
	### Create connection.
	tran, sftp = get_paramiko_tran('myria.criann.fr', 'ljia03')

	# for each line
	for idx, line in tqdm(enumerate(lines_log[::]), desc='Downloading', file=sys.stdout):
		line_c = line.strip().replace(path_remote, '').strip('/')
		path_local_c = os.path.join(path_local, line_c)
		sub_path = '/'.join(line_c.split('/')[:-1])
		os.makedirs(os.path.join(path_local, sub_path), exist_ok=True)


#		sftp.get(line.strip(), path_local_c)

#		### if delete on the fly.
#		if delete_otf:
#			# if the file is successfully downloaded.
#			if os.path.exists(path_local_c):
#				# Delete the file on remote server.
#				sftp.remove(line.strip())

		try:
			sftp.get(line.strip(), path_local_c)
		except FileNotFoundError:
			import warnings
			warnings.warn('File ' + line.strip() + ' does not exist. Please check.')
		else:

			### if delete on the fly.
			if delete_otf:
				# if the file is successfully downloaded.
				if os.path.exists(path_local_c):
					# Delete the file on remote server.
					sftp.remove(line.strip())


		### Update backup log file every 5000 iterations to support breakpoint resumption.
		if (idx + 1) % save_interval == 0:
			with open(fn_log_bp, 'w') as f_log_bp:
				f_log_bp.write(''.join(lines_log[idx + 1:]))


	### Close connection.
	sftp.close()
	tran.close()


#	#########################################################
#	### Create connection.
#	tran, sftp = get_paramiko_tran('myria.criann.fr', 'ljia03')

#	# Compute the interval to update backup file.
#	intval_up = int(batch_size * int(5000 / batch_size))

#	for idx, _ in tqdm(enumerate(lines_log[:100:batch_size]), desc='Downloading', file=sys.stdout):

#		for idx_in in range(idx, idx + batch_size):
#			if idx_in >= len(lines_log):
#				break

#			line = lines_log[idx_in]
#			line_c = line.strip().replace(path_remote, '').strip('/')
#			sub_path = '/'.join(line_c.split('/'))
#			path_local_c = os.path.join(path_local, sub_path)


#			sftp.get(line.strip(), path_local_c)

#		if (idx + batch_size) % intval_up == 0:
#			with open(fn_log_bp, 'w') as f_log_bp:
#				f_log_bp.write(''.join(lines_log[idx + batch_size:]))


#	### Close connection.
#	sftp.close()
#	tran.close()


#	####################################################################
#	# Compute the interval to update backup file.
#	intval_up = int(batch_size * int(5000 / batch_size))

#	for idx, _ in tqdm(enumerate(lines_log[:100:batch_size]), desc='Downloading', file=sys.stdout):
#		command = ''

#		for idx_in in range(idx, idx + batch_size):
#			if idx_in >= len(lines_log):
#				break

#			line = lines_log[idx_in]
#			line_c = line.strip().replace(path_remote, '').strip('/')
#			sub_path = '/'.join(line_c.split('/')[:-1])
# #		if not line.strip().endswith('/'):
#			path_local_c = os.path.join(path_local, sub_path)
#			os.makedirs(path_local_c, exist_ok=True)

#			command += 'scp ljia03@myria.criann.fr:"' + line.strip() + '" "' + path_local_c + '"\n'
# #			command += 'rsync -avP ljia03@myria.criann.fr:"' + line.strip() + '" "' + path_local_c + '"\n'

#		output = run_command(command, output=False)

#		if (idx + batch_size) % intval_up == 0:
#			with open(fn_log_bp, 'w') as f_log_bp:
#				f_log_bp.write(''.join(lines_log[idx + batch_size:]))


#	####################################################################
#	for idx, line in tqdm(enumerate(lines_log[::]), desc='Downloading', file=sys.stdout):
#		line_c = line.strip().replace(path_remote, '').strip('/')
#		sub_path = '/'.join(line_c.split('/')[:-1])
# #		if not line.strip().endswith('/'):
#		path_local_c = os.path.join(path_local, sub_path)
#		os.makedirs(path_local_c, exist_ok=True)

#		command = 'scp -r ljia03@myria.criann.fr:"' + line.strip() + '" "' + path_local_c + '"'

#		output = run_command(command)

#		if (idx + 1) % 5000 == 0:
#			with open(fn_log_bp, 'w') as f_log_bp:
#				f_log_bp.write(''.join(lines_log[idx + 1:]))


def check_download_completeness(fn_log, path_local, path_remote):
	files_missing = []
	with open(fn_log, 'r') as f_log:
		for idx, line in tqdm(enumerate(f_log), desc='Checking', file=sys.stdout):
			line_c = line.strip().replace(path_remote, '').strip('/')
			if not os.path.exists(os.path.join(path_local, line_c)):
				files_missing.append(line)

	if files_missing != []:
		print('Files missing:')
		print(files_missing)
		return False
	else:
		return True


def delete_contents_on_server(path_remote, ext=None):
	if ext is None:
		# @todo: use find to support path with blankspaces.
		command = 'ssh ljia03@myria.criann.fr "rm -r \'' + path_remote + '/\'"'
	else:
		command = 'ssh ljia03@myria.criann.fr "find \'' + path_remote + '/\' -type f -name \'*.' + ext + '\' -exec rm -f {} \\;"'
#		command = 'ssh ljia03@myria.criann.fr "rm -r \\"' + path_remote + '/*.' + ext + '\\""'

	output = run_command(command)
	return output


def download_delete_output_from_criann(dataset, method_s, root_dir='wfx_tmp', ext=None, delete_server=True, delete_otf=False):
	path_remote = '/home/2021007/ljia03/octupussy/outputs/' + dataset + '/' + method_s + '/' + root_dir + '/'
	path_local = '../outputs/' + dataset + '/' + method_s + '/' + root_dir + '/'
	os.makedirs(path_local, exist_ok=True)


	### Check files.
	print('Checking all files...')
	nb_of_files = get_nb_files(path_remote, ext=ext)
	print('There are ' + nb_of_files + ' files.')


	### Save paths to log.
	print('Saving all paths to log...')
	fn_log = '.dwn.' + dataset.replace('/', '_') + '.' + method_s + ('' if ext is None else ('.' + ext)) + '.log'
	save_paths_to_log(fn_log, path_remote, ext=ext)


	### Download files from CRIANN...
	print('Downloading files from CRIANN...')
	download_files(fn_log, path_local, path_remote, delete_otf=delete_otf)


	### Check if everything is downloaded.
	print('Checking if every file is downloaded...')
	is_dwn_complt = check_download_completeness(fn_log, path_local, path_remote)
	if not is_dwn_complt:
		import warnings
		warnings.warn('Not all files are downloaded! Program abort. Please check for the problems.')
		return


	### Delete all contents on the remote server. # @todo: change it  back.
#	if delete_server:
#		delete_contents_on_server(path_remote, ext=ext)


	### Remove log files.
	os.remove(fn_log)
	os.remove(fn_log + '.backup')


if __name__ == '__main__':

	# Default values.
	ext = None

	#%%

#	### Test.
#	dataset = 'QM7'
#	method_s = 'svwn'
#	root_dir = 'test'


	#%%


# 	### Exp 1.2, 1.5
# 	dataset = 'QM7'
# 	method_s = 'svwn'
# 	root_dir = 'wfx_tmp' # 'gaussian'
# 	ext = None


#	### Exp 1.8
#	dataset = 'QM7'
#	method_s = 'pbeqidh'
#	root_dir = 'gaussian'
#	ext = None


#	### Exp 1.6
#	dataset = 'QM7'
#	method_s = 'svwn.6-31Gd'
#	root_dir = 'wfx_tmp'
#	ext = 'sum'

	#-----------------------------------------

	### Exp 2.2, 2.5
	#dataset = 'QMrxn20/reactant-conformers'
	dataset = 'QMrxn20/transition-states/e2'
	#dataset = 'QMrxn20/transition-states/sn2'
	method_s = 'svwn'
	root_dir = 'wfx_tmp'
	ext = None

	#-----------------------------------------

	### Exp 3
	#dataset="Diatomic"
	#method_s="svwn sto-3g"

	download_delete_output_from_criann(dataset, method_s, root_dir=root_dir, ext=ext, delete_server=True, delete_otf=True) # @todo: change delete_otf