import sys
import glob
import os
import pickle
import socket
import time

import matplotlib
import numpy as np
import pymc as pm
import pytensor as ptt
from pytensor.graph.op import Op
from scipy.stats import norm

matplotlib.use('Agg')
# matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

import kinetic_model as ptk

NP_DTYPE = np.float64
FLAG_PLOT = False

# %% class used for kinetic modelling forward model for compatibility with pytensor and pymc fraemwork

class CreateTAC_SRTM2(Op):
	# Properties attribute
	__props__ = ()

	itypes = [ptt.tensor.dvector, ptt.tensor.dvector, ptt.tensor.dscalar]
	otypes = [ptt.tensor.dmatrix]

	def __init__(self, k_srtm):
		self.k_srtm = k_srtm

	# Python implementation:
	def perform(self, node, inputs, outputs, **kwargs):
		outputs[0][0] = self.k_srtm.create_activity_curve(DVR=inputs[0], R1=inputs[1], k2p=inputs[2]).T


## Get paths (gpu-node008 if on server, Dropbox else)

# Set current directory to base of the git repo
CUR_DIR = './'
os.chdir(CUR_DIR)

root_data_dir = CUR_DIR
data_dir = os.path.join(root_data_dir, 'sim_data')

##

n_ROI_test = 48
n_samples_test = 100
mean_sigma_noise_load = 1e-1
iter_mcmc = 200
burn_mcmc = 400
chains = 4
str_noise = '_s{:.1e}'.format(mean_sigma_noise_load)

# Set dir for testing set
load_samples_roi_dir = os.path.join(data_dir, 'nROI{}'.format(n_ROI_test))
load_km_dir_pattern = os.path.join(load_samples_roi_dir,
                                   '*_test', 'data_nROI{}_n{}{}.pik'.
                                   format(n_ROI_test, n_samples_test, str_noise))

load_km_dir = sorted(glob.glob(load_km_dir_pattern))[-1]
load_km_fname = load_km_dir.split(os.sep)[-1]
load_km_dir = load_km_dir.replace(os.sep + load_km_fname, '')

load_test_dict = pickle.load(open(os.path.join(load_km_dir, load_km_fname), 'rb'))

time_vector = np.array(load_test_dict['time_vector'], dtype=NP_DTYPE)
dt = np.array(load_test_dict['dt'], dtype=NP_DTYPE)

DVR_load = np.array(load_test_dict['varDVR'], dtype=NP_DTYPE)
R1_load = np.array(load_test_dict['varR1'], dtype=NP_DTYPE)
k2p_load = np.array(load_test_dict['vark2p'], ndmin=2, dtype=NP_DTYPE).T
tac_load = np.array(load_test_dict['tac_noisy_sampled'], dtype=NP_DTYPE
                    ) / dt[None, None, :]
tac_noiseless_load = np.array(load_test_dict['tac_sampled'], dtype=NP_DTYPE
                              ) / dt[None, None, :]

stats_dict = pickle.load(open(os.path.join(CUR_DIR, 'prior_stats_nROI{}.pik'.
                                           format(n_ROI_test)), 'rb'))

mu_DVR = np.array(stats_dict['mu_DVR'], dtype=NP_DTYPE)
Cov_DVR = np.array(stats_dict['Cov_DVR'], dtype=NP_DTYPE)

mu_R1 = np.array(stats_dict['mu_R1'], dtype=NP_DTYPE)
Cov_R1 = np.array(stats_dict['Cov_R1'], dtype=NP_DTYPE)

mu_k2p = stats_dict['mu_k2p'].astype(NP_DTYPE)

mu_noise = np.array(load_test_dict['mu_noise'], dtype=NP_DTYPE)
sigma_noise = np.array(load_test_dict['sigma_noise'], dtype=NP_DTYPE)
mean_sigma_noise = load_test_dict['mean_sigma_noise']

##

# Parameters order and names
km_param_str = ['DVR', 'R1', ]
# Check for ROI #No ...
for sample_plot in range(0, 10):

	km_obs = {'DVR': DVR_load[sample_plot],
	          'R1':  R1_load[sample_plot],
	          'k2p': k2p_load[sample_plot]}
	y_obs = tac_load[sample_plot].reshape([n_ROI_test, -1])

	# Check y_obs and km_obs are 3D
	y_obs = y_obs[None,] if y_obs.ndim < 2 else y_obs

	## Check if MCMC file exist

	mcmc_roi_dir = os.path.join(load_km_dir, 'MCMC{}'.format(str_noise))
	if not os.path.isdir(mcmc_roi_dir):
		os.makedirs(mcmc_roi_dir)
	save_dir_filename = os.path.join(
		mcmc_roi_dir, 'MH_MCMC_nROI{}_it{:.1e}_brn{:.1e}_km_obs-{:.3f}-{:.3f}-{:.3f}.pik'.format(
			n_ROI_test, iter_mcmc, burn_mcmc,
			km_obs['DVR'][0], km_obs['R1'][0], km_obs['k2p'][0]
		))

	if os.path.isfile(save_dir_filename):
		print('MCMC File already exists with these parameters '
		      '(sample {})... Skipping.'.format(sample_plot))
		continue

	## MCMC test

	# Create SRTM / MRTRM object
	k_srtm = ptk.SRTM2(frame_time_list=time_vector, frame_duration_list=dt,
	                   tac_reference=load_test_dict['vartacref'][sample_plot])

	Cov_dt = np.stack([np.diag(sigma_noise[roi_i] ** 2) for roi_i in range(n_ROI_test)])
	Cov_dt_inv = np.stack([np.diag(1 / sigma_noise[roi_i] ** 2) for roi_i in range(n_ROI_test)])

	# Define operators for MCMC
	Op_create_tac = CreateTAC_SRTM2(k_srtm=k_srtm)

	## Define the model and run sampling

	tic = time.time()
	MCMC_model = pm.Model()

	with MCMC_model:
		varDVR = pm.MvNormal("var_DVR", mu=mu_DVR, cov=Cov_DVR)
		varR1 = pm.MvNormal("var_R1", mu=mu_R1, cov=Cov_R1)
		vark2p = pm.Deterministic('var_k2p', ptt.tensor.as_tensor_variable(km_obs['k2p'][0]))
		sn = Op_create_tac(varDVR, varR1, vark2p)
		sn = ptt.tensor.switch(sn < 0, 1e-6, sn)
		scaled_sigma_noise = np.sqrt(sn) * sigma_noise
		varsn = pm.TruncatedNormal(name="var_sn", mu=sn, sigma=scaled_sigma_noise,
		                           lower=0, observed=y_obs)
		step = pm.Metropolis(proposal_dist=pm.NormalProposal)
		idata = pm.sample(draws=iter_mcmc, tune=burn_mcmc, step=step)

	elapsed_time = time.time() - tic
	print('elapsed time: {:.1f} sec'.format(elapsed_time))

	DVR_mcmc = np.asarray(idata.posterior['var_DVR'])
	R1_mcmc = np.asarray(idata.posterior['var_R1'])
	k2p_mcmc = np.asarray(idata.posterior['var_k2p'])

	save_mcmc_dic = {
		'idata':        idata,
		'DVR_mcmc':     DVR_mcmc,
		'k2p_mcmc':     k2p_mcmc,
		'R1_mcmc':      R1_mcmc,
		'iter':         iter_mcmc,
		'burn':         burn_mcmc,
		'y_obs':        y_obs,
		'km_obs':       km_obs,
		'chains':       chains,
		'elapsed_time': elapsed_time,
	}

	pickle.dump(save_mcmc_dic, open(save_dir_filename, 'wb'))
	with open(save_dir_filename.replace('.pik', '_summary.csv'), 'w') as f:
		f.write(pm.summary(idata).to_csv())

	# R_hat is a convergence indicator and should close to 1.
	# Saving in text file when MCMC results have R_hat values > 1.02
	# (i.e., potentially not fully convergded)
	rhat_DVR = np.array(pm.rhat(idata)['var_DVR'])
	rhat_R1 = np.array(pm.rhat(idata)['var_R1'])

	if np.any(rhat_DVR > 1.02) or np.any(rhat_R1 > 1.02):
		with open(os.path.join(mcmc_roi_dir,
		                       'rhat_less_than_102.txt'), 'at') as file_txt:
			file_txt.write(save_dir_filename.split(os.sep)[-1] +
			               ' - sample {} - rhat_max = {:.4f}\n'.format(
				               sample_plot, np.max([rhat_DVR.max(), rhat_R1.max()])))

	## Plot figures

	if FLAG_PLOT:
		# Separate windows for plots or subplot in one window
		n_window_plot = 3
		# Check for ROI #No ...
		roi_plot = 0

		fig, ax = [None, None, None], [None, None, None]

		if n_window_plot == 1:
			fig, ax = plt.subplots(1, 3, figsize=(14, 5))
		else:
			fig[0], ax[0] = plt.subplots(1, 1, figsize=(12, 5))
			fig[1], ax[1] = plt.subplots(1, 1, figsize=(12, 5))

		for pp, km_param_str_i in enumerate(km_param_str):
			if 'k2p' not in km_param_str_i:
				var_all = eval(km_param_str_i + '_mcmc')
				var = eval(km_param_str_i + '_mcmc[:,:,{}].flatten()'.format(roi_plot))
				mu_tmp = eval('mu_' + km_param_str_i + '[roi_plot]')
				km_tmp = km_obs[km_param_str_i][roi_plot]
			else:
				var = eval(km_param_str_i + '_mcmc.flatten()')
				mu_tmp = eval('mu_' + km_param_str_i)
				km_tmp = km_obs[km_param_str_i][0]
			ax[pp].hist(var, bins=100, color='red', label=['MCMC ' + km_param_str_i], density=True)
			x_axis = np.arange(ax[pp].axis()[0], ax[pp].axis()[1], 1e-4)
			ax[pp].plot(x_axis, norm.pdf(x_axis, np.mean(var), np.std(var)), linewidth=3.)
			ax[pp].axvline(x=np.mean(var), color='black',
			               linestyle='--', label='mean')
			ax[pp].set_title('prior: $\mu = {:.4f}$ ({}_observed = {:.4f}) \n'
			                 'MCMC: $\mu = {:.4f}$ - $\sigma = {:.4f}$'.format(mu_tmp,
			                                                                   km_param_str_i, km_tmp,
			                                                                   np.mean(var), np.std(var)))
			save_fig_filename = save_dir_filename.replace('.pik', '_{}_ROI{:d}.png'.format(km_param_str_i, roi_plot))
			fig[pp].savefig(save_fig_filename)
			plt.close(fig[pp])

			fig_all, ax_all = plt.subplots(int(np.ceil(np.sqrt(n_ROI_test))),
			                               int(np.ceil(np.sqrt(n_ROI_test))),
			                               figsize=(16, 10))
			ax_all = np.array(ax_all) if isinstance(ax_all, matplotlib.axes._axes.Axes) else ax_all

			for ni in range(n_ROI_test):
				ax_all.flatten()[ni].hist(var_all[:, :, ni].flatten(),
				                          density=True, label=format(ni), bins=100)
				x_axis = np.linspace(ax_all.flatten()[ni].axis()[0],
				                     ax_all.flatten()[ni].axis()[1], num=500)
				ax_all.flatten()[ni].plot(x_axis, norm.pdf(x_axis, np.mean(var_all[:, :, ni].flatten()),
				                                           np.std(var_all[:, :, ni].flatten())),
				                          linewidth=3., color='black')
				km_tmp = km_obs[km_param_str_i][ni]
				ax_all.flatten()[ni].set_title('({}_obs = {:.2f}) '
				                               '$\mu = {:.2f}$ - $\sigma = {:.3f}$'.format(km_param_str_i, km_tmp,
				                                                                           np.mean(
					                                                                           var_all[:, :, ni].flatten()),
				                                                                           np.std(var_all[:, :,
				                                                                                  ni].flatten())))
				ax_all.flatten()[ni].legend()
			save_fig_all_filename = save_dir_filename.replace('.pik', '_{}_ROI_all.png'.format(km_param_str_i))
			fig_all.savefig(save_fig_all_filename)
			plt.close(fig_all)
