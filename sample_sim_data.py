import json
import os
import pickle
import time
from datetime import datetime

import numpy as np
import tqdm
from scipy import stats as spst
from scipy.spatial.distance import mahalanobis
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

# Set current directory to base of the git repo
CUR_DIR = './'
os.chdir(CUR_DIR)

import kinetic_model as ptk
import helper_func as hlp

data_dir = os.path.join(CUR_DIR, 'sim_data')

## Define dt and time vector

# PET Acquisition time frame windows (in minutes)
acquisition_time_frames = 1 / 60 * np.array([[0., 10.],
                                             [10., 20.],
                                             [20., 30.],
                                             [30., 40.],
                                             [40., 50.],
                                             [50., 60.],
                                             [60., 75.],
                                             [75., 90.],
                                             [90., 105.],
                                             [105., 120.],
                                             [120., 135.],
                                             [135., 150.],
                                             [150., 165.],
                                             [165., 180.],
                                             [180., 210.],
                                             [210., 240.],
                                             [240., 270.],
                                             [270., 300.],
                                             [300., 330.],
                                             [330., 360.],
                                             [360., 420.],
                                             [420., 480.],
                                             [480., 540.],
                                             [540., 600.],
                                             [600., 660.],
                                             [660., 720.],
                                             [720., 780.],
                                             [780., 840.],
                                             [840., 960.],
                                             [960., 1080.],
                                             [1080., 1200.],
                                             [1200., 1320.],
                                             [1320., 1440.],
                                             [1440., 1560.],
                                             [1560., 1680.],
                                             [1680., 1800.],
                                             [1800., 2100.],
                                             [2100., 2400.],
                                             [2400., 2700.],
                                             [2700., 3000.],
                                             [3000., 3300.],
                                             [3300., 3600.],
                                             [3600., 3900.],
                                             [3900., 4200.],
                                             [4200., 4500.],
                                             [4500., 4800.],
                                             [4800., 5100.],
                                             [5100., 5400.],
                                             [5400., 5700.],
                                             [5700., 6000.],
                                             [6000., 6300.],
                                             [6300., 6600.],
                                             [6600., 6900.],
                                             [6900., 7200.]])

time_vector = acquisition_time_frames[:, 1]
dt = acquisition_time_frames[:, 1] - acquisition_time_frames[:, 0]

## Parameters

n_samples = 100
n_ROI = 48
flag_testing_data = True
mean_sigma_noise_save = 1e-1  # mean noise level for simulated TACs
alpha = 0.8  # rejection criterion for testing data
flag_mahalanobis = flag_testing_data  # Use mahalanobis distance for rejection criterion
MK_half_T = 109.8

str_test = '_test' if flag_testing_data else '_train'

save_samples_dir = os.path.join(data_dir, 'nROI{}'.format(n_ROI))
if not os.path.isdir(save_samples_dir):
	os.makedirs(save_samples_dir)

## Defining Explicit prior from scans

# load file for prior parameters
stats_dict = pickle.load(open(os.path.join(CUR_DIR, 'prior_stats_nROI{}.pik'.format(n_ROI)), 'rb'))

target_ROI_names = stats_dict['ROI_names']

# simulating DVR and R1 mean and covariance
mu_DVR = stats_dict['mu_DVR']
Cov_DVR = stats_dict['Cov_DVR']
Cov_DVR_inv = np.linalg.inv(Cov_DVR)

mu_R1 = stats_dict['mu_R1']
Cov_R1 = stats_dict['Cov_R1']
Cov_R1_inv = np.linalg.inv(Cov_R1)

mu_k2p = stats_dict['mu_k2p']

# Mean and Cov from reference TAC values
mu_tac_ref = stats_dict['mu_tac_ref']
Cov_tac_ref = stats_dict['Cov_tac_ref']
Cov_tac_ref_inv = np.linalg.inv(Cov_tac_ref)

## Sampling from explicit prior

if flag_testing_data:
	if flag_mahalanobis:
		def cond_test(vec, mu, Cov_inv):
			d_square = mahalanobis(mu, vec, Cov_inv) ** 2
			cond = spst.chi2.cdf(d_square, mu_DVR.shape[0]) < alpha
			return cond
	else:
		cond_test = lambda vec, mu, Cov_inv: np.all(np.abs(vec - mu) / mu < alpha)
else:
	cond_test = lambda vec, mu, Cov_inv: True

# DVR
t = time.time()
varDVR = hlp.truncnormal_samples(mu_DVR, Cov_DVR, Cov_DVR_inv, n_samples, cond_test)
elapsed = time.time() - t
print('elapsed time: {:.2f} sec'.format(elapsed))

# R1
t = time.time()
varR1 = hlp.truncnormal_samples(mu_R1, Cov_R1, Cov_R1_inv, n_samples, cond_test)
elapsed = time.time() - t
print('elapsed time: {:.2f} sec'.format(elapsed))

# k2p is assumed fixed for the whole population scanned
t = time.time()
vark2p = [mu_k2p for _ in range(n_samples)]
elapsed = time.time() - t
print('elapsed time: {:.2f} sec'.format(elapsed))

t = time.time()
vartacref = hlp.truncnormal_samples(mu_tac_ref, Cov_tac_ref,
                                    Cov_tac_ref_inv, n_samples, cond_test)
elapsed = time.time() - t
print('elapsed time: {:.2f} sec'.format(elapsed))

cur_time_cp = datetime.now()
save_km_dir = os.path.join(save_samples_dir, '{}{}'.format(cur_time_cp.strftime("%y-%m-%d_%H-%M-%S"), str_test))

if not os.path.isdir(save_km_dir):
	os.makedirs(save_km_dir)

## Create new time activity curves from sampled kinetic params

tac_sampled = []
pp = 0
for n_i in tqdm.tqdm(range(n_samples)):
	xsingle = -1. * np.ones([n_ROI, len(dt)])
	# Failsafe: sometimes, with weird choices of DVR and R1, the produced TAC may become negative
	while np.sum(xsingle < 0) > 0:
		params_SRTM = {"k2p": vark2p[n_i], "R1": varR1[n_i], "DVR": varDVR[n_i]}
		# Create SRTM / MRTM object
		k_mrtm = ptk.SRTM2(frame_time_list=time_vector, frame_duration_list=dt, tac_reference=vartacref[n_i])
		xsingle = (k_mrtm.create_activity_curve(**params_SRTM) * dt[:, None]).T
		if np.sum(xsingle < 0) > 0:
			pp += 1
			varDVR[n_i] = hlp.truncnormal_samples(mu_DVR, Cov_DVR, Cov_DVR_inv, 1, cond_test)[0]
			varR1[n_i] = hlp.truncnormal_samples(mu_R1, Cov_R1, Cov_R1_inv, 1, cond_test)[0]
			vartacref[n_i] = hlp.truncnormal_samples(mu_tac_ref, Cov_tac_ref,
			                                         Cov_tac_ref_inv, 1, cond_test)[0]
	tac_sampled.append(xsingle)
print('{} samples produced a negative TAC and were resampled (DVR, R1, and TAC ref).'.format(pp))

## Simple noise model for TACs with std varying for each ROI
# (Alternatively, can use eq 12. from manuscript if in-vivo TAC available to estimate noise std in each ROI)

MK_lambda = np.log(2) / MK_half_T

mean_sigma_noise = mean_sigma_noise_save
str_noise = '_s{:.1e}'.format(mean_sigma_noise)
# vary std by 30% of the mean noise std for different values of std per ROI (example for git repo)
sigma_roi = hlp.trunc_normal(mean_sigma_noise, 0.3 * mean_sigma_noise, low=0, size=n_ROI)
sigma_noise = sigma_roi[:, None] / np.sqrt(dt[None, :] * np.exp(-MK_lambda * time_vector))

mu_noise = np.zeros_like(sigma_noise)

## Produce noisy samples

tac_noisy_sampled = []
for xsingle in tqdm.tqdm(tac_sampled):
	xsingle_noisy = xsingle / dt[None, :]
	for roi_i in range(n_ROI):
		# TAC cannot have negative values
		xsingle_noisy[roi_i,] += np.sqrt(xsingle_noisy[roi_i,]) * \
		                         hlp.trunc_normal(mean=mu_noise[roi_i,], std=sigma_noise[roi_i,],
		                                          low=-np.sqrt(xsingle_noisy[roi_i,]), upp=None)

	xsingle_noisy = xsingle_noisy * dt[None, :]
	tac_noisy_sampled.append(xsingle_noisy)

# Save pickle file with simulated data
pickle.dump({'varDVR':           varDVR, 'varR1': varR1, 'vark2p': vark2p, 'vartacref': vartacref,
             'tac_sampled':      tac_sampled, 'tac_noisy_sampled': tac_noisy_sampled,
             'mu_noise':         mu_noise, 'sigma_noise': sigma_noise,
             'mean_sigma_noise': mean_sigma_noise, 'flag_mahalanobis': flag_mahalanobis,
             'target_ROI_names': target_ROI_names, 'time_vector': time_vector, 'dt': dt,
             }, open(os.path.join(save_km_dir, 'data_nROI{}_n{}{}.pik'.
                                  format(n_ROI, n_samples, str_noise)), 'wb'))

# Save json readable file with simulated data
with open(os.path.join(
		save_km_dir, 'args_nROI{}_n{}{}.txt'.format(
			n_ROI, n_samples, str_noise)), 'wt') as file_pi:
	save_args_json = {
		'mean_sigma_noise': mean_sigma_noise,
		'target_ROI_names': target_ROI_names,
		'MK_half_T':        MK_half_T,
		'MK_lambda':        MK_lambda,
		'n_samples':        n_samples,
		'n_ROI':            n_ROI,
		'save_samples_dir': save_samples_dir,
		'flag_mahalanobis': flag_mahalanobis
	}
	json.dump(save_args_json, file_pi, indent=2, sort_keys=True)

# %% Quick plot of time-activity curves

sample_plot = 0
linewidth_plot = 2

n_ROI_list_show = np.arange(0, n_ROI, 2).tolist()

for xi, x in enumerate([time_vector, np.arange(len(time_vector))]):
	# plot with and without time vector along x to better show activity in early frames
	fig_all, ax_all = plt.subplots(int(np.ceil(np.sqrt(len(n_ROI_list_show)))),
	                               int(np.ceil(np.sqrt(len(n_ROI_list_show)))),
	                               figsize=(16, 13), sharex=True, sharey=True)

	for ni in range(len(n_ROI_list_show)):
		ax_all.flatten()[ni].plot(x, tac_noisy_sampled[sample_plot][n_ROI_list_show[ni],] / dt,
		                          label='noisy', linewidth=linewidth_plot, color='red')
		ax_all.flatten()[ni].plot(x, tac_sampled[sample_plot][n_ROI_list_show[ni],] / dt,
		                          label='no noise', linewidth=linewidth_plot, alpha=0.6, color='black')

	fig_all.suptitle('n_ROI= {} TACs for sample {}'.format(n_ROI, sample_plot), fontsize=12)

	ax_all.flatten()[0].legend()
	fig_all.tight_layout()

	for fmt_save in ['pdf', 'png']:
		fig_all.savefig(os.path.join(save_km_dir, 'TAC_plot_nROI{}_sample{}_{}.{}'.format(n_ROI, sample_plot,
		                                                                                  xi, fmt_save)), )
