
import sys
import gc

import tensorflow as tf
import tensorflow.keras as keras
import tqdm
import numpy as np
import os
import socket
import matplotlib
from labellines import labelLine, labelLines
import importlib
import pickle
import itertools
import json
from datetime import datetime
import glob
from scipy.stats import norm
import tensorflow_probability as tfp
import time

import matplotlib.pyplot as plt

try:
	import imageio.v3 as iio
except ModuleNotFoundError:
	iio = None

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

FLAG_XLA = True  # Deactivate XLA if tensorflow cannot find the cuda libraries
# Set cuda library directory for conda environment
# (change for the appropriate cuda lib  directory if XLA acceleration is wanted)
cuda_dir = os.path.join(os.environ.get('CONDA_PREFIX'), 'lib')

## Get paths (gpu-node008 if on server, Dropbox else)

if FLAG_XLA:
	os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=' + cuda_dir
	tf.config.optimizer.set_jit(True)  # Enable XLA.

# Set current directory to base of the git repo
CUR_DIR = './'
os.chdir(CUR_DIR)

import networks as cl   # cl: Custom Layer
import diffusion_model as cm  # cm: Custom Model
from diffusion_model import DTYPE, TF_DTYPE, NP_DTYPE
import helper_func as hlp

root_data_dir = CUR_DIR
data_dir = os.path.join(CUR_DIR, 'sim_data')
res_dir = os.path.join(root_data_dir, 'results')


## Load TACs

n_samples_orig = 100000
n_samples_test = 100
n_ROI = 48
mean_sigma_noise = 1e-1

km_param_str = ['DVR', 'R1',]
str_noise = '_s{:.1e}'.format(mean_sigma_noise)

load_samples_roi_dir = os.path.join(data_dir, 'nROI{}'.format(n_ROI))

# Set folders
load_km_dir_pattern = os.path.join(load_samples_roi_dir, '*_train', 'data_nROI{}_n{}{}.pik'.
								   format(n_ROI, n_samples_orig, str_noise))

# Load latest training/testing data created
load_km_dir = sorted(glob.glob(load_km_dir_pattern))[-1]
load_km_fname = load_km_dir.split(os.sep)[-1]
load_km_dir = load_km_dir.replace(os.sep + load_km_fname, '')

res_roi_dir = os.path.join(res_dir, 'nROI{}'.format(n_ROI), load_km_dir.split(os.sep)[-1])
if not os.path.isdir(res_roi_dir):
	os.makedirs(res_roi_dir)

# load traning data and set variables
load_dict = pickle.load(open(os.path.join(load_km_dir, load_km_fname), 'rb'))

# Set dir for testing set
load_km_dir_test_pattern = os.path.join(load_samples_roi_dir,
								   '*_test', 'data_nROI{}_n{}{}.pik'.
								   format(n_ROI, n_samples_test, str_noise))

load_km_dir_test = sorted(glob.glob(load_km_dir_test_pattern))[-1]

load_km_test_fname = load_km_dir_test.split(os.sep)[-1]
load_km_dir_test = load_km_dir_test.replace(os.sep + load_km_test_fname, '')

load_test_dict = pickle.load(open(os.path.join(load_km_dir_test, load_km_test_fname), 'rb'))

target_ROI_names = load_test_dict['target_ROI_names']

## Load variables to workspace

dt = (load_dict['dt']).astype(DTYPE)

dset = np.stack([load_dict['varDVR'], load_dict['varR1']], axis=-1).astype(DTYPE)
dset_test = np.stack([load_test_dict['varDVR'], load_test_dict['varR1']], axis=-1).astype(DTYPE)

label = np.concatenate([load_dict['tac_noisy_sampled'] / dt[None, None, :],
									   np.array(load_dict['vartacref'])[:, None, :]], axis=1).astype(DTYPE)
label_test = np.concatenate([load_test_dict['tac_noisy_sampled'] / dt[None, None, :],
									   np.array(load_test_dict['vartacref'])[:, None, :]], axis=1).astype(DTYPE)

# Setting an index for the position of DVR values and R1 values within dset to reciver the values easily
dict_idx = {'DVR': [np.arange(0, n_ROI).tolist(), [0,]], 'R1': [np.arange(0, n_ROI).tolist(), [1,]]}


del load_dict
gc.collect()

# Check number of sample and shuffle indices for training
n_samples = dset.shape[0]
rand_idx = np.arange(n_samples)
np.random.shuffle(rand_idx)
dset = dset[rand_idx,]
label = label[rand_idx,]

## Parameters for neural network

timesteps = 1000
learn_variance = 'learn_ranged'

# iDDPM
net_args = {
	'num_channels_out':    dset.shape[-1],
	'learn_variance':      learn_variance,
	'num_filt_start':      128,
	'pool_size':           2,
	'depth':               4,
	'block_type':          'conv',
	'block_params':        {'flag_res': True,
							# 'norm_type': 'layer',
							# 'l2_reg': 5e-6,
							# 'activation': 'relu',
							'kernel_size': 6
							},
	'cond_params':          {
		'flag_flatten_input': False,
		'network_name': 'encoder',
		'network_kwargs': {
			'enc_size': [256, 128, 64],
			'latent_dim': 32,
			# 'activation': 'leaky_relu',
			'final_activation': None,  # 'swish',
			# 'l2_reg': 1e-6,
		}
	},
	'skip_conn_type':      'concat',
	'skip_conn_op':        None,
	'skip_conn_post_op':   None,
	'dropout':             None,
	'final_activation':    None,
	'sin_emb_dim':         64,
	'normalize_feature_dict': None,
	'normalize_label_dict': None
}

batch_size = 256
lrn_rate_i = 2e-4
lrn_rate_f = 5e-5
epochs = 500
period = 50
clipnorm = 1.
validation_split = 0.1
loss = 'MeanSquaredError'

# DDPM/iDDPM
diff_args = {
	'timesteps': timesteps,
	'noise_schedule': {#'beta_start': 1e-4, # Default values in DDPM class
					   #'beta_end': 2e-2, # Default values in DDPM class
					   'schedule_name': 'cosine'},
	'lambda_vlb': 1e-1,
	'ndim': 1,
}

# Exp decreasing learning rate
decay_rate = (lrn_rate_f / lrn_rate_i) ** (1 / epochs)
decay_steps = n_samples // batch_size
lrn_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
	lrn_rate_i, decay_steps, decay_rate)

dir_arg = {
	'res_roi_dir': res_roi_dir,
	'load_km_dir': load_km_dir,
	'load_km_fname': load_km_fname,
	'load_km_dir_test': load_km_dir_test,
	'load_km_test_fname': load_km_test_fname,
}

misc_params = {
	'FLAG_XLA': FLAG_XLA,
	'DTYPE': DTYPE,
	'hostname': socket.gethostname(),
	'n_samples': n_samples,
	'batch_size': batch_size,
	'lrn_rate_i': lrn_rate_i,
	'lrn_rate_f': lrn_rate_f,
	'epochs': epochs,
	'period': period,
	'validation_split': validation_split,
	'clipnorm': clipnorm,
	'loss': loss,
	'str_noise': str_noise,
	'mean_sigma_noise': mean_sigma_noise,
}

## Build and compile model

importlib.reload(cl)
importlib.reload(cm)

network = cl.UnetConditional(**net_args)
diff_model = cm.ImprovedDDPM(network=network, **diff_args)

# initialize the model in the memory of our GPU and build (create Variables)
test_images = dset[:5, ...]
test_label = label[:5, ...]
test_timestamps = tf.random.uniform(shape=[5, ], minval=0, maxval=timesteps, dtype=tf.int32)
k = diff_model({'x': test_images, 'time': test_timestamps, 'condition': test_label})

opt = keras.optimizers.Adam(learning_rate=lrn_schedule, clipnorm=clipnorm)
diff_model.compile(optimizer=opt, loss=tf.keras.losses.deserialize(loss))


## Add compile args for saving and create model_args dict to save later
compile_args = {
	'opt_config': opt.get_config(),
	'loss': loss
}

model_args = {'net_args':       net_args.copy(),
			  'diff_args':      diff_args.copy(),
			  'compile_args':   compile_args.copy(),
			  'dir_arg':        dir_arg.copy(),
			  'misc_params':    misc_params.copy()
			  }


## Training

cur_time_cp = datetime.now()
save_model_name = '{}_{}_f{}_d{}_p{}{}'.format(
	cur_time_cp.strftime("%y-%m-%d_%H-%M"), diff_model.name,
	model_args['net_args']['num_filt_start'],
	model_args['net_args']['depth'],
	model_args['net_args']['pool_size'], str_noise)

save_model_dir = os.path.join(res_roi_dir, save_model_name)

# # Create input spec signature to pass for saving checkpoints
# specs = [
#     tf.TensorSpec((None,) + dset.shape[1:],   TF_DTYPE, name="x"),
#     tf.TensorSpec((None,),                    tf.int32,  name="time"),
#     tf.TensorSpec((None,) + label.shape[1:],  TF_DTYPE, name="condition"),
# ]
#
# @tf.function(input_signature=specs)
# def serve(x, time, condition):
# 	# Force `training=False` so there’s no stochasticity
# 	return diff_model(x, time, condition, training=False)
#
# concrete_fn = serve.get_concrete_function()  # binds the signature


# cp = cl.SavedModelCheckpoint(root_dir=save_model_dir, every_n_epochs=1, input_spec=specs,)
cp = cl.WeightsCheckpoint(root_dir=save_model_dir, every_n_epochs=period)

# Train
tic = time.time()
history = diff_model.fit(x=dset, y=label,
						 batch_size=batch_size,
						 epochs=epochs,
						 callbacks=[cp,],
						 validation_split=validation_split)

elapsed_time = time.time() - tic
print('elapsed time: {:.1f} sec'.format(elapsed_time))

model_args['elapsed_time'] = elapsed_time

history_dict = history.history

if 'history_dict' in locals():
	fig_cost = plt.figure()
	for (key, val) in history_dict.items():
		if np.min(val) < 0:
			plot_func = plt.plot
			break
		else:
			plot_func = plt.semilogy
	for (key, val) in history_dict.items():
		plot_func(val, label=key.replace('_', ' '))
	fig_cost.axes[0].legend()
	fig_cost.savefig(os.path.join(save_model_dir, 'cost.png'))
	plt.close(fig_cost)

##

if not os.path.isdir(save_model_dir):
	os.makedirs(save_model_dir)

diff_model.save(os.path.join(save_model_dir, 'saved_model.keras'))

with open(os.path.join(save_model_dir, 'trainHistoryDict.pik'), 'wb') as file_pi:
	pickle.dump(history_dict, file_pi)
with open(os.path.join(save_model_dir, 'model_args.pik'), 'wb') as file_pi:
	pickle.dump(model_args, file_pi)


with open(os.path.join(save_model_dir, 'model_args.txt'), 'wt') as file_pi:
	model_args_json = model_args
	json.dump(model_args_json, file_pi, indent=2, sort_keys=True)


## Params for plot

# number of posterior estimation
n_posterior = 10000
# histogram (hist) or seaborn plot (sns)
plot_type = 'hist'
# chunk size for diffusion processing
chunk_size = n_posterior
# ROI index for plot
roi_plot = 0
# Flag to keep DDPM output for all timestep  ({x_t}, t = [T,...,0])
keep_all_xt = False

# Load trained model and inference params
load_model = True
save_posterior_res = True  # save inference results in .pik file
flag_load_res = True  # load results instead of inference if .pik file available
timesteps_model = getattr(diff_model, 'timesteps', None)
make_png_period = [50]  #range(period, epochs + period, period)
sample_range = range(0, 2)

# Params for plots
hist_pred_color = 'royalblue'
hist_mcmc_color = 'indianred'
hist_linewidth_dist = .1
hist_linewidth_bar = .4
font_title = 19
font_label = 16
font_ticks = 11
font_legend = 12
ratio_mul_std_errorbar = 3
capsize = 3
capthick = 2
errobar_lw = 2
FLAG_PLOT = True
FLAG_IOFF = True

str_method = ['MCMC', diff_model.name.replace('_', ' ').capitalize()]

if FLAG_IOFF:
	plt.ioff()


# sample index for plot
for sample_plot in sample_range:

	km_obs = {key: dset_test[sample_plot, dict_idx[key][0], dict_idx[key][1]] for key in km_param_str}
	km_obs['k2p'] = load_test_dict['vark2p'][sample_plot]
	y_obs = label_test[sample_plot][None,]

	# load MCMC
	iter_mcmc = 20000
	burn_in_mcmc = 40000
	step_name = 'MH_MCMC'   # 'MCMC'
	mcmc_roi_dir = os.path.join(load_km_dir_test, 'MCMC{}'.format(str_noise))
	save_dir_filename = os.path.join(
		mcmc_roi_dir, '{}_nROI{}_it{:.1e}_brn{:.1e}_km_obs-{:.3f}-{:.3f}-{:.3f}.pik'.format(
			step_name, n_ROI, iter_mcmc, burn_in_mcmc, km_obs['DVR'][0], km_obs['R1'][0], km_obs['k2p']
		))
	try:
		mcmc_res = pickle.load(open(save_dir_filename, 'rb'))
		flag_load_mcmc = True
	except FileNotFoundError:
		flag_load_mcmc = False
		print('\nNo MCMC file found.\n')
		# raise ValueError('')


	## Load trained model and infer neural network

	for count, epoch_i in enumerate(tqdm.tqdm(make_png_period, desc="epoch")): #range(period, epochs + period, period)): #enumerate([epochs,]):
		## Testing
		if not load_model:
			load_weight_dir = save_model_dir
			load_model_dir = save_model_dir
		else:
			load_weight_dir = os.path.join(save_model_dir, 'cp_{}'.format(epoch_i))
			load_model_dir = os.path.join(load_weight_dir, str_noise)
			if not os.path.isdir(load_model_dir):
				os.makedirs(load_model_dir)

		save_post_res_fname = os.path.join(load_model_dir,
										'posterior_epoch{}_km_obs-{:.3f}-{:.3f}-{:.3f}.pik'.format(
											epoch_i, km_obs['DVR'][0],
											km_obs['R1'][0], km_obs['k2p']))
		save_post_all_t_res_fname = save_post_res_fname.replace(os.sep + 'posterior_', os.sep + 'posterior_all_t_')

		if flag_load_res and os.path.isfile(save_post_res_fname):
			with open(save_post_res_fname, 'rb') as file_i:
				load_res = pickle.load(file_i)
			km_pred, pred, dict_idx = load_res['km_pred'], load_res['pred'], load_res['dict_idx']
			km_pred_dict = {key: km_pred[:, dict_idx[key][0], dict_idx[key][1]]
					for key in km_param_str}
			if os.path.isfile(save_post_all_t_res_fname):
				with open(save_post_all_t_res_fname, 'rb') as file_i:
					km_pred_all_t = pickle.load(file_i)['km_pred_all_t']

		else:
			# diff_model.set_weights(tf.keras.models.load_model(os.path.join(
			# 	load_weight_dir, 'ckpt.weights.h5')).get_weights())
			diff_model.load_weights(os.path.join(load_weight_dir, 'ckpt.weights.h5'))
			km_pred_list, km_pred_all_t_list = [], []
			if 'km_pred_T' not in locals():
				km_pred_T = tf.random.normal((n_posterior, n_ROI, 2), dtype=TF_DTYPE)
			km_pred = km_pred_T

			for x in tqdm.tqdm(hlp.chunker(km_pred, chunk_size), total=n_posterior // chunk_size, desc="chunk"):
				x = diff_model.ddpm_loop(x, condition=np.repeat(y_obs, [chunk_size, ], axis=0),
										 keep_all_xt=keep_all_xt)  # Deactivate ``keep_all_xt`` if not needed
				if keep_all_xt:
					x_all_t = x
					km_pred_all_t_list.append(x_all_t)
					x = x[-1,]
				km_pred_list.append(x)

			km_pred = np.concatenate(km_pred_list, axis=0)
			km_pred_dict = {key: km_pred[:, dict_idx[key][0], dict_idx[key][1]]
								  for key in km_param_str}
			if keep_all_xt:
				km_pred_all_t = np.concatenate(km_pred_all_t_list, axis=0)

			pred = {'mean_' + key: km_pred[:, dict_idx[key][0], dict_idx[key][1]].mean(axis=0)
					for key in km_param_str}
			pred.update({'std_' + key: km_pred[:, dict_idx[key][0], dict_idx[key][1]].std(axis=0)
						 for key in km_param_str})

			if save_posterior_res:
				with open(save_post_res_fname, 'wb') as file_pi:
					pickle.dump({'km_pred': km_pred, 'pred': pred, 'dict_idx': dict_idx,
								 'km_param_str': km_param_str}, file_pi)
				if keep_all_xt:
					with open(save_post_all_t_res_fname, 'wb') as file_pi:
						pickle.dump({'km_pred_all_t': km_pred_all_t}, file_pi)

			_ = gc.collect()

		## Movie figure with posterior distribution for all timesteps
		if keep_all_xt and FLAG_PLOT and iio is not None:
			tmp_dir = os.path.join(load_model_dir, 'tmp_files')
			if not os.path.isdir(tmp_dir):
				os.makedirs(tmp_dir)
			frame_files_fnames = {key: [] for key in km_param_str}

			gamma = 0.5
			num_images = 100
			gif_duration = 4
			u = np.linspace(0, 1, num_images)
			v = u ** gamma  # concave when gamma<1
			idx = np.round(v * (timesteps_model - 1)).astype(int)
			timesteps_range_plot = np.unique(idx).tolist()  # remove duplicates after rounding
			frame_duration = gif_duration / len(timesteps_range_plot)
			ROI_range_show = np.arange(0, 4).tolist()

			for pp, km_pred_str_i in enumerate(km_param_str):
				for t_i in tqdm.tqdm(timesteps_range_plot, desc=km_pred_str_i):
					save_tmp_name = 'dist_{}_epoch{}_km_obs-{:.3f}-{:.3f}-{:.3f}_t{}.png'.format(
						km_pred_str_i, epoch_i, km_obs['DVR'][0],
						km_obs['R1'][0], km_obs['k2p'], timesteps_model - 1 - t_i)
					frame_files_fnames[km_pred_str_i].append(os.path.join(tmp_dir, save_tmp_name))
					if not os.path.isfile(os.path.join(tmp_dir, save_tmp_name)):
						var_pred_all = km_pred_all_t[t_i][:, dict_idx[km_pred_str_i][0], dict_idx[km_pred_str_i][1]]
						if flag_load_mcmc:
							var_mcmc_all = (mcmc_res[km_pred_str_i + '_mcmc'].
											reshape(-1, mcmc_res[km_pred_str_i + '_mcmc'].shape[-1]))
							var_all_list = [var_mcmc_all, var_pred_all]
						else:
							var_mcmc_all = var_pred_all
							str_method = str_method[1]
							var_all_list = [var_pred_all,]

						fig_all, ax_all = plt.subplots(int(np.ceil(np.sqrt(len(ROI_range_show)))),
													   int(np.ceil(np.sqrt(len(ROI_range_show)))),
													   figsize=(16, 10))
						for ni in ROI_range_show:
							for cc_method, var_tmp in enumerate(var_all_list):
								if len(ROI_range_show) == 1:
									ax_all = np.array(ax_all)
								ax_all.flatten()[ni].hist(var_tmp[:, ni].flatten(),
														  bins=100, color=[hist_mcmc_color, hist_pred_color][cc_method],
														  label=[str_method[cc_method]],
														  density=True, alpha=0.7, edgecolor='black',
														  linewidth=hist_linewidth_dist)
								x_axis = np.linspace(ax_all.flatten()[ni].axis()[0],
													 ax_all.flatten()[ni].axis()[1], num=500)
								ax_all.flatten()[ni].plot(x_axis, norm.pdf(x_axis, np.mean(var_tmp[:, ni].flatten()),
																		   np.std(var_tmp[:, ni].flatten())),
														  linewidth=3., color=['red','blue'][cc_method])
								ax_all.flatten()[ni].set_title(target_ROI_names[ni].replace('-', ' '), fontsize=12)
							ax_all.flatten()[ni].axvline(x=var_mcmc_all[..., ni].mean(),
														 color='black', linestyle='--', label='Posterior mean')

						fig_all.suptitle(f'Timestep {timesteps_model - 1 - t_i}', fontweight="bold", fontsize=15)
						ax_all.flatten()[0].legend()
						fig_all.tight_layout()

						for fmt_save in ['eps', 'pdf', 'png']:
							save_tmp_name = 'dist_{}_epoch{}_km_obs-{:.3f}-{:.3f}-{:.3f}_t{}.{}'.format(
									km_pred_str_i, epoch_i, km_obs['DVR'][0],
									km_obs['R1'][0], km_obs['k2p'], timesteps_model - 1 - t_i, fmt_save)
							fig_all.savefig(os.path.join(tmp_dir, save_tmp_name))
						plt.close(fig_all)

				frames = [iio.imread(f) for f in frame_files_fnames[km_pred_str_i]]
				save_gif_name = (frame_files_fnames[km_pred_str_i][-1].split(os.sep)[-1].
				                 replace(f'_t{timesteps_model - 1 - t_i}', ''))
				iio.imwrite(os.path.join(load_model_dir, save_gif_name.replace('.png', '.gif')),
				            frames, duration=frame_duration, loop=0)  # loop=0 => forever

				# Optional MP4 (requires imageio-ffmpeg backend installed)
				fps = 1.0 / frame_duration
				iio.imwrite(os.path.join(load_model_dir, save_gif_name.replace('.png', '.mp4')), frames, fps=fps)

		plt.close('all')

		## Bar Plot

		if FLAG_PLOT:
			fig_barplot, axs = plt.subplot_mosaic("AA;BB", figsize=(11, 6))
			width = 0.4
			multiplier = 1
			offset = width * multiplier
			x = np.linspace(0, n_ROI - 1, num=n_ROI)

			for ax_str, key in zip(['A', 'B'], km_param_str):
				axs[ax_str].bar(x + offset, pred['mean_' + key], width=width,
								yerr=ratio_mul_std_errorbar * pred['std_' + key], label=str_method[1],
								color=hist_pred_color, alpha=1, ecolor='black',
								edgecolor='black', linewidth=hist_linewidth_bar,
								error_kw=dict(lw=errobar_lw, capsize=capsize, capthick=capthick))
				if flag_load_mcmc:
					mcmc_mean = mcmc_res[key + '_mcmc'].mean(axis=(0, 1))
					mcmc_std = mcmc_res[key + '_mcmc'].std(axis=(0, 1))
					rect = axs[ax_str].bar(x, mcmc_res[key + '_mcmc'].mean(axis=(0, 1)),
										   width=width, yerr=ratio_mul_std_errorbar * mcmc_std,
										   label=str_method[0], color=hist_mcmc_color, alpha=1, ecolor='black',
										   edgecolor='black', linewidth=hist_linewidth_bar,
										   error_kw=dict(lw=errobar_lw, capsize=capsize, capthick=capthick))
					ax_ylim = (hlp.round_base(0.9 * (mcmc_mean - ratio_mul_std_errorbar * mcmc_std).min(), 5, 2),
							   hlp.round_base(1.05 * (mcmc_mean + ratio_mul_std_errorbar * mcmc_std).max(), 5, 2))
				else:
					ax_ylim = (hlp.round_base(0.9 * (pred['mean_' + key] -
													 ratio_mul_std_errorbar * pred['std_' + key]).min(), 5, 2),
							   hlp.round_base(1.05 * (pred['mean_' + key] +
													 ratio_mul_std_errorbar * pred['std_' + key]).max(), 5, 2))

				axs[ax_str].set(xlim=(x[0] - 1.5 * width, x[-1] + 2.5 * width))
				axs[ax_str].set(ylim=ax_ylim)
				ytick_i = np.linspace(ax_ylim[0], ax_ylim[1], num=7)[:-1]
				axs[ax_str].set_yticks(ytick_i, fontsize=font_ticks,
									   labels=map(lambda n: format(n, '.2f'), ytick_i.tolist()))
			# Hide the right and top spines
			# axs[ax_str].spines[['right', 'top']].set_visible(False)

			axs['A'].legend(prop={'size': font_legend})
			axs['A'].set_title('Mean and SD of the posterior distribution (ROI-wise)',
							   fontweight="bold", fontsize=font_title, loc='center')
			axs['A'].set_xticks(x + width / 2, fontsize=font_ticks,
								labels=map(format, np.arange(n_ROI).tolist()))
			axs['A'].set_ylabel('DVR', fontweight="bold", fontsize=font_label)
			# axs['A'].set_xlabel('ROIs')
			for ax in [axs['B']]:
				ax.set_xticks(x + width / 2,
							  labels=map(format, np.arange(n_ROI).tolist()),
							  fontsize=font_ticks)
				ax.set_xlabel('ROI index', fontweight="bold", fontsize=font_label)
				ax.set_ylabel('R1', fontweight="bold", fontsize=font_label)
			# ax.legend(prop={'weight': "bold", 'size': font_legend})
			fig_barplot.tight_layout()

			for fmt_save in ['eps', 'pdf', 'png']:
				fig_barplot.savefig(os.path.join(load_model_dir, 'barplot_epoch{}_km_obs-{:.3f}-{:.3f}-{:.3f}.{}'.format(
					epoch_i, km_obs['DVR'][0],
					km_obs['R1'][0], km_obs['k2p'], fmt_save)))
			plt.close(fig_barplot)

			## Distribution plot (ISBI abstract-like figure)

			fig_dist, axs = plt.subplot_mosaic("CD", figsize=(8, 3))
			# Distribution plots
			ax_DVR = axs['C']
			ax_R1 = axs['D']

			legend = []

			for pp, var in enumerate(km_param_str):
				ax_i = [ax_DVR, ax_R1][pp]
				if plot_type == 'hist':
					ax_i.hist(km_pred_dict[var][:, roi_plot], bins=40, color=hist_pred_color, label=[str_method[1]],
							  density=True, alpha=1, edgecolor='black', linewidth=hist_linewidth_dist)
					x_axis = np.linspace(ax_i.axis()[0], ax_i.axis()[1], 1000)
					ax_i.plot(x_axis,
							  norm.pdf(x_axis, np.mean(km_pred_dict[var][:, roi_plot]),
									   np.std(km_pred_dict[var][:, roi_plot])),
							  color='blue', linewidth=4., linestyle='--')

					# MCMC
					if flag_load_mcmc:
						mcmc_tmp_dist = mcmc_res[var + '_mcmc'][:, :, roi_plot].flatten()
						ax_i.hist(mcmc_tmp_dist, bins=40, color=hist_mcmc_color,
								  label=[str_method[0]], density=True, alpha=1, edgecolor='black', linewidth=hist_linewidth_dist)
						x_axis = np.arange(ax_i.axis()[0], ax_i.axis()[1], 1e-4)
						ax_i.plot(x_axis, norm.pdf(x_axis, np.mean(mcmc_tmp_dist), np.std(mcmc_tmp_dist)),
								  color='red', linewidth=4., linestyle='--')
						ax_i.set_xlabel(var, fontweight="bold", fontsize=font_label - 2)
					else:
						mcmc_tmp_dist = km_pred_dict[var][:, roi_plot]

				mean_mcmc = np.mean(mcmc_tmp_dist)
				mean_pred = pred['mean_' + var][roi_plot]
				std_mcmc = np.std(mcmc_tmp_dist)
				std_pred = pred['std_' + var][roi_plot]

				ax_xlim = ((mean_mcmc + mean_pred) / 2 - 3.8 * std_mcmc,
						   (mean_mcmc + mean_pred) / 2 + 3.8 * std_mcmc)
				ax_i.set_xlim(ax_xlim)
				xtick_i = np.linspace(ax_xlim[0], ax_xlim[1], num=5)[1:-1]
				ax_i.set_xticks(xtick_i, fontsize=font_ticks,
								labels=map(lambda n: format(n, '.2f'), xtick_i.tolist()))

				ax_ylim = ax_i.get_ylim()
				ax_i.set_ylim(ax_ylim)
				# ytick_i = hlp.round_base(np.linspace(hlp.round_base(ax_ylim[0], 5, 0),
				#                       hlp.round_base(ax_ylim[1], 5, 0), num=4)[:-1], 5, 0)
				ytick_i = np.array([0., hlp.round_base(ax_ylim[1] // 2, 5, 0), hlp.round_base(ax_ylim[1], 5, 0)])
				ax_i.set_yticks(ytick_i, fontsize=font_ticks,
								labels=map(lambda n: format(n, '.0f'), ytick_i.tolist()))

				ax_i.set_title('p(x$_{' + var + '}$|y) for ROI #' + format(roi_plot),
							   fontweight="bold", fontsize=font_title)

				if var in 'DVR':
					ax_i.set_ylabel('Density', fontweight="bold", fontsize=font_label)
					ax_i.legend(prop={'size': font_legend})

			fig_dist.tight_layout()

			# Fine tuning axes position
			# ax_A_pos = list(axs['A'].get_position().bounds)
			# ax_A_pos[1] -= 0.07 * ax_A_pos[1]
			# axs['A'].set_position(ax_A_pos)
			#
			# ax_B_pos = list(axs['B'].get_position().bounds)
			# ax_B_pos[1] += 0.5 * ax_B_pos[1]
			# axs['B'].set_position(ax_B_pos)

			for fmt_save in ['eps', 'pdf', 'png']:
				fig_dist.savefig(os.path.join(load_model_dir, 'dist_epoch{}_km_obs-{:.3f}-{:.3f}-{:.3f}.{}'.format(
					epoch_i, km_obs['DVR'][0],
					km_obs['R1'][0], km_obs['k2p'], fmt_save)))

			plt.close(fig_dist)


		## Large Distribution plot for all ROIs

			for pp, km_pred_str_i in enumerate(km_param_str):
				var_pred_all = km_pred_dict[km_pred_str_i]
				if flag_load_mcmc:
					var_mcmc_all = (mcmc_res[km_pred_str_i + '_mcmc'].
									reshape(-1, mcmc_res[km_pred_str_i + '_mcmc'].shape[-1]))
					var_all_list = [var_mcmc_all, var_pred_all]
				else:
					var_mcmc_all = var_pred_all
					str_method = str_method[1]
					var_all_list = [var_pred_all,]

				err_mean = 100 * np.abs(np.mean(var_pred_all, axis=0) - np.mean(var_mcmc_all, axis=0)) / np.mean(var_mcmc_all, axis=0)
				err_std = 100 * np.abs(np.std(var_pred_all, axis=0) - np.std(var_mcmc_all, axis=0)) / np.std(var_mcmc_all,
																										   axis=0)

				fig_all, ax_all = plt.subplots(int(np.ceil(np.sqrt(var_pred_all.shape[-1]))),
											   int(np.ceil(np.sqrt(var_pred_all.shape[-1]))),
											   figsize=(16, 10))
				for ni in range(var_pred_all.shape[-1]):
					for cc_method, var_tmp in enumerate(var_all_list):
						if var_pred_all.shape[-1] == 1:
							ax_all = np.array(ax_all)
						ax_all.flatten()[ni].hist(var_tmp[:, ni].flatten(),
												  bins=100, color=[hist_mcmc_color, hist_pred_color][cc_method],
												  label=['{} m={}'.format(str_method[cc_method], ni)],
												  density=True, alpha=1, edgecolor='black',
												  linewidth=hist_linewidth_dist)
						x_axis = np.linspace(ax_all.flatten()[ni].axis()[0],
											 ax_all.flatten()[ni].axis()[1], num=500)
						ax_all.flatten()[ni].plot(x_axis, norm.pdf(x_axis, np.mean(var_tmp[:, ni].flatten()),
																   np.std(var_tmp[:, ni].flatten())),
												  linewidth=3., color=['red','blue'][cc_method])
						km_tmp = km_obs[km_pred_str_i][ni]
						ax_all.flatten()[ni].set_title('({}_obs = {:.2f}) '
													   '$\delta_\mu = {:.1f}$% - $\delta_\sigma = {:.1f}$%'.
													   format(km_pred_str_i, km_tmp,
															  err_mean[ni], err_std[ni]),
													   fontsize=10)

				ax_all.flatten()[0].legend()
				fig_all.tight_layout()

				for fmt_save in ['eps', 'pdf', 'png']:
					fig_all.savefig(
						os.path.join(load_model_dir, 'dist_ROIall_{}_epoch{}_km_obs-{:.3f}-{:.3f}-{:.3f}.{}'.format(
							km_pred_str_i, epoch_i, km_obs['DVR'][0],
							km_obs['R1'][0], km_obs['k2p'], fmt_save)))
				plt.close(fig_all)

			## Calculate metrics (mean, std and relative diff with MCMC)

		for pp, var in enumerate(km_param_str):
			if flag_load_mcmc and 'k2p' not in var:
				Cov_dict = {'MCMC': np.cov(mcmc_res[var + '_mcmc'].reshape([-1, n_ROI]), rowvar=False).reshape([n_ROI, n_ROI]),
							'NN':   np.cov(km_pred[..., pp], rowvar=False).reshape([n_ROI, n_ROI]),
							}
				Cov_dict['Norm_diff'] = np.abs((Cov_dict['MCMC'] - Cov_dict['NN']) / Cov_dict['MCMC'])
				Corr_dict = {'MCMC': np.corrcoef(mcmc_res[var + '_mcmc'].reshape([-1, n_ROI]),
												 rowvar=False) - np.identity(n_ROI),
							 'NN':   np.corrcoef(km_pred[..., pp], rowvar=False) - np.identity(n_ROI),
							 }
				Corr_dict['Norm_diff'] = np.abs((Corr_dict['MCMC'] - Corr_dict['NN']) / (Corr_dict['MCMC'] + np.identity(n_ROI)))
				mu_dict = {'MCMC': np.mean(mcmc_res[var + '_mcmc'].reshape([-1, n_ROI]), axis=0)[:, None],
						   'NN':   np.mean(km_pred[..., pp], axis=0)[:, None]
						   }
				mu_dict['Norm_diff'] = np.abs((mu_dict['MCMC'] - mu_dict['NN']) / mu_dict['MCMC'])

				std_dict = {'MCMC': np.sqrt(np.diag(Cov_dict['MCMC']))[:, None],
							'NN':   np.sqrt(np.diag(Cov_dict['NN']))[:, None],
							}
				std_dict['Norm_diff'] = np.abs((std_dict['MCMC'] - std_dict['NN']) / std_dict['MCMC'])

				mu_std_dict = {'mu': mu_dict, 'std': std_dict}

				vmin = {'mu':   [0, 0, 0],
						'std':  [0, 0, 0],
						}
				vmax = {'mu':   [1.2 * np.percentile(mu_dict['MCMC'], 98), 1.2 * np.percentile(mu_dict['MCMC'], 98),
								 None],
						'std':  [1.2 * np.percentile(std_dict['MCMC'], 98), 1.2 * np.percentile(std_dict['MCMC'], 98),
								 None]
						}

				if FLAG_PLOT:
					fig_mu, ax_mu = plt.subplots(1, 3, figsize=(10, 3.5))
					fig_std, ax_std = plt.subplots(1, 3, figsize=(10, 3.5))

					for ax_name in ['mu', 'std']:
						exec('ax_tmp = ax_' + ax_name)
						exec('fig_tmp = fig_' + ax_name)
						dict_tmp = mu_std_dict[ax_name]
						for iii, var_cov in enumerate(dict_tmp.keys()):
							triu_dict_tmp = np.triu(dict_tmp[var_cov]) if \
								dict_tmp[var_cov].shape[1] != 1 else dict_tmp[var_cov]
							pos = ax_tmp[iii].imshow(triu_dict_tmp, cmap='hot',
													 vmin=vmin[ax_name][iii], vmax=vmax[ax_name][iii])
							fig_tmp.colorbar(pos, ax=ax_tmp[iii], shrink=0.6)
							if var_cov in 'Norm_diff':
								inf_norm_diff = np.max(triu_dict_tmp)
								mean_diff = np.mean(triu_dict_tmp)
								ax_tmp[iii].set_title('Relative abs diff \n'
													  '$\ell_\infty$ = {:.3f}, '
													  '$mean$ = {:.3f}'.format(inf_norm_diff, mean_diff))
							else:
								ax_tmp[iii].set_title(var_cov.replace('_', ' '))
							ax_tmp[iii].xaxis.set_tick_params(labelbottom=False)
							ax_tmp[iii].yaxis.set_tick_params(labelleft=False)
							ax_tmp[iii].set_xticks([])
							ax_tmp[iii].set_yticks([])
						fig_tmp.suptitle('Epoch {}'.format(epoch_i))
						fig_tmp.subplots_adjust(top=0.85)
						fig_tmp.set_tight_layout(True)
						save_fig_cov_fname = os.path.join(load_model_dir,
														  '{}_{}_epoch{}_km_obs-{:.3f}-{:.3f}-{:.3f}.png'.format(
															  ax_name, var,
															  epoch_i, km_obs['DVR'][0],
															  km_obs['R1'][0], km_obs['k2p']))
						fig_tmp.savefig(save_fig_cov_fname)
					plt.close(fig_Cov)
					plt.close(fig_Corr)
					plt.close(fig_mu)
					plt.close(fig_std)

				save_stats_fname = os.path.join(load_model_dir,
												  'stats_{}_epoch{}_km_obs-{:.3f}-{:.3f}-{:.3f}.pik'.format(
													  var, epoch_i, km_obs['DVR'][0],
													  km_obs['R1'][0], km_obs['k2p']))
				with open(save_stats_fname, 'wb') as file_pi:
					pickle.dump(mu_std_dict, file_pi)

				ppp = 0
				for (key1, key1_val) in mu_std_dict.items():
					for key2, key2_val in key1_val.items():
						open_mode = 'wb' if ppp == 0 else 'ab'
						with open(save_stats_fname.replace('.pik', '.csv'), open_mode) as file_pi:
							np.savetxt(file_pi, key2_val, delimiter=',', fmt='%.2e',
									   header=key1 + '-' + key2)
						ppp += 1

					# json.dump(mu_std_dict, file_pi, indent=2, sort_keys=True)

		## Calculate ESS
		tmp = np.stack([np.asarray(mcmc_res['DVR_mcmc'], dtype=NP_DTYPE),
						np.asarray(mcmc_res['R1_mcmc'], dtype=NP_DTYPE)], axis=-1)
		ess_mcmc = tfp.mcmc.effective_sample_size(np.moveaxis(tmp, 0, -1),
												  cross_chain_dims=-1).numpy()
		ess_pred = tfp.mcmc.effective_sample_size(km_pred).numpy()

		ess_dict = {'MCMC': np.concatenate([ess_mcmc, np.mean(ess_mcmc, axis=-1, keepdims=True)], axis=-1),
					'NN': np.concatenate([ess_pred, np.mean(ess_pred, axis=-1, keepdims=True)], axis=-1),
					}

		save_ess_fname = os.path.join(load_model_dir,
										'ess_epoch{}_km_obs-{:.3f}-{:.3f}-{:.3f}.pik'.format(
											epoch_i, km_obs['DVR'][0],
											km_obs['R1'][0], km_obs['k2p']))
		with open(save_ess_fname, 'wb') as file_pi:
			pickle.dump(ess_dict, file_pi)

		for jj, (ess_name, ess_val) in enumerate(ess_dict.items()):
			open_mode = 'wb' if (jj == 0) else 'ab'
			with open(save_ess_fname.replace('.pik', '.csv'), open_mode) as file_pi:
				np.savetxt(file_pi, ess_val, delimiter=',', fmt='%d',
						   header='ESS DVR {0}, ESS R1 {0}, ESS mean {0}'.format(ess_name))

		## Save csv file with metrics for each sample

		list_pik_file_DVR = sorted(glob.glob(os.path.join(load_model_dir, 'stats_DVR_*.pik')))
		list_pik_file_R1 = sorted(glob.glob(os.path.join(load_model_dir, 'stats_R1_*.pik')))
		list_pik_file_ess = sorted(glob.glob(os.path.join(load_model_dir, 'ess_*.pik')))

		DVR_stats = [pickle.load(open(DVR_stat_file, 'rb')) for DVR_stat_file in list_pik_file_DVR]
		R1_stats = [pickle.load(open(R1_stat_file, 'rb')) for R1_stat_file in list_pik_file_R1]
		ess_stats = [pickle.load(open(ess_stat_file, 'rb')) for ess_stat_file in list_pik_file_ess]


		norm_diff = {'mu_DVR':  [100 * dic['Norm_diff'] for dic in [a['mu'] for a in DVR_stats]],
					 'std_DVR': [100 * dic['Norm_diff'] for dic in [a['std'] for a in DVR_stats]],
					 'mu_R1':   [100 * dic['Norm_diff'] for dic in [a['mu'] for a in R1_stats]],
					 'std_R1':  [100 * dic['Norm_diff'] for dic in [a['std'] for a in R1_stats]]
					 }

		ess_save = {'MCMC': np.stack([val['MCMC'][..., -1] for val in ess_stats], axis=-1),
					'NN':   np.stack([val['NN'][..., -1] for val in ess_stats], axis=-1)}
		ess_save['MCMC'] = np.concatenate([ess_save['MCMC'], np.mean(ess_save['MCMC'],
																	 axis=-1, keepdims=True)], axis=-1)
		ess_save['NN'] = np.concatenate([ess_save['NN'], np.mean(ess_save['NN'],
																 axis=-1, keepdims=True)], axis=-1)

		for jj, (ess_name, ess_val) in enumerate(ess_save.items()):
			open_mode = 'wb' if jj == 0 else 'ab'
			with open(os.path.join(load_model_dir, 'ess.csv'), open_mode) as file_pi:
				np.savetxt(file_pi, ess_val,
						   header='ESS ' + ess_name + ',' * (ess_val.shape[-1] - 1) + 'mean',
						   delimiter=',', fmt='%d')

		tmp_avg = {key: np.mean(np.array(val)[..., 0].T, axis=(1,))[:, None] for key, val in norm_diff.items()}

		if count == 0:
			norm_diff_avg = tmp_avg
		else:
			for key in norm_diff_avg.keys():
				norm_diff_avg[key] = np.hstack([norm_diff_avg[key], tmp_avg[key]])

		norm_diff_save_pik = {}
		for count_i, (stat_i, var_i) in enumerate(itertools.product(['mu', 'std'], ['DVR', 'R1'])):
			tmp = np.array(norm_diff[stat_i + '_' + var_i])[..., 0].T
			tmp_mean = np.mean(tmp, axis=(1,), dtype=np.float64)[:, None]
			norm_diff_save_pik.update({stat_i + '_' + var_i: tmp,
									   stat_i + '_' + var_i + '_mean': tmp_mean
									   })
			tmp = np.hstack([tmp, tmp_mean])

			open_mode = 'wb' if count_i == 0 else 'ab'
			with open(os.path.join(load_model_dir, 'norm_diff.csv'),
					  open_mode) as file_pi:
				np.savetxt(file_pi, tmp,
						   header=stat_i + ' ' + var_i + ', (in %)' + ',' * (tmp.shape[-1] - 2) + 'mean',
						   # footer='\n',
						   delimiter=',', fmt='%.2f')

		with open(os.path.join(load_model_dir, 'norm_diff.pik'),'wb') as file_i:
			pickle.dump(norm_diff_save_pik, file_i)

		if not load_model:
			break


# %% Creat csv and pik files with metrics such as MAPE (error)

norm_diff_load_mean = {}
for count_cp, cp in enumerate(make_png_period):
	# Save csv file for metrics
	# load norm_diff files for each checkpoint
	load_norm_diff_dir = os.path.join(save_model_dir, f'cp_{cp}', str_noise)
	norm_diff_load = pickle.load(open(os.path.join(load_norm_diff_dir, 'norm_diff.pik'), 'rb'))
	if count_cp == 0:
		norm_diff_load_mean.update({key: val for key, val in norm_diff_load.items() if '_mean' in key})
	else:
		norm_diff_load_mean.update({key: np.concatenate([norm_diff_load_mean[key], val], axis=-1)
									for key, val in norm_diff_load.items() if '_mean' in key})

for count_i, (stat_i, var_i) in enumerate(itertools.product(['mu', 'std'], ['DVR', 'R1'])):
	open_mode = 'wb' if count_i == 0 else 'ab'
	with open(os.path.join(save_model_dir,
						   'norm_diff_avg_{}.csv'.format(str_noise)),
			  open_mode) as file_pi:
		np.savetxt(file_pi, norm_diff_load_mean[stat_i + '_' + var_i + '_mean'],
				   header='mean ' + stat_i + ' ' + var_i + ', (in %)\n epoch ' +
						  format(make_png_period[0]) + ',' + ','.join(map(format, make_png_period[1:])),
				   delimiter=',', fmt='%.2f')

with open(os.path.join(save_model_dir,
					   'norm_diff_avg_{}.pik'.format(str_noise)),'wb') as file_pi:
	pickle.dump(norm_diff_load_mean, file_pi)

for count_i, (stat_i, var_i) in enumerate(itertools.product(['mu', 'std'], ['DVR', 'R1'])):
	open_mode = 'wb' if count_i == 0 else 'ab'
	with open(os.path.join(save_model_dir,
						   'norm_diff_avg_meanROI_{}.csv'.format(str_noise)), open_mode) as file_pi:
		np.savetxt(file_pi, np.mean(norm_diff_load_mean[stat_i + '_' + var_i + '_mean'], 0,
									keepdims=True),
				   header='mean ' + stat_i + ' ' + var_i + ', (in %)\n epoch ' +
						  format(make_png_period[0]) + ',' + ','.join(map(format, make_png_period[1:])),
				   delimiter=',', fmt='%.2f')


