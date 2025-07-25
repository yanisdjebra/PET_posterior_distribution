import tensorflow as tf
from tensorflow.keras import Model
import inspect
import numpy as np
import tqdm

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)
TF_DTYPE = tf.dtypes.as_dtype(DTYPE)
NP_DTYPE = getattr(np, DTYPE)

import networks as cl
import helper_func as hlp

np.random.seed(12345)
tf.random.set_seed(12345)
tf.keras.utils.set_random_seed(12345)

# Clear previously registered custom objects from this module
del_key = []
for key in tf.keras.utils.get_custom_objects().keys():
	if 'CustomModels' in key:
		del_key.append(key)
for del_key_i in del_key:
	tf.keras.utils.get_custom_objects().pop(del_key_i)


## Diffusion Model

@tf.keras.utils.register_keras_serializable(package="CustomModels")
class DDPM(Model):
	"""
	Generic diffusion model class with built-in training method.
	Can be used with any type of input network, so long they
	take as argument the input Tensor ``x``, time embedding
	``time`` and optional condition (for conditional generative
	networks) ``condition``.
	"""

	def __init__(self, timesteps=200, noise_schedule=None,
	             network=None, ndim=None, constrained_func_output_t0=None,
				 flag_cosine_similarity=False, cosine_similarity_dict_idx=None, **kwargs):
		"""
		Parameters
		----------

		:param timesteps: (int) Number of timestep for the diffusion process.
		:param noise_schedule: (dict) Dictionary containing
		``{'beta_start', 'beta_end'}`` corresponding to the low
		and high value of beta. The scheduled noise alpha is then
		calculated from beta as ``alpha = 1 - beta``. Defaults to
		``{'beta_start': 1e-4, 'beta_end': 2e-2}``
		:param network: (tf.keras.layers.Layer or Model) Tensorflow Neural network
		to process the data (e.g., U-Net). The model has to take **at least
		1 named input** in addition to the input data to process:
		``time`` (int), the current timestep of the diffusion process to be
		embedded in the network. A condition can also be optionally passed
		for conditional generative networks, using the keyword ``condition``.
		Note: the Layer must be already built and ideally has an ``ndim``
		attribute (``ndim = len(input_shape) - 2``). If the network uses
		layers such as Dropout or Batch Norm, make sure to pass the
		``training`` argument to these layers.
		:param ndim: (int) Rank of the input data, excluding batch and channel
		dimensions: ``len(input_shape) - 2`` (e.g, (batch_size, M, N,
		channel) --> ndim = 2). Fetched from network.ndim or network.input_shape
		if attribute exists.
		:param: constrained_func_output_t0: (func) Constrain the output
		of the model at the timestep t=0. Can be a user-defined function
		tf function, or lambda function.

		Careful: ``call`` function takes a dict input with keys ``x``, ``time``, and ``condition``.

		kwargs
		----------

		"""
		super(DDPM, self).__init__()

		self.timesteps = timesteps
		self.noise_schedule = noise_schedule
		self.constrained_func_output_t0 = constrained_func_output_t0
		self.constrained_func_output_t0_eval = self.constrained_func_output_t0


		self.beta_start = 1e-4 if noise_schedule is None else noise_schedule.get('beta_start', 1e-4)
		self.beta_end = 2e-2 if noise_schedule is None else noise_schedule.get('beta_end', 2e-2)
		self.offset_s = 0.008 if noise_schedule is None else noise_schedule.get('offset_s', 0.008)
		self.max_beta = 0.999 if noise_schedule is None else noise_schedule.get('max_beta', 0.999)

		self.schedule_name = 'linear' if noise_schedule is None else noise_schedule.get('schedule_name', 'linear')

		self.beta = hlp.get_beta_schedule(self.schedule_name, self.timesteps,
		                                  beta_start=self.beta_start,
		                                  beta_end=self.beta_end,
		                                  offset_s=self.offset_s,
		                                  max_beta=self.max_beta)

		self.alpha = 1 - self.beta
		self.alpha_bar = np.cumprod(self.alpha, 0, dtype=DTYPE)
		self.alpha_bar = np.concatenate((np.array([1.], dtype=DTYPE),
		                                      self.alpha_bar[:-1]), axis=0)
		self.alpha_bar_prev = np.concatenate((self.alpha_bar[1:],
		                                      np.array([0.], dtype=DTYPE)), axis=0)
		self.sqrt_alpha_bar = np.sqrt(self.alpha_bar, dtype=DTYPE)
		self.sqrt_one_minus_alpha_bar = np.sqrt(1 - self.alpha_bar, dtype=DTYPE)

		self.flag_cosine_similarity = flag_cosine_similarity
		self.cosine_similarity_dict_idx = cosine_similarity_dict_idx

		# Check input network
		if network is not None:
			self.network = network
		else:
			raise ValueError('Input network is required to create model.')

		# Check operating number of dimension
		if ndim is not None:
			self.ndim = ndim
		else:
			if hasattr(network, 'ndim'):
				self.ndim = network.ndim
			elif hasattr(network, 'input_shape'):
				self.ndim = len(network.input_shape) - 2
			else:
				raise AttributeError('``ndim`` was not passed as argument. '
				                     'Could not be retrieved from input network'
				                     ' {} (``input_shape`` attribute '
				                     'does not exist).'.format(network.name))

		self.reshape_dim = (-1,) + (1,) * (self.ndim + 1)

		if 'condition' in inspect.getfullargspec(network.call)[0]:
			self.flag_condition = True
		else:
			self.flag_condition = False

		self.loss_tracker = tf.keras.metrics.Mean(name="loss")
		self.noise_loss_tracker = tf.keras.metrics.Mean(name="noise_loss")


	def call(self, inputs, **kwargs):
		"""Model call. inputs is a dict containing ``x``, ``time``, and ``condition``."""
		time = inputs.get('time', None)
		condition = inputs.get('condition', None)
		x = inputs.get('x')
		output = self.network(x, time=time, condition=condition, **kwargs)

		if self.constrained_func_output_t0_eval is not None:
			if hasattr(self, 'flag_learn_var'):
				prediction, logvar = tf.split(output, num_or_size_splits=2, axis=-1)
				prediction = tf.where(tf.reshape(time, self.reshape_dim) == 0,
									  self.constrained_func_output_t0_eval(prediction), prediction)
				output = tf.concat([prediction, logvar], axis=-1)
			else:
				output = tf.where(tf.reshape(time, self.reshape_dim) == 0,
									  self.constrained_func_output_t0_eval(output), output)

		return output

	def compile(self, **kwargs):
		super().compile(**kwargs)
		self.loss_fn = tf.keras.losses.get(kwargs['loss'])

	@property
	def metrics(self):
		return [self.loss_tracker, self.noise_loss_tracker]

	def cosine_similarity_loss(self, y_true, y_pred):
		# Convert angles to sine-cosine representation
		y_true_cos, y_true_sin  = tf.cos(y_true), tf.sin(y_true)
		y_pred_cos, y_pred_sin = tf.cos(y_pred), tf.sin(y_pred)

		# Compute cosine similarity loss
		loss = 1 - (y_pred_sin * y_true_sin + y_pred_cos * y_true_cos)
		return tf.reduce_sum(tf.reduce_mean(loss, axis=0))

	def compute_loss(self, x=None, y=None, y_pred=None):
		"""
		Compute loss (e.g., l2 norm between noise and predicted noise).
		Regularization losses (as well as any other loss in self.network
		passed in init) are also calculated here.
		"""
		noise_loss = self.loss(y_true=y, y_pred=y_pred)
		reg_loss = 0.
		if len(self.losses) > 0:
			reg_loss = tf.add_n(self.losses)

		cosine_loss = 0.
		if self.flag_cosine_similarity and self.cosine_similarity_dict_idx is not None:
			for key, val in self.cosine_similarity_dict_idx.items():
				angle = tf.gather(tf.gather(y, val[0], axis=1), val[1], axis=2)
				angle_pred = tf.gather(tf.gather(y_pred, val[0], axis=1), val[1], axis=2)
				cosine_loss += tf.keras.losses.cosine_similarity(tf.stack([tf.cos(angle), tf.sin(angle)], -1),
																 tf.stack([tf.cos(angle_pred), tf.sin(angle_pred)], -1),
																 axis=-1)
				# cosine_loss += - tf.reduce_mean(1e-1 * tf.cos(angle - angle_pred))
		loss = noise_loss + reg_loss + cosine_loss
		return loss, noise_loss

	# @tf.function
	def train_step(self, data):
		"""
		Override training step in model.fit() for the diffusion process.
		A loss needs to be provided when compiling the model (e.g., l2 norm).
		"""
		images = data[0]
		if len(data) < 2:
			condition = None
		else:
			condition = data[1]
		images_shape = tf.shape(images)
		batch_size = tf.shape(images)[0]

		t = tf.random.uniform(shape=[batch_size],
		                      minval=0, maxval=self.timesteps,
		                      dtype=tf.int32)
		noise = tf.random.normal(shape=images_shape, dtype=TF_DTYPE)

		# Retrieve the current timestep for each batch, and reshape for broadcast
		sqrt_alpha_bar_t = tf.reshape(tf.gather(self.sqrt_alpha_bar, t, axis=0), self.reshape_dim)
		sqrt_one_minus_alpha_bar_t = tf.reshape(tf.gather(self.sqrt_one_minus_alpha_bar, t, axis=0), self.reshape_dim)
		noised_image = sqrt_alpha_bar_t * images + sqrt_one_minus_alpha_bar_t * noise

		with tf.GradientTape() as tape:
			prediction = self.call({'x': noised_image, 'time': t, 'condition': condition}, training=True)
			loss, noise_loss = self.compute_loss(y=noise, y_pred=prediction)

		gradients = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		self.loss_tracker.update_state(loss)
		self.noise_loss_tracker.update_state(noise_loss)

		return {m.name: m.result() for m in self.metrics}

	# @tf.function
	def test_step(self, data):
		"""
		Override default validation step in model.evaluate() for the diffusion process.
		A loss needs to be provided when compiling the model (e.g., l2 norm).
		"""
		images = data[0]
		if len(data) < 2:
			condition = None
		else:
			condition = data[1]
		images_shape = tf.shape(images)
		batch_size = tf.shape(images)[0]

		t = tf.random.uniform(shape=[batch_size],
		                      minval=0, maxval=self.timesteps,
		                      dtype=tf.int32)
		noise = tf.random.normal(shape=images_shape, dtype=TF_DTYPE)

		# Retrieve the current timestep for each batch, and reshape for broadcast
		sqrt_alpha_bar_t = tf.reshape(tf.gather(self.sqrt_alpha_bar, t, axis=0), self.reshape_dim)
		sqrt_one_minus_alpha_bar_t = tf.reshape(tf.gather(self.sqrt_one_minus_alpha_bar, t, axis=0), self.reshape_dim)
		noised_image = sqrt_alpha_bar_t * images + sqrt_one_minus_alpha_bar_t * noise

		prediction = self.call({'x': noised_image, 'time': t, 'condition': condition}, training=False)
		loss, noise_loss = self.compute_loss(y=noise, y_pred=prediction)

		self.loss_tracker.update_state(loss)
		self.noise_loss_tracker.update_state(noise_loss)

		return {m.name: m.result() for m in self.metrics}

	@tf.function
	def ddpm(self, x_t, time, condition=None):
		"""Predicts x^{t-1} based on x^{t} using the DDPM model.
		Use in a for loop from T to 1 to retrieve input from noise."""

		alpha_t = tf.reshape(tf.gather(self.alpha, time, axis=0), self.reshape_dim)
		alpha_bar_t_prev = tf.reshape(tf.gather(self.alpha_bar, time - 1, axis=0), self.reshape_dim)
		alpha_bar_t = tf.reshape(tf.gather(self.alpha_bar, time, axis=0), self.reshape_dim)
		sqrt_one_minus_alpha_bar_t = tf.reshape(tf.gather(
			self.sqrt_one_minus_alpha_bar, time, axis=0), self.reshape_dim)
		pred_noise = self.call({'x': x_t, 'time': time, 'condition': condition}, training=False)

		eps_coef = (1 - alpha_t) / sqrt_one_minus_alpha_bar_t
		mean = (1 / tf.sqrt(alpha_t)) * (x_t - eps_coef * pred_noise)

		beta_t = tf.reshape(tf.gather(self.beta, time, axis=0), self.reshape_dim)
		var_beta = beta_t
		var_beta_tilde = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t

		z = tf.random.normal(shape=tf.shape(x_t), dtype=x_t.dtype)

		return mean, tf.sqrt(var_beta) * z, tf.sqrt(var_beta_tilde) * z

	def get_config(self):
		"""
		Serialize object
		"""
		config = super().get_config()
		config.update({
			'timesteps':      self.timesteps,
			'noise_schedule': self.noise_schedule,
			'ndim':           self.ndim,
			'network':        tf.keras.layers.serialize(self.network),
			'constrained_func_output_t0': self.constrained_func_output_t0,
			'flag_cosine_similarity': self.flag_cosine_similarity,
			'cosine_similarity_dict_idx': self.cosine_similarity_dict_idx,
		})
		return config

	@classmethod
	def from_config(cls, config):
		"""
		Create object from configuration.
		"""
		config["network"] = tf.keras.layers.deserialize(config["network"])
		return cls(**config)



@tf.keras.utils.register_keras_serializable(package="CustomModels")
class ImprovedDDPM(DDPM):
	def __init__(self, lambda_vlb, parameterization='eps', **kwargs):
		super().__init__(**kwargs)
		self.lambda_vlb = lambda_vlb
		self.parameterization = parameterization
		self.eps_param_name_list = ['eps', 'epsilon']
		self.x0_param_name_list = ['x0', 'x_0', 'x_start', 'xstart', 'start_x']
		self.x_prev_param_name_list = ['x_{t-1}', 'x_prev', 'xprev', 'prev_x']
		self.v_param_name_list = ['v',]
		self.all_param_name_list = (self.eps_param_name_list + self.x0_param_name_list +
									self.x_prev_param_name_list + self.v_param_name_list)
		if self.parameterization.lower() not in self.all_param_name_list:
			raise ValueError(f'Invalid parameterization (got ``{parameterization}``). '
							 f'Value must be in ``{self.all_param_name_list}``')

		self.learn_variance = kwargs['network'].learn_variance
		self.flag_learn_var = 'learn' in self.learn_variance.lower()
		self.flag_ranged_var = 'ranged' in self.learn_variance.lower()

		self.alpha_bar = np.cumprod(self.alpha, 0, dtype=DTYPE)
		self.alpha_bar_prev = np.concatenate((np.array([1.], dtype=DTYPE),
		                                      self.alpha_bar[:-1]), axis=0)
		self.sqrt_alpha_bar = np.sqrt(self.alpha_bar, dtype=DTYPE)
		self.sqrt_one_minus_alpha_bar = np.sqrt(1 - self.alpha_bar, dtype=DTYPE)

		# Added from IDDPM GitHub for learning variance
		# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
		# calculations for posterior q(x_{t-1} | x_t, x_0)
		self.posterior_variance = self.beta * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
		# log calculation clipped because the posterior variance is 0 at the
		# beginning of the diffusion chain.
		self.posterior_log_variance_clipped = self.posterior_variance
		self.posterior_log_variance_clipped[0] = self.posterior_log_variance_clipped[1]
		self.posterior_log_variance_clipped = np.log(self.posterior_log_variance_clipped)

		self.posterior_mean_coef1 = self.beta * np.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
		self.posterior_mean_coef2 = (1.0 - self.alpha_bar_prev) * np.sqrt(self.alpha) / \
		                            (1.0 - self.alpha_bar)

		self.lambda_vlb_loss_tracker = tf.keras.metrics.Mean(name="lambda_vlb_loss")

	def compile(self, **kwargs):
		super().compile(**kwargs)

	@property
	def metrics(self):
		return super().metrics + [self.lambda_vlb_loss_tracker,]

	def _predict_xstart_from_xprev(self, x_t, t, xprev):
		return (self.extract_params(1.0 / self.posterior_mean_coef1, t) * xprev -
				self.extract_params(self.posterior_mean_coef2 / self.posterior_mean_coef1, t) * x_t)

	def _predict_xstart_from_eps(self, x_t, t, eps):
		# assert tf.shape(x_t) == tf.shape(eps)
		sqrt_alpha_bar_t = self.extract_params(self.sqrt_alpha_bar, t)
		alpha_bar_t = self.extract_params(self.alpha_bar, t)
		return 1.0 / sqrt_alpha_bar_t * x_t - tf.sqrt(1.0 / alpha_bar_t - 1) * eps

	def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
		alpha_bar_t = self.extract_params(self.alpha_bar, t)
		return (tf.sqrt(1.0 / alpha_bar_t) * x_t - pred_xstart) / tf.sqrt(1.0 / alpha_bar_t - 1)

	def _predict_xstart_from_v(self, x_t, t, v):
		"""
		Given v̂, recover x̂₀  (  x̂₀ = √ᾱₜ · xₜ  −  √(1-ᾱₜ) · v̂  ).
		"""
		sqrt_alpha_bar_t = self.extract_params(self.sqrt_alpha_bar, t)
		sqrt_one_minus_alpha_t = self.extract_params(self.sqrt_one_minus_alpha_bar, t)
		return sqrt_alpha_bar_t * x_t - sqrt_one_minus_alpha_t * v

	def _predict_eps_from_v(self, x_t, t, v):
		"""
		Given v̂, recover ε̂  (  ε̂ = √ᾱₜ · v̂  +  √(1-ᾱₜ) · xₜ  ).
		"""
		sqrt_alpha_bar_t = self.extract_params(self.sqrt_alpha_bar, t)
		sqrt_one_minus_alpha_t = self.extract_params(self.sqrt_one_minus_alpha_bar, t)
		return sqrt_alpha_bar_t * v + sqrt_one_minus_alpha_t * x_t

	def get_v(self, x, noise, t):
		return (self.extract_params(self.sqrt_alpha_bar, t) * noise -
				self.extract_params(self.sqrt_one_minus_alpha_bar, t) * x)

	def extract_params(self, param, t):
		return tf.reshape(tf.gather(param, t, axis=0), self.reshape_dim)

	def q_posterior_mean_variance(self, x_start, x_t, t):
		"""
		Compute the mean and variance of the diffusion posterior:

			q(x_{t-1} | x_t, x_0)

		"""

		posterior_mean_coef1 = tf.reshape(tf.gather(self.posterior_mean_coef1, t, axis=0),
		                                  self.reshape_dim)
		posterior_mean_coef2 = tf.reshape(tf.gather(self.posterior_mean_coef2, t, axis=0),
		                                  self.reshape_dim)
		posterior_mean = posterior_mean_coef1 * x_start + posterior_mean_coef2 * x_t

		posterior_variance = tf.reshape(tf.gather(self.posterior_variance, t, axis=0),
		                                self.reshape_dim)
		posterior_log_variance_clipped = tf.reshape(tf.gather(self.posterior_log_variance_clipped,
		                                                      t, axis=0), self.reshape_dim)

		return posterior_mean, posterior_variance, posterior_log_variance_clipped

	def p_mean_variance(self, model_output, x, t):
		"""
		Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
		the initial x, x_0.

		:param model_output: The model output, computed from an input batch and timesteps.
		:param x: the [N x C x ...] tensor at time t.
		:param t: a 1-D Tensor of timesteps.
		:param model_kwargs: if not None, a dict of extra keyword arguments to
			pass to the model. This can be used for conditioning.
		:return: a dict with the following keys:
				 - 'mean': the model mean output.
				 - 'variance': the model variance output.
				 - 'log_variance': the log of 'variance'.
				 - 'pred_xstart': the prediction for x_0.
		"""

		if self.flag_learn_var:
			model_output, model_var_values = tf.split(model_output, num_or_size_splits=2, axis=-1)
			if not self.flag_ranged_var:
				model_log_variance = model_var_values
				model_variance = tf.exp(model_log_variance)
			else:
				min_log = self.extract_params(self.posterior_log_variance_clipped, t)
				max_log = tf.math.log(self.extract_params(self.beta, t))
				# The model_var_values is [-1, 1] for [min_var, max_var].
				frac = (model_var_values + 1) / 2
				model_log_variance = frac * max_log + (1 - frac) * min_log
				model_variance = tf.exp(model_log_variance)
			model_variance_tilde, model_log_variance_tilde = \
				model_variance, model_log_variance
		else:
			model_variance, model_log_variance = self.beta, tf.math.log(self.beta)
			model_variance_tilde, model_log_variance_tilde = self.posterior_variance, \
				self.posterior_log_variance_clipped,
			model_variance = self.extract_params(model_variance, t)
			model_log_variance = self.extract_params(model_log_variance, t)
			model_variance_tilde = self.extract_params(model_variance_tilde, t)
			model_log_variance_tilde = self.extract_params(model_log_variance_tilde, t)

		pred_xstart, model_mean = self._predict_x_start(x, t, model_output)

		return {
			"mean":         model_mean,
			"variance":     model_variance,
			"log_variance": model_log_variance,
			"variance_tilde":     model_variance_tilde,
			"log_variance_tilde": model_log_variance_tilde,
			"pred_xstart":  pred_xstart,
		}

	def _predict_x_start(self, x, t, model_output):
		if self.parameterization.lower() in self.x_prev_param_name_list:
			# model_output is x_{t-1}
			pred_xstart = self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
			model_mean = model_output
		elif self.parameterization.lower() in self.x0_param_name_list + self.eps_param_name_list + self.v_param_name_list:
			if self.parameterization.lower() in self.x0_param_name_list:
				# model_output is x_0
				pred_xstart = model_output
			elif self.parameterization.lower() in self.v_param_name_list:
				# model_output is v̂ (reperameterized noise)
				pred_xstart = self._predict_xstart_from_v(x_t=x, t=t, v=model_output)
			elif self.parameterization.lower() in self.eps_param_name_list:
				# model_output is epsilon (noise)
				pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
			else:
				raise ValueError('Unknown parameterization')

			model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)


		return pred_xstart, model_mean

	def _vb_terms_bpd(
			self, model_output, x_start, x_t, t, **kwargs):
		"""
		Get a term for the variational lower-bound.

		The resulting units are bits (rather than nats, as one might expect).
		This allows for comparison to other papers.

		:return: a dict with the following keys:
				 - 'output': a shape [N] tensor of NLLs or KLs.
				 - 'pred_xstart': the x_0 predictions.
		"""
		out = self.p_mean_variance(model_output=model_output, x=x_t, t=t)
		true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
			x_start=x_start, x_t=x_t, t=t)
		kl = cl.normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
		kl = tf.math.reduce_mean(kl, axis=range(1, len(self.reshape_dim))) / tf.cast(tf.math.log(2.), TF_DTYPE)

		bin_width = self.freedman_diaconis_rule(iqr=1.34896,
		                                        n_samples=tf.reduce_prod(tf.shape(x_start)[:-1]))
		decoder_nll = -cl.discretized_gaussian_log_likelihood(
			x=x_start, means=out["mean"], log_scales=0.5 * out["log_variance"], bin_width=bin_width)

		decoder_nll = tf.math.reduce_mean(decoder_nll,
		                                  axis=range(1, len(self.reshape_dim))) / tf.cast(tf.math.log(2.), TF_DTYPE)

		# At the first timestep return the decoder NLL,
		# otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
		output = tf.where((t == 0), decoder_nll, kl)
		return output

	def freedman_diaconis_rule(self, iqr, n_samples):
		return 2 * iqr / tf.pow(tf.cast(n_samples, TF_DTYPE), 1. / 3.)

	# @tf.function
	def train_step(self, data, debug=False):
		""" Override training step in model.fit() for the diffusion process.
		A loss needs to be provided when compiling the model (e.g., l2 norm)."""
		images = data[0]
		if not isinstance(data, tuple):
			condition = None
		else:
			condition = data[1]
		images_shape = tf.shape(images)
		batch_size = tf.shape(images)[0]

		t = tf.random.uniform(shape=[batch_size],
		                      minval=0, maxval=self.timesteps,
		                      dtype=tf.int32)
		noise = tf.random.normal(shape=images_shape, dtype=TF_DTYPE)

		# Retrieve the current timestep for each batch, and reshape for broadcast
		sqrt_alpha_bar_t = tf.reshape(tf.gather(self.sqrt_alpha_bar, t, axis=0), self.reshape_dim)
		sqrt_one_minus_alpha_bar_t = tf.reshape(tf.gather(self.sqrt_one_minus_alpha_bar, t, axis=0), self.reshape_dim)
		noised_image = sqrt_alpha_bar_t * images + sqrt_one_minus_alpha_bar_t * noise

		if self.parameterization.lower() in self.x_prev_param_name_list:
			target = self.q_posterior_mean_variance(x_start=images, x_t=noised_image, t=t)[0]
		elif self.parameterization.lower() in self.x0_param_name_list:
			target = images
		elif self.parameterization.lower() in self.v_param_name_list:
			target = self.get_v(x=images, noise=noise, t=t)
		else:
			target = noise

		with tf.GradientTape() as tape:
			if self.flag_learn_var:
				prediction, logvar = tf.split(self.call({'x': noised_image, 'time': t, 'condition': condition},
				                                        training=True),
				                              num_or_size_splits=2, axis=-1)
				frozen_prediction = tf.stop_gradient(prediction)
				frozen_output = tf.concat([frozen_prediction, logvar], axis=-1)
				vlb = self.lambda_vlb * self._vb_terms_bpd(model_output=frozen_output,
				                                           x_start=images, x_t=noised_image, t=t)
			else:
				prediction = self.call({'x': noised_image, 'time': t, 'condition': condition}, training=True)
				vlb = 0.
			loss, noise_loss = self.compute_loss(y=target, y_pred=prediction)
			loss = loss + vlb

		gradients = tape.gradient(loss, self.trainable_variables)
		if debug:
			grad_none = len([g for g in gradients if g is None])
			grad_abs = [tf.reduce_mean(tf.abs(g))  # mean |grad|
						for g in gradients if g is not None]
			grad_max = [tf.reduce_max(tf.abs(g))  # max |grad|
						for g in gradients if g is not None]

			tf.print("Step", self.optimizer.iterations, " average timestep", tf.reduce_mean(t),
					 "\nloss", tf.reduce_mean(loss), "loss_vlb", tf.reduce_mean(vlb),
					 "\ngrad|mean|", tf.reduce_mean(grad_abs),
					 "grad|max|", tf.reduce_max(grad_max),
					 "\nlen(grad == None)", grad_none)

		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		self.loss_tracker.update_state(loss)
		self.noise_loss_tracker.update_state(noise_loss)
		self.lambda_vlb_loss_tracker.update_state(vlb)

		return {m.name: m.result() for m in self.metrics}

	# @tf.function
	def test_step(self, data):
		""" Override default validation step in model.evaluate() for the diffusion process.
		A loss needs to be provided when compiling the model (e.g., l2 norm)."""
		images = data[0]
		if not isinstance(data, tuple):
			condition = None
		else:
			condition = data[1]
		images_shape = tf.shape(images)
		batch_size = tf.shape(images)[0]

		t = tf.random.uniform(shape=[batch_size],
		                      minval=0, maxval=self.timesteps,
		                      dtype=tf.int32)
		noise = tf.random.normal(shape=images_shape, dtype=TF_DTYPE)

		# Retrieve the current timestep for each batch, and reshape for broadcast
		sqrt_alpha_bar_t = tf.reshape(tf.gather(self.sqrt_alpha_bar, t, axis=0), self.reshape_dim)
		sqrt_one_minus_alpha_bar_t = tf.reshape(tf.gather(self.sqrt_one_minus_alpha_bar, t, axis=0), self.reshape_dim)
		noised_image = sqrt_alpha_bar_t * images + sqrt_one_minus_alpha_bar_t * noise

		if self.parameterization.lower() in self.x_prev_param_name_list:
			target = self.q_posterior_mean_variance(x_start=images, x_t=noised_image, t=t)[0]
		elif self.parameterization.lower() in self.x0_param_name_list:
			target = images
		elif self.parameterization.lower() in self.v_param_name_list:
			target = self.get_v(x=images, noise=noise, t=t)
		else:
			target = noise

		if self.flag_learn_var:
			prediction, logvar = tf.split(self.call({'x': noised_image, 'time': t, 'condition': condition}, training=False),
			                              num_or_size_splits=2, axis=-1)
			frozen_prediction = tf.stop_gradient(prediction)
			frozen_output = tf.concat([frozen_prediction, logvar], axis=-1)
			vlb = self.lambda_vlb * self._vb_terms_bpd(model_output=frozen_output,
			                         x_start=images, x_t=noised_image, t=t)
		else:
			prediction = self.call({'x': noised_image, 'time': t, 'condition': condition}, training=False)
			vlb = 0.
		loss, noise_loss = self.compute_loss(y=target, y_pred=prediction)
		loss = loss + vlb

		self.loss_tracker.update_state(loss)
		self.noise_loss_tracker.update_state(noise_loss)
		self.lambda_vlb_loss_tracker.update_state(vlb)

		return {m.name: m.result() for m in self.metrics}

	# @tf.function
	def ddpm(self, x_t, time, condition=None):
		"""Predicts x^{t-1} based on x^{t} using the DDPM model.
		Use in a for loop from T to 1 to retrieve input from noise."""

		pred_noise = self.call({'x': x_t, 'time': time, 'condition': condition}, training=False)
		z = tf.random.normal(shape=tf.shape(x_t), dtype=x_t.dtype)
		out = self.p_mean_variance(model_output=pred_noise, x=x_t, t=time)
		nonzero_mask = tf.cast(tf.where(time == 0, 0., 1.), TF_DTYPE)  # no noise when t == 0
		nonzero_mask = tf.reshape(nonzero_mask, self.reshape_dim)
		var = nonzero_mask * tf.exp(0.5 * out["log_variance"]) * z
		var_tilde = nonzero_mask * tf.exp(0.5 * out["log_variance_tilde"]) * z

		return out['mean'], var, var_tilde

	@tf.function
	def tfunc_ddpm(self, x_t, time, condition=None):
		"""ddpm method wrapped in tf.function for improved speed."""
		return self.ddpm(x_t, time, condition)

	def ddpm_loop(self, x_T, condition, num_timesteps=None,
				  sub_sequence_type='linear', flag_var_tilde=True,
				  keep_all_xt=False):
		"""
		Predicts x_{0} based on x^{T} using the DDPM model.
		If ``num_timesteps < T`` is provided, predicts posterior samples with
		a subsequence of num_timesteps elements instead of T elements. The way
		the subsequence is formed depends on ``sub_sequence_type`` (``linear``
		or ``quadratic``).
		"""
		if num_timesteps in (None, 0, self.timesteps):
			indices = list(range(self.timesteps))[::-1]
		else:
			if sub_sequence_type in 'linear':
				indices = np.linspace(0, self.timesteps - 1, num=num_timesteps,
				                      dtype=np.int32)[::-1].tolist()
			elif sub_sequence_type in 'quadratic':
				indices = (np.linspace(0, np.sqrt(self.timesteps - 1), num=num_timesteps,
				                       dtype=np.int32)[::-1] ** 2).tolist()
			else:
				raise ValueError('Subsequence type not recognized '
				                 '(given {})'.format(sub_sequence_type))

		# Shape checks
		x_out = x_T
		x_all_out = []
		batch_size = tf.shape(x_out)[0]
		if condition is not None:
			if tf.shape(condition)[0] != batch_size:
				condition = tf.repeat(condition, [batch_size, ], axis=0)

		for i in tqdm.tqdm(indices):
			t = tf.repeat(tf.reshape(i, shape=(-1,)),
			              [batch_size,], axis=0)
			mean, var, var_tilde = self.tfunc_ddpm(x_out, time=t, condition=condition)
			if flag_var_tilde:
				x_out = mean + var_tilde
			else:
				x_out = mean + var
			if keep_all_xt:
				x_all_out.append(x_out)

		if keep_all_xt:
			x_out = np.stack(x_all_out, axis=0)

		return x_out


	@tf.function
	def tfunc_ddpm_loop(self, x_T, condition):
		"""
		``ddpm_loop`` wrapped in a tf_function for improved speed
		 (at the cost of flexibility).
		  Predicts x^{0} based on x^{T} using the DDPM model.
		  Predicts posterior samples with T timesteps (T taken from model
		  attributes self.timesteps).
		  Output defaults the posterior calculated using beta_tilde.
		  For more flexibility, use ``ddpm_loop``
		"""
		indices = list(range(self.timesteps))[::-1]
		batch_size = tf.shape(x_T)[0]

		for i in tqdm.tqdm(indices):
			t = tf.repeat(tf.reshape(i, shape=(-1,)),
			              [batch_size, ], axis=0)
			mean, _, var_tilde = self.tfunc_ddpm(x_T, time=t, condition=condition)
			x_T = mean + var_tilde
		return x_T



	def get_config(self):
		"""Serialize object"""
		config = super().get_config()
		config.update({
			'lambda_vlb':     self.lambda_vlb,
			'parameterization': self.parameterization,
		})
		return config


