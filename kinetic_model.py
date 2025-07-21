
# %% Imports

from collections import OrderedDict as _odict
import scipy.ndimage as _spnd
import numpy as np


# %% Vectorized 1D interpolation


def estimate_continuous_convolution(x, y0, y1, num_points_resample=None):
	if num_points_resample is None:
		num_points = 2 * np.unique(x).size
	else:
		num_points = num_points_resample
	x_rs = np.linspace(np.min(x), np.max(x), num_points)
	delta_x = x_rs[1] - x_rs[0]

	# Resampling on common regular grid
	y0_rs = np.interp(x_rs, x, y0)
	y1_rs = interp1d_linear_vec(x_rs, x, y1)

	# Compute convolution
	if y1.ndim == 1:
		discrete_conv = np.convolve(y0_rs, y1_rs)[:x_rs.size] * delta_x
	else:
		discrete_conv = _spnd.convolve1d(y1_rs, y0_rs, axis=0, mode='constant',
		                                 origin=-x_rs.size // 2) * delta_x

	# Resample on original grid
	return interp1d_linear_vec(x, x_rs, discrete_conv)


def interp1d_linear_vec(x, xp, fp, dim=0):
	if fp.ndim == 1:
		fp_in = fp.reshape([-1, 1])
	else:
		fp_in = fp

	# Interpolation weights
	distances = np.abs(xp[np.newaxis, :] -
	                   x[:, np.newaxis]).astype(np.float64)
	x_indices = np.searchsorted(xp, x)
	weights = np.zeros_like(distances)
	idx = np.arange(len(x_indices))
	weights[idx, x_indices] = distances[idx, x_indices - 1]
	weights[idx, x_indices - 1] = distances[idx, x_indices]
	weights /= np.sum(weights, axis=1)[:, np.newaxis]

	output_shape = [x.size, ] + list(fp.shape[1:])
	output = np.tensordot(weights, fp_in, axes=[[1, ], [dim, ]])
	if fp.ndim == 1 or dim == 0:
		return output.reshape(output_shape)

	# Reorder axes
	return np.moveaxis(output, 0, dim)


# %% SRTM

class SRTM:

	def __init__(self, frame_time_list, frame_duration_list):

		self._frame_time_list = frame_time_list
		self._frame_duration_list = frame_duration_list

	def forward_model(self, DVR=None, k2=None, R1=None, tac_ref=None):
		# k2 = 0.03
		# R1 = 0.51
		bp = DVR - 1
		self._tac_reference = tac_ref
		time_vector = self._frame_time_list
		c_r = self._tac_reference
		if not np.isscalar(bp):
			c_r_v = c_r.reshape([-1, ] + [1, ] * bp.ndim)
		else:
			c_r_v = c_r
		k2a = k2 / DVR
		# Exponential
		c_exp = SRTM.make_time_exponential(-k2a, time_vector)
		return R1 * c_r_v + (k2 - R1 * k2a) * \
			SRTM.convolve(time_vector, c_r, c_exp)

	# Allow calling forward model directly via object
	__call__ = forward_model

	@staticmethod
	def make_time_func(param, time_vector, func, time_scale=None,
	                   space_scale=None):
		r"""Create time-space map"""
		if np.isscalar(param):
			output = func(param, time_vector)
			if time_scale is not None:
				output *= time_scale
			if space_scale is not None:
				output *= space_scale
			return output

		time_reshape = [-1, ] + [1, ] * param.ndim
		time_vector_r = time_vector.reshape(time_reshape)
		output = func(param.reshape([1, ] + list(param.shape)),
		              time_vector_r)
		if time_scale is not None:
			if time_scale.ndim == 1:
				output *= time_scale.reshape(time_reshape)
			else:
				output *= time_scale
		if space_scale is not None:
			if space_scale.ndim == output.ndim:
				output *= space_scale
			else:
				output *= space_scale[np.newaxis]
		return output

	@staticmethod
	def make_time_exponential(param, time_vector, time_scale=None, **kwargs):

		return SRTM.make_time_func(
			param, time_vector, lambda x, t: np.exp(x * t),
			time_scale=time_scale, **kwargs)

	@staticmethod
	def convolve(time_vector, c_0, c_1):
		"""Convolution"""
		return estimate_continuous_convolution(time_vector, c_0, c_1)






class SRTM2:

	def __init__(self, frame_time_list, frame_duration_list, tac_reference):

		self._frame_time_list = frame_time_list
		self.frame_duration_list = frame_duration_list
		self._tac_reference = tac_reference

	def create_activity_curve(self, DVR=None, R1=None, k2p=None):
		# k2 = 0.03
		# R1 = 0.51
		bp = DVR - 1

		time_vector = self._frame_time_list
		c_r = self._tac_reference
		if not np.isscalar(bp):
			c_r_v = c_r.reshape([-1, ] + [1, ] * bp.ndim)
		else:
			c_r_v = c_r
		k2 = k2p * R1
		k2a = k2 / DVR
		# Exponential
		c_exp = SRTM.make_time_exponential(-k2a, time_vector)
		return R1 * c_r_v + (k2 - R1 * k2a) * \
			SRTM.convolve(time_vector, c_r, c_exp)

	# Allow calling forward model directly via object
	__call__ = create_activity_curve

	@staticmethod
	def make_time_func(param, time_vector, func, time_scale=None,
	                   space_scale=None):
		r"""Create time-space map"""
		if np.isscalar(param):
			output = func(param, time_vector)
			if time_scale is not None:
				output *= time_scale
			if space_scale is not None:
				output *= space_scale
			return output

		time_reshape = [-1, ] + [1, ] * param.ndim
		time_vector_r = time_vector.reshape(time_reshape)
		output = func(param.reshape([1, ] + list(param.shape)),
		              time_vector_r)
		if time_scale is not None:
			if time_scale.ndim == 1:
				output *= time_scale.reshape(time_reshape)
			else:
				output *= time_scale
		if space_scale is not None:
			if space_scale.ndim == output.ndim:
				output *= space_scale
			else:
				output *= space_scale[np.newaxis]
		return output

	@staticmethod
	def make_time_exponential(param, time_vector, time_scale=None, **kwargs):

		return SRTM.make_time_func(
			param, time_vector, lambda x, t: np.exp(x * t),
			time_scale=time_scale, **kwargs)

	@staticmethod
	def convolve(time_vector, c_0, c_1):
		"""Convolution"""
		return estimate_continuous_convolution(time_vector, c_0, c_1)
