
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
import tensorflow.keras.layers as nn
from functools import partial
import copy
import numpy as np
import helper_func as hlp
from diffusion_model import DTYPE, TF_DTYPE, NP_DTYPE

tf.keras.backend.set_floatx(DTYPE)

np.random.seed(1234)
tf.random.set_seed(1234)
tf.keras.utils.set_random_seed(1234)

# Clear previously registered custom objects from this module
del_key = []
for key in tf.keras.utils.get_custom_objects().keys():
    if 'CustomLayers' in key:
        del_key.append(key)
for del_key_i in del_key:
    tf.keras.utils.get_custom_objects().pop(del_key_i)

## Basic Layers and Blocks

def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + tf.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * tf.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + tf.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, means, log_scales, bin_width=1./255.):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1], unless bin_width is given.
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :param bin_width: (optional) bin width for distribution discretization, if not 1./255.
    :return: a tensor like x of log probabilities (in nats).
    """

    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + bin_width)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - bin_width)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = tf.math.log(tf.clip_by_value(cdf_plus, clip_value_min=1e-12,
                                                clip_value_max=TF_DTYPE.max))
    log_one_minus_cdf_min = tf.math.log(
        tf.clip_by_value(1.0 - cdf_min, clip_value_min=1e-12,
                         clip_value_max=TF_DTYPE.max))

    cdf_delta = cdf_plus - cdf_min
    log_probs = tf.where(
        x < -0.999,
        log_cdf_plus,
        tf.where(x > 0.999, log_one_minus_cdf_min,
                 tf.math.log(tf.clip_by_value(cdf_delta, clip_value_min=1e-12,
                                              clip_value_max=TF_DTYPE.max))),
    )
    # assert tf.shape(log_probs) == tf.shape(x)
    return log_probs




# Kernel initializer to use
def kernel_init(scale, seed=1234):
    scale = max(scale, 1e-10)
    return tf.keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform", seed=seed
    )


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class SinusoidalPosEmb(Layer):
    def __init__(self, dim, max_positions=10000.):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        self.max_positions = max_positions

    def call(self, x, training=True):
        x = tf.cast(x, TF_DTYPE)
        half_dim = self.dim // 2
        emb = tf.cast(tf.math.log(self.max_positions), dtype=TF_DTYPE) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=TF_DTYPE) * -emb)
        emb = x[:, None] * emb[None, :]

        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)

        return emb

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim":           self.dim,
            "max_positions": self.max_positions,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class DeserializeActivation(Layer):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation or 'linear'
        self.activation_func = tf.keras.activations.deserialize(self.activation)

    def call(self, x, training=True):
        return self.activation_func(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "activation": self.activation,
        })
        return config

@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class SiLU(Layer):
    def __init__(self):
        super(SiLU, self).__init__()

    def call(self, x, training=True):
        return x * tf.nn.sigmoid(x)


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
def gelu_func(x, approximate=False):
    if approximate:
        coeff = tf.cast(0.044715, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class GELU(Layer):
    def __init__(self, approximate=False):
        super(GELU, self).__init__()
        self.approximate = approximate

    def call(self, x, training=True):
        return gelu_func(x, self.approximate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "approximate": self.approximate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class LayerNorm(Layer):
    def __init__(self, dim, eps=1e-5, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.eps = eps

        self.g = tf.Variable(tf.ones([1, 1, 1, dim], dtype=tf.float64))
        self.b = tf.Variable(tf.zeros([1, 1, 1, dim], dtype=tf.float64))

    def call(self, x, training=True):
        var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)

        x = (x - mean) / tf.sqrt((var + self.eps)) * self.g + self.b
        return x


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class NormalizationLayer(Layer):
    """
    Pre-processing layer that normalizes or standardizes the input.
    Pass named arguments ``min`` and ``max`` to scale the input from 0 to 1 (normalize).
    Pass named arguments ``mean`` and ``var`` (or ``std``) to scale the distribution
    of the input data to a normal gaussian (standardize).
    flag ``invert=True`` indicates that the layer should undo the normalization.
    """
    def __init__(self, min=None, max=None, mean=None, var=None,
                 std=None, invert=False, **kwargs):
        super().__init__(**kwargs)
        self.min = min
        self.max = max
        self.mean = mean
        self.var = var
        self.std = std
        self.invert = invert

        if (min is not None) and (max is not None):
            self.var_sub = min
            self.var_div = (max - min)
            self.mode = 'minmax'
        elif mean is not None:
            self.var_sub = mean
            self.var_div = np.sqrt(var, dtype=NP_DTYPE) if var is not None else std
            self.mode = 'std'
        else:
            raise ValueError('``min`` and ``max`` or ``mean`` and '
                             '``var`` (or ``std``) must be passed as '
                             'input arguments')

        # Make sure var_div is not too close to zero
        self.var_div = tf.clip_by_value(self.var_div, 1e-6, np.inf)

    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            raise NotImplementedError('')
        if not self.invert:
            outputs = (inputs - self.var_sub) / self.var_div
        else:
            outputs = inputs * self.var_div + self.var_sub

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'min':    self.min,
            'max':    self.max,
            'mean':   self.mean,
            'var':    self.var,
            'std':    self.std,
            'invert': self.invert,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class Identity(Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def call(self, x, training=True):
        return tf.identity(x)


class ConcatOutputFromLayer(Layer):
    def __init__(self, layer, axis=-1, **kwargs):
        super(ConcatOutputFromLayer, self).__init__()
        self.axis = axis
        self.layer = layer
        self.concat_layer = nn.Concatenate(axis, **kwargs)

    def call(self, x, training=True):
        return self.concat_layer(self.layer(x))

    def get_config(self):
        """Serialize object"""
        config = super().get_config()
        config.update({
            'axis':           self.axis,
            'layer':        tf.keras.layers.serialize(self.layer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create object from configuration"""
        config["layer"] = tf.keras.layers.deserialize(config["layer"])
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class MLP(Layer):
    def __init__(self, hidden_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.net = Sequential([
            nn.Flatten(),
            nn.Dense(units=hidden_dim),
            GELU(),
            LayerNorm(hidden_dim),
            nn.Dense(units=hidden_dim),
            GELU(),
            LayerNorm(hidden_dim),
            nn.Dense(units=hidden_dim),
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)



## Attention layer

class Attention1D(Layer):
    """Applies attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=0, num_heads=1, layer_type='dense', seed=1234, **kwargs):
        self.units = units
        self.groups = groups
        self.num_heads = num_heads
        self.layer_type = layer_type
        self.seed = seed
        super().__init__(**kwargs)

        self.group_norm = Identity()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=self.num_heads, **kwargs)
        self.add = tf.keras.layers.Add()

        if isinstance(self.groups, int):
            if self.groups > 0:
                self.group_norm = nn.GroupNormalization(groups=groups)

    def build(self, input_shape):
        super().build(input_shape)
        if 'dense' in self.layer_type.lower():
            self.query = nn.Dense(self.units, kernel_initializer=kernel_init(1.0, seed=self.seed))
            self.key = nn.Dense(self.units, kernel_initializer=kernel_init(1.0, seed=self.seed))
            self.value = nn.Dense(self.units, kernel_initializer=kernel_init(1.0, seed=self.seed))
            self.proj = None
        elif 'conv' in self.layer_type.lower():
            self.query = nn.Conv1D(filters=self.units * self.num_heads, kernel_size=1, use_bias=False)
            self.key = nn.Conv1D(filters=self.units * self.num_heads * 2, kernel_size=1, use_bias=False)
            self.proj = nn.Conv1D(filters=self.units * self.num_heads, kernel_size=1)
        else:
            raise ValueError('Unsupported ``layer_type`` argument')

    def call(self, input1, condition=None, input1_data_format='channels_first',
             condition_data_format='channels_first', training=False):
        if condition is None:
            condition = input1

        input1 = self.group_norm(input1)
        condition = self.group_norm(condition)
        q = self.query(input1)
        if 'dense' in self.layer_type.lower():
            k = self.key(condition)
            v = self.value(condition)
        elif 'conv' in self.layer_type.lower():
            k, v = tf.split(self.key(condition), num_or_size_splits=2, axis=-1)
        else:
            raise ValueError('Unsupported ``layer_type`` argument')

        attn_output, attn_scores = self.mha(
            query=q, value=v, key=k, training=training,
            return_attention_scores=True)

        x = self.add([q, attn_output])
        # x = self.group_norm(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
        'units': self.units,
        'groups': self.groups,
        'num_heads': self.num_heads,
        'layer_type': self.layer_type,
        'seed': self.seed,
        })
        return config



## Advanced Layers and Blocks


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class Encoder_v3(Layer):
    """Maps Inputs to latent variables z."""
    def __init__(self, enc_size, latent_dim=16,
                 activation='relu', l2_reg=None, l1_reg=None,
                 flag_skip=False, final_activation=None,
                 concat_output=True, layer_type='dense', **kwargs):
        name = kwargs.pop("name", None)
        init_kwargs = {'name': name} if name is not None else {}
        super().__init__(**init_kwargs)

        self.enc_size = enc_size
        self.n_layers = len(enc_size)
        self.activation = activation
        self.final_activation = final_activation
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.layer_type = layer_type
        if 'dense' in self.layer_type:
            self.f_layer = nn.Dense
        elif 'conv' in self.layer_type:
            self.f_layer = partial(nn.Conv1D, kernel_size=1, strides=1,
                                   padding='same')
        else:
            raise NotImplementedError("Layer type {} not implemented.".format(layer_type))

        if isinstance(latent_dim, int):
            self.latent_dim = (latent_dim,)
        elif isinstance(latent_dim, (tuple, list)):
            self.latent_dim = latent_dim
        else:
            raise ValueError('Input argument ``latent_dim`` must be an integer, or a'
                             ' tuple.')
        self.n_latent = len(self.latent_dim)
        self.flag_skip = flag_skip
        self.concat_output = concat_output
        self.encoder = {}
        self.dense_z = {}
        self.kwargs = {**kwargs}
        if (l2_reg is not None) and (l1_reg is not None):
            raise ValueError('keywords for kernel regularization are either ``l2_reg``,'
                             'or ``l1_reg``. Both cannot be given.')
        if l2_reg is not None:
            self.kernel_reg = tf.keras.regularizers.L2(self.l2_reg)
        elif l1_reg is not None:
            self.kernel_reg = tf.keras.regularizers.L1(self.l1_reg)
        else:
            self.kernel_reg = None


    def build(self, input_shape):

        for ee in range(self.n_layers):
            self.encoder['layer_' + format(ee)] = self.f_layer(self.enc_size[ee],
                                                           activation=self.activation,
                                                           name='hidden_encoder_layer_{}'.format(ee),
                                                           kernel_regularizer=self.kernel_reg, **self.kwargs)
        for ee in range(self.n_latent):
            self.dense_z['layer_' + format(ee)] = self.f_layer(self.latent_dim[ee],
                                                           kernel_regularizer=self.kernel_reg,
                                                           activation=self.final_activation,
                                                           name='dense_z', **self.kwargs)

    def call(self, inputs, training=False):
        x = inputs
        h = []
        for key_i, layer_i in self.encoder.items():
            x = layer_i(x, training=training)
            h.append(x)

        z = []
        for key_i, layer_i in self.dense_z.items():
            z.append(layer_i(x, training=training))

        if self.concat_output:
            z = tf.concat(z, axis=-1)

        return z, h

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            'latent_dim': self.latent_dim,
            'enc_size':   self.enc_size,
            'activation': self.activation,
            'final_activation': self.final_activation,
            'flag_skip':  self.flag_skip,
            'concat_output': self.concat_output,
            'l2_reg':   self.l2_reg,
            'l1_reg':   self.l1_reg,
            'layer_type': self.layer_type,
        })
        return {**base_config}

@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class Encoder_v3_noskip(Encoder_v3):
    """Maps Inputs to latent variables z."""

    def call(self, inputs, training=False):
        x = inputs
        for key_i, layer_i in self.encoder.items():
            x = layer_i(x, training=training)

        z = []
        for key_i, layer_i in self.dense_z.items():
            z.append(layer_i(x, training=training))

        if self.concat_output:
            z = tf.concat(z, axis=-1)

        return z


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class ConvBlock(Layer):
    """Convolution block"""

    def __init__(self, ndim, units, num_convs=1, kernel_size=3, conv_args=None,
                 norm_list=None, norm_type='layer', activation='relu',
                 l2_reg=None, flag_res=True, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        ndim : int
            Input dimensions (excluding batch and channel dimensions).
        units : int
            Number of convolution units.
        num_conv_per_block : int
            Number of consecutive convolution layers.
        kernel_size : int or list(int)
            Convolution kernel size.
        conv_args : dict
            Additional parameters passed to convolutional layers.
        norm_list : str
            Batch normalization (disabled if None, ``pre`` or ``post``
            activation).
        norm_type : str
            Normalization type (``batch`` or ``layer``).
        activation : str
            Activation function.
        l2_reg : float
            Kernel regularization.
        flag_res : bool
            When True, use ResNet-like skip connection (additive) around each
            block.
        """
        # Attributes
        self.units = units
        self.ndim = ndim
        self.num_convs = num_convs
        self.conv_args = conv_args or {}
        if norm_list is None:
            self.norm_list = []
        elif isinstance(norm_list, str):
            self.norm_list = [s.strip() for s in norm_list.split(',')]
        elif isinstance(norm_list, (list, tuple)):
            self.norm_list = norm_list
        self.norm_type = norm_type
        self.activation = activation
        self.flag_res = flag_res

        if isinstance(kernel_size, (list, tuple)):
            kernel_size_d = kernel_size
        else:
            kernel_size_d = (kernel_size,) * ndim
        self.kernel_size = kernel_size_d
        self.l2_reg = l2_reg

        # Call parent constructor
        super(ConvBlock, self).__init__()

    def build(self, input_shape):
        """Create layers for convolution block"""
        super().build(input_shape)
        # Helper functions
        f_conv = getattr(nn, 'Conv{}D'.format(self.ndim))
        if self.norm_type == 'batch':
            f_norm = nn.BatchNormalization
        elif self.norm_type == 'layer':
            f_norm = nn.LayerNormalization
        else:
            raise NotImplementedError('``norm_type`` input argument not recognized '
                                      '(given {})'.format(self.norm_type))
        # Weight regularizer
        conv_args = self.conv_args.copy()
        if self.l2_reg is not None:
            conv_args['kernel_regularizer'] = tf.keras.regularizers.L2(
                self.l2_reg)
        # Define layers
        self.conv_l = [f_conv(self.units, kernel_size=self.kernel_size,
                              **conv_args)
                       for c in range(self.num_convs)]
        if self.flag_res:
            self.res_conv = f_conv(self.units, kernel_size=1, **conv_args)
            self.res_add = nn.Add()
        self.activation_l = nn.Activation(self.activation)
        # Normalization
        if 'pre' in self.norm_list:
            self.norm_pre_l = [f_norm() for c in range(self.num_convs)]
        if 'post' in self.norm_list:
            self.norm_post_l = [f_norm() for c in range(self.num_convs)]

    def call(self, x, training=False):
        """Layer call"""
        x_s = x
        for fi in range(self.num_convs):
            x = self.conv_l[fi](x, training=training)
            if 'pre' in self.norm_list:
                x = self.norm_pre_l[fi](x, training=training)
            if self.flag_res and fi == self.num_convs - 1:
                x = self.res_add([x, self.res_conv(x_s)])
            x = self.activation_l(x)
            if 'post' in self.norm_list:
                x = self.norm_post_l[fi](x, training=training)
        return x

    def get_config(self):
        """Serialize object"""
        base_config = super().get_config()
        config = {'units':       self.units,
                  'num_convs':   self.num_convs,
                  'kernel_size': self.kernel_size,
                  'conv_args':   self.conv_args,
                  'ndim':        self.ndim,
                  'norm_list':   self.norm_list,
                  'norm_type':   self.norm_type,
                  'activation':  self.activation,
                  'l2_reg':      self.l2_reg,
                  'flag_res':    self.flag_res}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """Create object from configuration"""
        return cls(**config)



## Complex Blocks and Layers using custom layers


@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class UnetConditional(Layer):
    """Conditional U-Net implemented as layer for an end-to-end training."""

    def __init__(self, num_filt_start=64,
                 pool_size=2, depth=5, block_type='conv', block_params=None,
                 skip_conn_type='concat', skip_conn_op=None,
                 skip_conn_post_op=None, dropout=None,
                 final_activation=None, cond_params=None,
                 resize_input_len=None, resize_input_type='dense',
                 constrained_func_output=None,
                 sin_emb_dim=None, normalize_feature_dict=None,
                 normalize_label_dict=None, learn_variance=None,
                 pool_type='max', **kwargs):
        super(UnetConditional, self).__init__()

        self.num_filt_start = num_filt_start
        self.pool_size = pool_size
        self.depth = depth
        self.block_type = block_type
        self.block_params = block_params
        self.skip_conn_type = skip_conn_type
        self.skip_conn_op = skip_conn_op
        self.skip_conn_post_op = skip_conn_post_op
        self.dropout = dropout
        self.final_activation = final_activation
        self.sin_emb_dim = sin_emb_dim
        self.learn_variance = learn_variance or ''
        self.flag_learn_var = 'learn' in self.learn_variance.lower()
        self.resize_input_len = resize_input_len
        self.resize_input_type = resize_input_type
        self.resize_input_layer = None
        self.constrained_func_output = constrained_func_output
        self.pool_type = pool_type


        self.sinusoidal_cond_mlp = False if sin_emb_dim in (None, 0) else True
        self.cond_params = cond_params if isinstance(cond_params,dict) \
            else {'network_name': 'encoder',
                                   'flag_flatten_input': True,
                                   'network_kwargs': {'enc_size': [256, 128, 64],
                                                      'latent_dim': 32}
                                           }
        self.encoder_cond = None
        self.normalize_feature_dict = normalize_feature_dict or {}
        self.normalize_label_dict = normalize_label_dict or {}
        # self.n_condition = n_condition  # number of conditions to embed (e.g., time and digit)

        self.conv_args = {'padding': 'same'}
        self.ndim = None
        self.im_size = None


        # Create layers
        self.downs = []
        self.ups = []

        # Conditional embedding
        self.cond_mlp_down = []
        self.cond_mlp_up = []
        self.normalize_layer = {}
        self.denormalize_layer = {}

    def build(self, input_shape):
        super().build(input_shape)

        # Build Normalization layers (create Normalization layers if normalize_*_dict is not None)
        for name_i, normalize_dict in zip(['feature', 'label'],
                                          [self.normalize_feature_dict,
                                           self.normalize_label_dict]):
            lower_keys = list(map(str.lower, normalize_dict.keys()))
            if len(normalize_dict) == 0:
                self.normalize_layer[name_i] = None
                self.denormalize_layer[name_i] = None
            else:
                if ('mean' in lower_keys) and ('var' in lower_keys or 'std' in lower_keys):
                    normalize_kwargs = {
                        'mean': np.array(normalize_dict.get('mean'), dtype=NP_DTYPE),
                        'var': np.array(normalize_dict.get('var', None), dtype=NP_DTYPE) if 'var' in lower_keys \
                        else (np.array(normalize_dict.get('std', None), dtype=NP_DTYPE) ** 2)
                    }
                elif ('min' in lower_keys) and ('max' in lower_keys):
                    normalize_kwargs = {
                        'min': np.array(normalize_dict.get('min'), dtype=NP_DTYPE),
                        'max': np.array(normalize_dict.get('max'), dtype=NP_DTYPE)
                    }
                else:
                    raise ValueError('Normalization method for {} not recognized. Normalization dict '
                                     'should have named variables ``min`` and ``max`` or ``mean`` and '
                                     '``var`` (or ``std``). Given {}'.format(name_i, normalize_dict.keys()))

                # Create a Normalization layer and set its internal state using passed mean and variance
                self.normalize_layer[name_i] = NormalizationLayer(**normalize_kwargs)
                self.denormalize_layer[name_i] = NormalizationLayer(invert=True, **normalize_kwargs)

        # Get shapes
        ndim = len(input_shape) - 2
        self._input_shape = input_shape
        self.ndim = ndim
        self.num_channels_out = 2 * input_shape[-1] if self.flag_learn_var else input_shape[-1]
        self.num_channels_in = input_shape[-1]

        # Create layers
        if self.resize_input_len is not None:
            if 'dense' in self.resize_input_type:
                self.resize_input_layer = [tf.keras.layers.Dense(self.resize_input_len, activation=None),
                                           tf.keras.layers.Dense(np.prod(input_shape[1:-1]), activation=None)]
            else:
                raise NotImplementedError('resize type not recognized (given {}, expected str ``dense``).'
                                          ' Only resizing through dense layer available '
                                          'at the moment.'.format(self.resize_input_type))

        if ndim == 1:
            im_size = (input_shape[1],) if self.resize_input_layer is None else (self.resize_input_len,)
        elif ndim == 2:
            im_size = (input_shape[1], input_shape[2])
        elif ndim == 3:
            im_size = (input_shape[1], input_shape[2], input_shape[3])
        else:
            raise NotImplementedError(
                'Cannot process data with ndim > 3 (ndim = {})'.format(ndim))
        self.im_size = [im_size, ]
        if isinstance(self.pool_size, (list, tuple)):
            kernel_size_up = self.pool_size
        else:
            kernel_size_up = (self.pool_size,) * ndim
        if len(kernel_size_up) == 1:
            kernel_size_up = kernel_size_up[0]

        f_conv = getattr(nn, 'Conv{}D'.format(ndim))
        if ('max') in self.pool_type.lower():
            f_maxpool = getattr(nn, 'MaxPooling{}D'.format(ndim))
        elif ('av') in self.pool_type.lower():
            f_maxpool = getattr(nn, 'AveragePooling{}D'.format(self.ndim))
        f_upsamp = getattr(nn, 'UpSampling{}D'.format(ndim))

        cond_emb_layer = [[nn.Dense(units=tf.reduce_prod(im_size)), GELU(),]]
        if self.encoder_cond is None:
            if isinstance(self.cond_params.get('load_model', None), str):
                autoencoder_load = tf.keras.models.load_model(self.cond_params['load_model'])
                encoder_load = hlp.rgetattr(autoencoder_load, 'network.' +
                                            self.cond_params['load_network_name'])
                encoder_load.trainable = self.cond_params.get('trainable', False)
                encoder_load = ConcatOutputFromLayer(layer=encoder_load, axis=-1)
                self.encoder_cond = [encoder_load,]

            elif self.cond_params['network_name'].lower() in 'encoder':
                self.encoder_cond = [Encoder_v3_noskip(**self.cond_params['network_kwargs']),]

            elif self.cond_params['network_name'].lower() in 'attention':
                self.encoder_cond = [Attention1D(**self.cond_params['network_kwargs']), ]

            elif self.cond_params['network_name'].lower() in 'cnn':
                cond_params_copy = copy.deepcopy(self.cond_params)
                # ndim_cond = cond_params_copy['network_kwargs'].pop('ndim', self.ndim)
                depth_cond = cond_params_copy['network_kwargs'].get('depth')
                num_units_cond_start = cond_params_copy['network_kwargs'].get('num_filt_start')
                data_format = {'data_format': cond_params_copy['network_kwargs'].
                                   get('conv_args', {}).get('data_format', 'channels_first')}
                data_format.update({'padding': cond_params_copy['network_kwargs'].
                                   get('conv_args', {}).get('padding', 'valid')})
                cond_pool_size = cond_params_copy['network_kwargs'].get('pool_size', 1)

                tmp_layer = []
                for di_cond in range(depth_cond):
                    num_units_cond = num_units_cond_start * 2 ** di_cond
                    tmp_layer.append(ConvBlock(units=num_units_cond,  # ndim=ndim_cond,
                                               **cond_params_copy['network_kwargs']))
                    if di_cond < depth_cond - 1:
                        if cond_pool_size > 1:
                            tmp_layer.append(f_maxpool(pool_size=cond_pool_size,
                                                       **data_format))
                        else:
                            tmp_layer.append(Identity())
                self.encoder_cond = tmp_layer

            elif self.cond_params['network_name'].lower() in 'mlp':
                self.encoder_cond = [nn.Dense(units=tf.reduce_prod(im_size)), tf.nn.relu()]
            elif self.cond_params['network_name'].lower() in 'identity':
                self.encoder_cond = [Identity(),]
            elif self.cond_params['network_name'].lower() not in ('encoder', 'mlp'):
                raise NotImplementedError('Network name for condition embedding not recognized. '
                                          'Expected ``encoder`` or ``mlp`` (given {})'.
                                          format(self.cond_params['network_name']))

        cond_emb_layer.append(self.encoder_cond)

        # Downsampling Block
        # conv_out = []
        for di in range(self.depth):
            # Condition embedding (time and label)
            tmp_cond_layer = []
            for cond_i in range(2):
                if self.sinusoidal_cond_mlp and cond_i == 0:
                    sin_emb = SinusoidalPosEmb(self.sin_emb_dim)
                else:
                    sin_emb = Identity()
                tmp_cond_layer.append(Sequential([
                    sin_emb, nn.Flatten() if self.cond_params.get('flag_flatten_input', True)
                    else Identity()] + cond_emb_layer[cond_i] + \
                    # nn.LayerNormalization(),
                    [nn.Dense(units=tf.reduce_prod(im_size)),
                    nn.Reshape(im_size + (-1,))
                     ], name="cond_{}_embedding".format(cond_i)))
            self.cond_mlp_down.append(tmp_cond_layer)

            num_units = self.num_filt_start * 2 ** di
            tmp_layer = []
            # Inner block
            if self.block_type == 'conv':
                tmp_layer.append(ConvBlock(ndim, num_units,
                                           **hlp.dict_merge({'conv_args': self.conv_args},
                                                            self.block_params)))
            else:
                raise NotImplementedError(
                    'Block type unknown ({})'.format(self.block_type))
            # Downsampling
            if di < self.depth - 1:
                tmp_layer.append(f_maxpool(pool_size=kernel_size_up, padding='same'))
                im_size = tuple([(size + 1) // self.pool_size for size in im_size])
                self.im_size.append(im_size)
                if self.dropout is not None:
                    tmp_layer.append(nn.Dropout(self.dropout))
                else:
                    tmp_layer.append(Identity())
            else:
                tmp_layer.append(Identity())  # fake maxpool
                tmp_layer.append(Identity())  # fake dropout

            self.downs.append(tmp_layer)

        # Upsampling Block
        for di in range(self.depth - 1, 0, -1):
            # Condition embedding
            tmp_cond_layer = []
            for cond_i in range(2):
                if self.sinusoidal_cond_mlp and cond_i == 0:
                    sin_emb = SinusoidalPosEmb(self.sin_emb_dim)
                else:
                    sin_emb = Identity()
                tmp_cond_layer.append(Sequential([
                    sin_emb, nn.Flatten() if self.cond_params.get('flag_flatten_input', True)
                    else Identity()] + cond_emb_layer[cond_i] + \
                    # nn.LayerNormalization(),
                    [nn.Dense(units=tf.reduce_prod(im_size)),
                    nn.Reshape(im_size + (-1,))
                     ], name="cond_{}_embedding".format(cond_i)))
            self.cond_mlp_up.append(tmp_cond_layer)

            tmp_layer = []
            num_units = self.num_filt_start * 2 ** (di - 1)
            # Upsampling
            tmp_layer.append(f_upsamp(kernel_size_up))
            im_size = self.im_size[::-1][-di]
                # tuple([size * self.pool_size for size in im_size])
            self.im_size.append(im_size)
            tmp_layer.append(f_conv(num_units, kernel_size=kernel_size_up,
                                    **self.conv_args))
            if self.dropout is not None:
                tmp_layer.append(nn.Dropout(self.dropout))
            else:
                tmp_layer.append(Identity())
            # Inner block
            if self.block_type == 'conv':
                tmp_layer.append(ConvBlock(ndim, num_units,
                                              **hlp.dict_merge({'conv_args': self.conv_args},
                                                           self.block_params)))
            else:
                raise NotImplementedError(
                    'Block type unknown ({})'.format(self.block_type))
            self.ups.append(tmp_layer)

        self.final_conv = f_conv(self.num_channels_out, kernel_size=(1,) * ndim,
                                 activation=self.final_activation, dtype=TF_DTYPE,
                                 **self.conv_args)

    def call(self, inputs, time=None, condition=None, training=False, **kwargs):
        x = inputs

        if self.normalize_layer['feature'] is not None:
            x = self.normalize_layer['feature'](x)
        if self.normalize_layer['label'] is not None:
            condition = self.normalize_layer['label'](condition)

        if self.resize_input_layer is not None:
            x_split = tf.unstack(x, axis=-1)
            for kk in range(len(x_split)):
                x_split[kk] = self.resize_input_layer[0](x_split[kk], training=training)
            x = tf.stack(x_split, axis=-1)

        conv_out = []

        for di, (conv, pool, dropout) in enumerate(self.downs):
            # Apply MLP and Concatenate condition
            if time is not None:
                time_cond = [self.cond_mlp_down[di][0](time, training=training),]
            else:
                time_cond = []
            if condition is not None:
                label_cond = [self.cond_mlp_down[di][1](condition, training=training),]
            else:
                label_cond = []
            # self.cond_shape = tf.shape(condition)
            # self.x_shape = tf.shape(time_cond)
            x = tf.concat(label_cond + time_cond + [x,], axis=-1,
                          name='concat_down_{}'.format(di))
            # Convolution
            x = conv(x, training=training)
            # Keep output for skip connection
            conv_out.append(x)
            # Downsample (or identity if last layer)
            x = pool(x, training=training)
            # Dropout (or identity if self.dropout is None)
            x = dropout(x, training=training)

        for di, (up, conv1, dropout, conv2) in enumerate(self.ups):
            # Apply MLP and Concatenate condition
            if time is not None:
                time_cond = [self.cond_mlp_up[di][0](time, training=training),]
            else:
                time_cond = []
            if condition is not None:
                label_cond = [self.cond_mlp_up[di][1](condition, training=training),]
            else:
                label_cond = []
            x = tf.concat(label_cond + time_cond + [x,], axis=-1,
                          name='concat_up_{}'.format(di))
            # Upsample
            x = up(x, training=training)
            x = conv1(x, training=training)
            # Pre-process incoming skip connection
            if self.skip_conn_op is None:
                x_skip_in = conv_out[self.depth - 2 - di]
            else:
                raise NotImplementedError(
                    'Skip connection pre-processing not implemented ({})'.
                    format(self.skip_conn_op))
            # Skip connection (concat or add)
            if self.skip_conn_type == 'concat':
                x = nn.concatenate([x_skip_in, x])
            elif self.skip_conn_type == 'add':
                x = nn.add([x_skip_in, x])
            else:
                raise NotImplementedError(
                    'Skip connection type unknown ({})'.
                    format(self.skip_conn_type))
            # Post-process output of skip connection
            if self.skip_conn_post_op is not None:
                raise NotImplementedError(
                    'Skip connection post-processing not implemented ({})'.
                    format(self.skip_conn_post_op))

            # Dropout
            x = dropout(x, training=training)
            x = conv2(x, training=training)

        x = self.final_conv(x, training=training)

        if self.resize_input_layer is not None:
            x_split = tf.unstack(x, axis=-1)
            for kk in range(len(x_split)):
                x_split[kk] = self.resize_input_layer[-1](x_split[kk], training=training)
            x = tf.stack(x_split, axis=-1)

        if self.denormalize_layer['feature'] is not None:
            if self.flag_learn_var:
                x_mean, x_var = tf.split(x, num_or_size_splits=2, axis=-1)
                x_mean = self.denormalize_layer['feature'](x_mean)
                x = tf.concat([x_mean, x_var], axis=-1)
            else:
                x = self.denormalize_layer['feature'](x)

        if self.constrained_func_output is not None:
            x = self.constrained_func_output(x)

        return x

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            # "num_channels_out":    self.num_channels_out // 2 if self.flag_learn_var else self.num_channels_out,
            "num_filt_start":      self.num_filt_start,
            "pool_size":           self.pool_size,
            "depth":               self.depth,
            "block_type":          self.block_type,
            "block_params":        self.block_params,
            "skip_conn_type":      self.skip_conn_type,
            "skip_conn_op":        self.skip_conn_op,
            "skip_conn_post_op":   self.skip_conn_post_op,
            "dropout":             self.dropout,
            "final_activation":    self.final_activation,
            "sin_emb_dim":         self.sin_emb_dim,
            "conv_args":           self.conv_args,
            'cond_params':         self.cond_params,
            'normalize_feature_dict': self.normalize_feature_dict,
            'normalize_label_dict':   self.normalize_label_dict,
            'learn_variance':       self.learn_variance,
            'resize_input_len': 	self.resize_input_len,
            'resize_input_type': 	self.resize_input_type,
            'constrained_func_output': self.constrained_func_output,

        })

        return base_config











