import copy
import functools

import numpy as np
from numpy.random import default_rng
from scipy.stats import truncnorm
from diffusion_model import NP_DTYPE


## Basic Layers and Blocks

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def merge(*dict_list, **kwargs):
    """Merge dictionary objects

    Contrary to :func:`dict.update`, perform a recursive merge (see examples).

    Parameters
    ----------
    dict_list : list
        Input dictionaries to merge.
    kwargs : dict
        Merge parameters:

        * ``flag_copy`` [bool]: Control setting operation (copy or assignment).

    Returns
    -------
    dict
        Output dictionary.

    Examples
    --------
    >>> val = [1, 2, 3]
    >>> dict_0 = {1: val, 'b': {'a': 11, 'b': 12}, 'c': {'key': 'value'}}
    >>> dict_1 = {'b': {'b': 13}, 3: 'c', 'c': {}}
    >>> dict_out = merge(dict_0, None, dict_1)
    >>> dict_out_ref = {1: val, 'b': {'a': 11, 'b': 13}, 3: 'c', 'c': {}}
    >>> dict_out == dict_out_ref
    True
    >>> dict_out_c = merge(dict_0, None, dict_1, flag_copy=True)
    >>> dict_out_c == dict_out_ref
    True
    >>> val[0] = 12
    >>> dict_out == dict_out_ref
    True
    >>> dict_out_c == dict_out_ref
    False
    """
    flag_copy = kwargs.get('flag_copy', False)
    dict_merged = {}
    for dict_t in dict_list:
        if dict_t is not None:
            for key, val in dict_t.items():
                if key not in dict_merged:
                    if flag_copy:
                        dict_merged[key] = copy.deepcopy(val)
                    else:
                        dict_merged[key] = val
                else:
                    if isinstance(val, dict) and \
                            isinstance(dict_merged[key], dict) and \
                            val:
                        dict_merged[key] = merge(dict_merged[key], val)
                    else:
                        if flag_copy:
                            dict_merged[key] = copy.deepcopy(val)
                        else:
                            dict_merged[key] = val
    return dict_merged


def dict_merge(*dict_list, **kwargs):
    """Merge dictionary objects

    Contrary to :func:`dict.update`, perform a recursive merge (see examples).

    Parameters
    ----------
    dict_list : list
        Input dictionaries to merge.
    kwargs : dict
        Merge parameters:

        * ``flag_copy`` [bool]: Control setting operation (copy or assignment).

    Returns
    -------
    dict
        Output dictionary.

    Examples
    --------
    >>> val = [1, 2, 3]
    >>> dict_0 = {1: val, 'b': {'a': 11, 'b': 12}, 'c': {'key': 'value'}}
    >>> dict_1 = {'b': {'b': 13}, 3: 'c', 'c': {}}
    >>> dict_out = merge(dict_0, None, dict_1)
    >>> dict_out_ref = {1: val, 'b': {'a': 11, 'b': 13}, 3: 'c', 'c': {}}
    >>> dict_out == dict_out_ref
    True
    >>> dict_out_c = merge(dict_0, None, dict_1, flag_copy=True)
    >>> dict_out_c == dict_out_ref
    True
    >>> val[0] = 12
    >>> dict_out == dict_out_ref
    True
    >>> dict_out_c == dict_out_ref
    False
    """
    flag_copy = kwargs.get('flag_copy', False)
    dict_merged = {}
    for dict_t in dict_list:
        if dict_t is not None:
            for key, val in dict_t.items():
                if key not in dict_merged:
                    if flag_copy:
                        dict_merged[key] = copy.deepcopy(val)
                    else:
                        dict_merged[key] = val
                else:
                    if isinstance(val, dict) and \
                            isinstance(dict_merged[key], dict) and \
                            val:
                        dict_merged[key] = merge(dict_merged[key], val)
                    else:
                        if flag_copy:
                            dict_merged[key] = copy.deepcopy(val)
                        else:
                            dict_merged[key] = val
    return dict_merged


def trunc_normal(mean=0, std=1, low=0, upp=None, **kwargs):
    if upp is None:
        upp = np.Inf
    return truncnorm.rvs(
        (low - mean) / std, (upp - mean) / std, loc=mean, scale=std, **kwargs)


def truncnormal_samples(mu, cov, cov_inv, n_samples, cond_test=None, **kwargs):
    if cond_test is None:
        def cond_test(vec_test, mu_test, cov_inv_test):
            return True
    samples_output = []
    while len(samples_output) < n_samples:
        tmp = np.random.multivariate_normal(mu, cov, **kwargs)
        if np.sum(tmp < 0) == 0 and cond_test(tmp, mu, cov_inv):
            samples_output.append(tmp)
    return samples_output


def random_cov_psd(diag_element, *, rng=None):
    """
    Create an n×n covariance matrix whose diagonal variances lie in
    [var_low, var_high] and whose off-diagonals are random but
    guaranteed positive-semi-definite.

    Parameters
    ----------
    diag_element : numpy.ndarray
        (n,) elements to put in the diagonal of the covariance matrix
    n                 : int
        Matrix size.
    rng               : np.random.Generator or None
        Pass your own RNG for reproducibility.
    full_rank         : bool  (default True)
        If False the matrix will be *semi*-definite (rank<n).
    """
    rng = default_rng() if rng is None else rng

    # choose the diagonal you want
    std = np.sqrt(diag_element)
    n = len(diag_element)

    # sample a random correlation matrix R (diag = 1)
    # Trick:  Wishart sample  →  scale →  shrink toward I if you want
    A = rng.standard_normal((n, n))
    R = A @ A.T  # PSD, full rank
    d = np.sqrt(np.diag(R))
    R = R / d[:, None] / d  # convert to correlation

    # Optional:  pull eigenvalues towards 1 to avoid wild correlations
    # λ̂ = ρ λ + (1-ρ)       with ρ∈[0,1)
    rho = 0.5
    eig, Q = np.linalg.eigh(R)
    eig = rho * eig + (1 - rho)  # now in (1-rho, 1]
    R = Q @ np.diag(eig) @ Q.T

    # scale back to covariance Σ = D R D
    Sigma = (std[:, None] * R) * std  # outer product trick
    return Sigma





def cos_beta_schedule(timesteps, offset_s=0.008, max_beta=0.999):
    """cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
    beta = []
    alpha_bar = lambda t: np.cos((t + offset_s) / (1 + offset_s) * np.pi / 2,
                                 dtype=NP_DTYPE) ** 2
    for i in range(timesteps):
        t1 = i / timesteps
        t2 = (i + 1) / timesteps
        beta.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(beta, dtype=NP_DTYPE)


def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    sigmoid = lambda t: 1 / (1 + np.exp(-t, dtype=NP_DTYPE))
    beta = np.linspace(-6, 6, timesteps, dtype=NP_DTYPE)
    return sigmoid(beta) * (beta_end - beta_start) + beta_start


def quadratic_beta_schedule(timesteps, beta_start, beta_end):
    return np.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps,
                       dtype=NP_DTYPE) ** 2


def linear_beta_schedule(timesteps, beta_start, beta_end):
    return np.linspace(beta_start, beta_end, timesteps, dtype=NP_DTYPE)


def get_beta_schedule(schedule_name, timesteps, **kwargs):
    """
    Retrieve the beta scheduler function from the input string ``schedule_name``.

    Params
    ---------

    :param schedule_name: String indicating the type of noise scheduling:
    ``linear`` (or ``lin``), ``cosine`` (or ``cos``), ``quadratic`` (or ``quad``),
    and ``sigmoid`` (or ``sig``).
    :param timesteps: Number of timesteps T (e.g., 1000)
    :param kwargs: parameters for noise scheduling. beta_start, beta_end,
    offset_s, max_beta
    :return: beta schedule
    """
    beta_start = kwargs.get('beta_start', None)
    beta_end = kwargs.get('beta_end', None)
    offset_s = kwargs.get('offset_s', None)
    max_beta = kwargs.get('max_beta', None)

    if schedule_name.lower() in ('lin', 'linear'):
        beta = linear_beta_schedule(timesteps, beta_start, beta_end)
    elif schedule_name.lower() in ('quad', 'quadratic'):
        beta = quadratic_beta_schedule(timesteps, beta_start, beta_end)
    elif schedule_name.lower() in ('sig', 'sigmoid'):
        beta = sigmoid_beta_schedule(timesteps, beta_start, beta_end)
    elif schedule_name.lower() in ('cos', 'cosine'):
        beta = cos_beta_schedule(timesteps, offset_s=offset_s, max_beta=max_beta)
    else:
        raise NotImplementedError('Schedule name ({}) not recognized or '
                                  'not implemented.'.format(schedule_name))
    return beta



def sampling_from_normal(mu, logvar, size=None, **kwargs):
    std = np.exp(0.5 * logvar)
    epsilon = np.random.normal(size=size or mu.shape, **kwargs)
    return mu + std * epsilon
