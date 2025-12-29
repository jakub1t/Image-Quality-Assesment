import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, zoom


def safe_clip_nonfinite(arr):
    """
    Used to eliminate inf, -inf and NaN signs in correlation coefficients calculation.
    
    :param arr: array-like
    """
    arr = np.asarray(arr, dtype=np.float64)
    # handle NaN
    if np.isnan(arr).any():
        arr = np.where(np.isnan(arr), np.nanmean(arr), arr)
    # handle +inf
    if np.isposinf(arr).any():
        max_finite = np.nanmax(arr[np.isfinite(arr)])
        arr = np.where(np.isposinf(arr), max_finite, arr)
    # handle -inf
    if np.isneginf(arr).any():
        min_finite = np.nanmin(arr[np.isfinite(arr)])
        arr = np.where(np.isneginf(arr), min_finite, arr)
    return arr


def logistic_regression_fun(x, b1, b2, b3, b4, b5):
    """
    Used to calculate logistic regression for array with numbers (used for PLCC).
    
    :param x: array-like
    :param b1, b2, b3, b4, b5 params: beta parameters to be fitted
    """
    return b1 * (0.5 - 1.0 / (1 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def conv2(x, y, mode='same'):
    """
    MATLAB-like conv2.
    
    :param x: array_like
    :param y: array_like
    :param mode: A string indicating the size of the output
    """
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode, boundary="symm"), 2)


def mad(x):
    x = np.ma.array(x).compressed()
    med = np.mean(x)
    return np.mean(np.abs(x - med))


def gauss2D(shape = (3, 3), sigma = 0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp( -(x * x + y * y) / (2. * sigma * sigma) )
    h[ h < np.finfo(h.dtype).eps * h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h