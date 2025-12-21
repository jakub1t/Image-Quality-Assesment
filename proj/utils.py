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
    MATLAB-like conv2d.
    
    :param x: array_like
    :param y: array_like
    :param mode: A string indicating the size of the output
    """
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def imresize(img, scale_or_size, method="bilinear", antialias=True):
    """
    MATLAB-like imresize.
    
    Parameters
    ----------
    img : ndarray
        Input image (H x W x C or H x W)
    scale_or_size : float, tuple
        - scalar: scale factor
        - (newH, newW): output size in pixels
    method : str
        "nearest", "bilinear", "bicubic"
    antialias : bool
        Apply anti-alias filtering when downsampling (default = True, matches MATLAB)
    """

    # Determine output size
    if isinstance(scale_or_size, (int, float)):
        scale = float(scale_or_size)
        out_h = int(np.round(img.shape[0] * scale))
        out_w = int(np.round(img.shape[1] * scale))
    else:
        out_h, out_w = scale_or_size
        scale_h = out_h / img.shape[0]
        scale_w = out_w / img.shape[1]
        scale = (scale_h, scale_w)

    # If scale_or_size was size tuple
    if not isinstance(scale_or_size, (int, float)):
        zoom_factors = (scale_h, scale_w) + (() if img.ndim == 2 else (1,))
    else:
        zoom_factors = (scale, scale) + (() if img.ndim == 2 else (1,))

    orders = {
        "nearest": 0,
        "bilinear": 1,
        "bicubic": 3
    }
    order = orders.get(method.lower(), 1)

    # Anti-alias filtering for downsampling
    if antialias:
        if isinstance(scale_or_size, (int, float)):
            scale_h = scale_w = scale
        if scale_h < 1 or scale_w < 1:
            sigma = max(1/scale_h, 1/scale_w) / 3
            img = gaussian_filter(img, sigma=(sigma, sigma, 0) if img.ndim == 3 else (sigma, sigma))

    out = zoom(img, zoom_factors, order=order)
    return out
