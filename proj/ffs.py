import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, zoom, convolve
from skimage.exposure import rescale_intensity

def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def imresize(img, scale_or_size, method='bilinear', antialias=True):
    """
    MATLAB-like imresize using SciPy.
    
    Parameters
    ----------
    img : ndarray
        Input image (H x W x C or H x W)
    scale_or_size : float, tuple
        - scalar: scale factor
        - (newH, newW): output size in pixels
    method : str
        'nearest', 'bilinear', 'bicubic'
    antialias : bool
        Apply anti-alias filtering when downsampling (default = True, matches MATLAB)
    """

    # --- determine output size ---
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

    # --- mapping interpolation method ---
    orders = {
        'nearest': 0,
        'bilinear': 1,
        'bicubic': 3
    }
    order = orders.get(method.lower(), 1)

    # --- Anti-alias filtering for downsampling ---
    if antialias:
        if isinstance(scale_or_size, (int, float)):
            scale_h = scale_w = scale
        if scale_h < 1 or scale_w < 1:
            sigma = max(1/scale_h, 1/scale_w) / 3
            img = gaussian_filter(img, sigma=(sigma, sigma, 0) if img.ndim == 3 else (sigma, sigma))

    # --- Perform resize ---
    out = zoom(img, zoom_factors, order=order)
    return out


# ---------------- spectral residue saliency ---------------- #

def spectral_residue_saliency(image):

    scale = 0.25
    aveKernelSize = 3
    gauSigma = 6
    gauSize = 15

    inImg = imresize(image, scale, method='bicubic')

    myFFT = np.fft.fft2(inImg)

    logAmplitude = np.log(np.abs(myFFT) + 1e-12)
    phase = np.angle(myFFT)

    aveKernel = np.ones((aveKernelSize, aveKernelSize), dtype=np.float64) / (aveKernelSize * aveKernelSize)

    logAmp_blur = convolve(logAmplitude, aveKernel, mode='nearest')

    spectralResidual = logAmplitude - logAmp_blur

    saliency = np.abs(np.fft.ifft2(np.exp(spectralResidual + 1j * phase))) ** 2

    saliency = gaussian_filter(saliency, gauSigma)

    saliency = rescale_intensity(saliency, in_range='image', out_range=(0, 1))

    saliency = imresize(saliency, image.shape, method='bilinear')

    return saliency


# ---------------------- main FFS function ---------------------- #

def calculate_ffs(reference_image, deformaed_image):
    
    img1 = reference_image.astype(np.float64)
    img2 = deformaed_image.astype(np.float64)

    rows, cols, _ = img1.shape

    alpha = 0.52
    Kv1 = 0.25
    Kv2 = 0.5 * Kv1
    Kv3 = 0.5 * Kv1
    Kc = 270
    Kg1 = 160
    Kg2 = 90

    L1 = 0.06*img1[:,:,0] + 0.63*img1[:,:,1] + 0.27*img1[:,:,2]
    L2 = 0.06*img2[:,:,0] + 0.63*img2[:,:,1] + 0.27*img2[:,:,2]

    M1 = 0.30*img1[:,:,0] + 0.04*img1[:,:,1] - 0.35*img1[:,:,2]
    M2 = 0.30*img2[:,:,0] + 0.04*img2[:,:,1] - 0.35*img2[:,:,2]

    N1 = 0.34*img1[:,:,0] - 0.60*img1[:,:,1] + 0.17*img1[:,:,2]
    N2 = 0.34*img2[:,:,0] - 0.60*img2[:,:,1] + 0.17*img2[:,:,2]

    minDimension = min(rows, cols)
    F = max(1, round(minDimension / 256))
    aveKernel = np.ones((F, F)) / (F * F)

    def downsample_matlab(x):
        blurred = conv2(x, aveKernel)
        return blurred[0:rows:F, 0:cols:F]

    L1 = downsample_matlab(L1)
    L2 = downsample_matlab(L2)
    M1 = downsample_matlab(M1)
    M2 = downsample_matlab(M2)
    N1 = downsample_matlab(N1)
    N2 = downsample_matlab(N2)

    FF = alpha * (L1 + L2)

    # ---------------- SR visual saliency ---------------- #

    vs1 = spectral_residue_saliency(L1)
    vs2 = spectral_residue_saliency(L2)
    vs3 = spectral_residue_saliency(FF)

    VSSim1 = (2 * vs1 * vs2 + Kv1) / (vs1**2 + vs2**2 + Kv1)
    VSSim2 = (2 * vs1 * vs3 + Kv2) / (vs1**2 + vs3**2 + Kv2)
    VSSim3 = (2 * vs3 * vs2 + Kv3) / (vs3**2 + vs2**2 + Kv3)

    VS_HVS = VSSim1 + VSSim3 - VSSim2

    # ---------------- Gradient similarity ---------------- #

    dx = np.array([[1,0,-1],
                   [1,0,-1],
                   [1,0,-1]], dtype=np.float64) / 3
    dy = np.atleast_2d(dx).T.conj()

    def grad_mag(x):
        Ix = conv2(x, dx)
        Iy = conv2(x, dy)
        return np.sqrt(Ix**2 + Iy**2)

    gR = grad_mag(L1)
    gD = grad_mag(L2)
    gF = grad_mag(FF)

    GS12 = (2*gR*gD + Kg1) / (gR**2 + gD**2 + Kg1)
    GS13 = (2*gR*gF + Kg2) / (gR**2 + gF**2 + Kg2)
    GS23 = (2*gD*gF + Kg2) / (gD**2 + gF**2 + Kg2)

    GS_HVS = GS12 + GS23 - GS13

    # ---------------- Chrominance similarity ---------------- #

    CS = (2*(N1*N2 + M1*M2) + Kc) / (N1**2 + N2**2 + M1**2 + M2**2 + Kc)

    # ---------------- Final FFS score ---------------- #

    score = 0.4 * VS_HVS + 0.4 * GS_HVS + 0.2 * CS

    score = score.flatten()
    # score[score < 0] = 0

    # score = score ** 0.5
    # score = score ** 0.5
    # score = np.mean(np.abs(score - np.mean(score))) ** 0.15

    # if np.isnan(score):
    #     print("Score is NaN")
    #     exit()

    x = np.asarray(score).reshape(-1)

    quarter_power = np.sign(x) * np.sqrt(np.sqrt(np.abs(x)))
    mad_val = np.median(np.abs(quarter_power - np.median(quarter_power)))
    score_out = mad_val ** 0.15

    # scipy built-in mad()
    # x = np.asarray(score).reshape(-1)
    # quarter_power = np.sign(x) * np.sqrt(np.sqrt(np.abs(x)))
    # mad_val = median_abs_deviation(quarter_power, scale=1)  # MATLAB uses scale=1
    # score_out = mad_val ** 0.15

    return score_out
