import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from skimage.exposure import rescale_intensity

from utils import conv2, imresize


# ---------------- spectral residue saliency ---------------- #

def spectral_residue_saliency(image):

    scale = 0.25
    aveKernelSize = 3
    gauSigma = 6

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

    def downsample_(x):
        result = conv2(x, aveKernel)
        return result[0:rows:F, 0:cols:F]

    L1 = downsample_(L1)
    L2 = downsample_(L2)
    M1 = downsample_(M1)
    M2 = downsample_(M2)
    N1 = downsample_(N1)
    N2 = downsample_(N2)

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

    x = np.asarray(score).reshape(-1)

    quarter_power = np.sign(x) * np.sqrt(np.sqrt(np.abs(x)))
    mad_val = np.median(np.abs(quarter_power - np.median(quarter_power)))
    score_out = mad_val ** 0.15

    return score_out
