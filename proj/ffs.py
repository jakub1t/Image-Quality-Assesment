import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from skimage.transform import resize

from utils import conv2, mad


# ---------------- spectral residue saliency ---------------- #

def spectral_residue_saliency(image):

    scale = 0.25
    ave_kernel_size = 3
    gau_sigma = 6

    in_img = resize(image, (int(image.shape[0] * scale), int(image.shape[1] * scale)), anti_aliasing=True)

    my_fft = np.fft.fft2(in_img)
    log_amplitude = np.log(np.abs(my_fft) + 1e-8)
    phase = np.angle(my_fft)

    ave_kernel = np.ones((ave_kernel_size, ave_kernel_size)) / (ave_kernel_size ** 2)
    smooth_log_amp = convolve2d(log_amplitude, ave_kernel, mode='same', boundary='symm')

    spectral_residual = log_amplitude - smooth_log_amp
    saliency_map = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * phase))) ** 2

    saliency_map = gaussian_filter(saliency_map, sigma=gau_sigma)
    saliency_map -= saliency_map.min()
    saliency_map /= (saliency_map.max() + 1e-8)

    saliency_map = resize(saliency_map, image.shape, anti_aliasing=True)

    return saliency_map


# ---------------------- main FFS function ---------------------- #

def calculate_ffs(reference_image, deformaed_image):
    
    img1 = np.float64(reference_image)
    img2 = np.float64(deformaed_image)

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

    min_dim = min(rows, cols)
    F = max(1, round(min_dim / 256))
    ave_kernel = np.ones((F, F)) / (F ** 2)

    def downsample_(x):
        result = conv2(x, ave_kernel)
        return result[::F, ::F]

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
                   [1,0,-1]], dtype=np.float64) / 3.0
    dy = dx.conj().transpose()

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

    score_out = mad((score.flatten() ** 0.5) ** 0.5) ** 0.15
    # score_out = np.mean(np.abs((score.flatten() ** 0.5) ** 0.5 - np.mean((score.flatten() ** 0.5) ** 0.5))) ** 0.15

    return score_out
