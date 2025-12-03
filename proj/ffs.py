import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from skimage.exposure import rescale_intensity


# ---------------- spectral residue saliency ---------------- #

def mat2gray(x):
    xmin, xmax = np.min(x), np.max(x)
    if xmax - xmin < 1e-12:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def spectral_residue_saliency(image):

    scale = 0.25
    aveKernelSize = 3
    gauSigma = 6
    gauSize = 15

    new_rows = int(image.shape[0] * scale)
    new_cols = int(image.shape[1] * scale)

    inImg = resize(
        image,
        (new_rows, new_cols),
        order=1,          # bilinear
        mode='reflect',
        anti_aliasing=False
    )

    myFFT = np.fft.fft2(inImg)

    logAmplitude = np.log(np.abs(myFFT) + 1e-12)
    phase = np.angle(myFFT)

    aveKernel = np.ones((aveKernelSize, aveKernelSize)) / (aveKernelSize ** 2)

    logAmp_blur = convolve2d(logAmplitude, aveKernel, mode='same', boundary='symm')

    spectralResidual = logAmplitude - logAmp_blur

    saliency = np.abs(np.fft.ifft2(np.exp(spectralResidual + 1j * phase))) ** 2

    saliency = gaussian_filter(saliency, gauSigma)

    saliency = mat2gray(saliency)

    saliency = resize(saliency, image.shape, order=1, mode='reflect')

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
        blurred = convolve2d(x, aveKernel, mode='same', boundary='symm')
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
    dy = dx.T

    def grad_mag(x):
        Ix = convolve2d(x, dx, mode='same', boundary='symm')
        Iy = convolve2d(x, dy, mode='same', boundary='symm')
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

    score[score < 0] = 0

    score = np.sqrt(score)
    score = np.sqrt(score)
    score = np.mean(np.abs(score - np.mean(score))) ** 0.15

    if np.isnan(score):
        print("Score is NaN")
        exit()

    return float(score)
