
from scipy.ndimage import convolve
from scipy import signal
import numpy as np

from quality_measure import QualityMeasure


class SG_ESSIM(QualityMeasure):

    def calculate_quality(self, reference_image, deformed_image):
        return calculate_sg_essim(reference_image, deformed_image)


def preprocess(reference_image, deformed_image):

    ref_image = reference_image.astype(np.float64)
    def_image = deformed_image.astype(np.float64)

    if ref_image.ndim == 3 and ref_image.shape[2] == 3:
        ref_image = 0.299 * ref_image[:, :, 0] + 0.587 * ref_image[:, :, 1] + 0.114 * ref_image[:, :, 2]
        def_image = 0.299 * def_image[:, :, 0] + 0.587 * def_image[:, :, 1] + 0.114 * def_image[:, :, 2]
    
    return ref_image, def_image


def automatic_downsampling(reference_image, deformed_image):

    n_rows, n_cols = reference_image.shape

    f = max(1, round(min(n_rows, n_cols) / 256)) # downsampling factor

    if f > 1:
        low_pass_filter = np.ones((f, f), dtype=np.float64)
        low_pass_filter /= low_pass_filter.sum()

        reference_image = convolve(reference_image, low_pass_filter, mode='reflect')
        deformed_image = convolve(deformed_image, low_pass_filter, mode='reflect')

        reference_image = reference_image[::f, ::f]
        deformed_image = deformed_image[::f, ::f]
    
    return reference_image, deformed_image    


def get_directional_gradient(image):

    n_rows, n_cols = image.shape

    gradient = np.zeros((n_rows, n_cols, 4), dtype=np.float64)

    kernel1 = np.zeros((5, 5), dtype=np.float64)

    Kt = (1 / 16) * np.array([[3, 10, 3],
                              [0,  0,  0],
                              [-3, -10, -3]], dtype=np.float64)
    
    kernel1[1:4, 1:4] = Kt

    kernel2 = kernel1.conj().transpose()

    kernel3 = (1 / 16) * np.array([
        [0,  0,  3,  0,  0],
        [0, 10,  0,  0,  0],
        [3,  0,  0,  0, -3],
        [0,  0,  0, -10, 0],
        [0,  0, -3,  0,  0]
    ], dtype=np.float64)

    kernel4 = np.rot90(kernel3)

    gradient[:, :, 0] = signal.correlate2d(image, kernel1, mode='same', boundary='fill', fillvalue=0)
    gradient[:, :, 1] = signal.correlate2d(image, kernel2, mode='same', boundary='fill', fillvalue=0)
    gradient[:, :, 2] = signal.correlate2d(image, kernel3, mode='same', boundary='fill', fillvalue=0)
    gradient[:, :, 3] = signal.correlate2d(image, kernel4, mode='same', boundary='fill', fillvalue=0)

    return gradient


def calculate_sg_essim(reference_image, deformed_image, h = 0.5, L = 255, K = 200):

    reference_image, deformed_image = preprocess(reference_image, deformed_image)
    reference_image, deformed_image = automatic_downsampling(reference_image, deformed_image)

    n_rows, n_cols = reference_image.shape

    ref_gradient = get_directional_gradient(reference_image)
    def_gradient = get_directional_gradient(deformed_image)

    C = (K * L) ** (2 * 0.5)

    grad_ref = np.abs(ref_gradient[:, :, [0, 2]] - ref_gradient[:, :, [1, 3]]) ** 0.5
    grad_def = np.abs(def_gradient[:, :, [0, 2]] - def_gradient[:, :, [1, 3]]) ** 0.5

    Y, X = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

    # max along gradient directions
    ind3 = np.argmax(grad_ref, axis=2)

    ind = (X.ravel() * n_cols * 2 + Y.ravel() * 2 + ind3.ravel())

    grad_ref_sel = grad_ref.ravel()[ind]
    grad_def_sel = grad_def.ravel()[ind]

    edgeMap = np.maximum(grad_ref_sel, grad_def_sel)

    H = C * np.exp(-edgeMap / h)

    SM = (2 * grad_ref_sel * grad_def_sel + H) / (grad_ref_sel**2 + grad_def_sel**2 + H)

    quality = np.mean(SM)

    return quality





