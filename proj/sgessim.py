import scipy
import cv2
import numpy as np


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

        reference_image = cv2.filter2D(reference_image, -1, low_pass_filter)
        deformed_image = cv2.filter2D(deformed_image, -1, low_pass_filter)

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

    kernel2 = kernel1.T

    kernel3 = (1 / 16) * np.array([
        [0,  0,  3,  0,  0],
        [0, 10,  0,  0,  0],
        [3,  0,  0,  0, -3],
        [0,  0,  0, -10, 0],
        [0,  0, -3,  0,  0]
    ], dtype=np.float64)

    kernel4 = np.rot90(kernel3)

    gradient[:, :, 0] = scipy.signal.convolve2d(image, kernel1, mode='same', boundary='symm')
    gradient[:, :, 1] = scipy.signal.convolve2d(image, kernel2, mode='same', boundary='symm')
    gradient[:, :, 2] = scipy.signal.convolve2d(image, kernel3, mode='same', boundary='symm')
    gradient[:, :, 3] = scipy.signal.convolve2d(image, kernel4, mode='same', boundary='symm')

    return gradient


def sg_essim(reference_image, deformed_image, h = 0.5, L = 255, K = 200):

    reference_image, deformed_image = preprocess(reference_image, deformed_image)
    reference_image, deformed_image = automatic_downsampling(reference_image, deformed_image)

    n_rows, n_cols = reference_image.shape

    ref_gradient = get_directional_gradient(reference_image)
    def_gradient = get_directional_gradient(deformed_image)

    C = (K * L) ** (2 * 0.5)

    grad_ref = np.abs(ref_gradient[:, :, [0, 2]] - ref_gradient[:, :, [1, 3]]) ** 0.5
    grad_def = np.abs(def_gradient[:, :, [0, 2]] - def_gradient[:, :, [1, 3]]) ** 0.5

    # meshgrid in MATLAB -> X=row indices, Y=col indices
    Y, X = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

    # max along gradient directions
    ind3 = np.argmax(grad_ref, axis=2)

    ind = (X.ravel() * n_cols * 2 + Y.ravel() * 2 + ind3.ravel())

    # Flatten grad arrays to use flat indices
    grad_ref_flat = grad_ref.reshape(-1, 2)
    grad_def_flat = grad_def.reshape(-1, 2)

    edgeMap = np.maximum(grad_ref_flat.ravel()[ind], grad_def_flat.ravel()[ind])

    H = C * np.exp(-edgeMap / h)

    SM = (2 * grad_ref_flat.ravel()[ind] * grad_def_flat.ravel()[ind] + H) / (grad_ref_flat.ravel()[ind]**2 + grad_def_flat.ravel()[ind]**2 + H)

    quality = np.mean(SM)

    return quality





