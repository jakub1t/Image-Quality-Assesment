import skimage
import scipy
import cv2
import numpy as np


def preprocess(original_image, deformed_image):

    ori_image = original_image.astype(np.float64)
    def_image = deformed_image.astype(np.float64)

    if ori_image.ndim == 3 and ori_image.shape[2] == 3:
        ori_image = 0.299 * ori_image[:, :, 0] + 0.587 * ori_image[:, :, 1] + 0.114 * ori_image[:, :, 2]
        def_image = 0.299 * def_image[:, :, 0] + 0.587 * def_image[:, :, 1] + 0.114 * def_image[:, :, 2]
    
    return ori_image, def_image


def automatic_downsampling(original_image, deformed_image):

    n_rows, n_cols = original_image.shape

    f = max(1, round(min(n_rows, n_cols) / 256)) # downsampling factor

    if f > 1:
        low_pass_filter = np.ones((f, f), dtype=np.float64)

        original_image = cv2.filter2D(original_image, -1, low_pass_filter)
        deformed_image = cv2.filter2D(deformed_image, -1, low_pass_filter)

        original_image = original_image[::f, ::f]
        deformed_image = deformed_image[::f, ::f]
    
    return original_image, deformed_image    


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


def sg_essim(original_image, deformed_image, h = 0.5, L = 255, K = 200):

    original_image, deformed_image = preprocess(original_image, deformed_image)
    original_image, deformed_image = automatic_downsampling(original_image, deformed_image)

    n_rows, n_cols = original_image.shape

    ori_gradient = get_directional_gradient(original_image)
    def_gradient = get_directional_gradient(deformed_image)

    C = (K * L) ** (2 * 0.5)

    grad_ori = np.abs(ori_gradient[:, :, [0, 2]] - ori_gradient[:, :, [1, 3]]) ** 0.5
    grad_def = np.abs(def_gradient[:, :, [0, 2]] - def_gradient[:, :, [1, 3]]) ** 0.5

    ind3 = np.argmax(grad_ori, axis=2)
    edgeMap = np.maximum(
        grad_ori[np.arange(n_rows)[:, None], np.arange(n_cols), ind3],
        grad_def[np.arange(n_rows)[:, None], np.arange(n_cols), ind3]
    )

    H = C * np.exp(-edgeMap / h)

    ori_selected = grad_ori[np.arange(n_rows)[:, None], np.arange(n_cols), ind3]
    def_selected = grad_def[np.arange(n_rows)[:, None], np.arange(n_cols), ind3]

    SM = (2 * ori_selected * def_selected + H) / (ori_selected**2 + def_selected**2 + H)
    
    quality = np.mean(SM)

    return quality





