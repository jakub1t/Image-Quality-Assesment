import numpy as np
from scipy.ndimage import convolve
from scipy.special import gammaln
from scipy.linalg import toeplitz

from utils import conv2

from quality_measure import QualityMeasure


class LGV(QualityMeasure):

    def __init__(self, name="lgv"):
        super().__init__(name)


    def calculate_quality(self, reference_image, deformed_image):
        return self.calculate_lgv(reference_image, deformed_image)


    def preprocess(self, reference_image, deformed_image):

        ref_image = reference_image.astype(np.float64)
        def_image = deformed_image.astype(np.float64)

        if ref_image.ndim == 3 and ref_image.shape[2] == 3:
            ref_image = 0.299 * ref_image[:, :, 0] + 0.587 * ref_image[:, :, 1] + 0.114 * ref_image[:, :, 2]
            def_image = 0.299 * def_image[:, :, 0] + 0.587 * def_image[:, :, 1] + 0.114 * def_image[:, :, 2]
        
        return ref_image, def_image


    def automatic_downsampling(self, reference_image, deformed_image):

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


    def fgl_deriv_maxtrix_norm(self, a, Y, h):
        Y = np.asarray(Y, dtype=float)

        # ---------- Horizontal ----------
        m, n = Y.shape
        J = np.arange(n)

        log_coeff = (gammaln(a + 1) - gammaln(J + 1) - gammaln(a + 1 - J) - a * np.log(h))
        coeff = ((-1.0) ** J) * np.exp(log_coeff)

        M = np.tril(np.ones((n, n)))
        T = np.meshgrid(coeff)

        Dx = np.zeros((m, n))
        for row in range(m):
            R = toeplitz(Y[row, :])
            Dx[row, :] = np.sum(R * M * T, axis=1)

        # ---------- Vertical ----------
        Yt = Y.conj().transpose()
        m, n = Yt.shape
        J = np.arange(n)

        log_coeff = (gammaln(a + 1) - gammaln(J + 1) - gammaln(a + 1 - J) - a * np.log(h))
        coeff = ((-1.0) ** J) * np.exp(log_coeff)

        M = np.tril(np.ones((n, n)))
        T = np.meshgrid(coeff)

        Dy = np.zeros((m, n))
        for row in range(m):
            R = toeplitz(Yt[row, :])
            Dy[row, :] = np.sum(R * M * T, axis=1)

        Dy = Dy.conj().transpose()

        DM = np.sqrt(Dx**2 + Dy**2)
        return DM


    def get_local_variation(self, ref_img, def_img, T):
        Sx = np.array([[3,  0, -3],
                   [10, 0, -10],
                   [3,  0, -3]], dtype=np.float64) / 16.0

        Sy = Sx.conj().transpose()

        Gx_ref = conv2(ref_img, Sx)
        Gy_ref = conv2(ref_img, Sy)
        G_ref = np.sqrt(np.square(Gx_ref) + np.square(Gy_ref))

        Gx_def = conv2(def_img, Sx)
        Gy_def = conv2(def_img, Sy)
        G_def = np.sqrt(np.square(Gx_def) + np.square(Gy_def))

        SL = (2 * G_ref * G_def + T) / (np.square(G_ref) + np.square(G_def) + T)

        return SL


    def get_global_variation(self, ref_img, def_img, args):

        a, h, C = args

        G_ref = self.fgl_deriv_maxtrix_norm(a, ref_img, h)
        G_def = self.fgl_deriv_maxtrix_norm(a, def_img, h)

        SG = (2 * G_ref * G_def + C) / (np.square(G_ref) + np.square(G_def) + C)

        return SG


    def calculate_lgv(self, reference_image, deformed_image):
        ref_img, def_img = self.preprocess(reference_image, deformed_image)

        ref_img, def_img = self.automatic_downsampling(ref_img, def_img)

        a = 0.6
        h = 80
        L = 255
        k1 = 0.2
        k2 = 0.1
        C = (k1 * L) ** 2
        T = (k2 * L) ** 2

        args = (a, h, C)
        
        SL = self.get_local_variation(ref_img, def_img, T)
        SG = self.get_global_variation(ref_img, def_img, args)

        sim_map = (SG ** 0.7) * (SL ** 0.3)

        quality = np.mean(sim_map)

        return quality



