from abc import ABC
from abc import abstractmethod
from skimage.metrics import structural_similarity, mean_squared_error, peak_signal_noise_ratio


class QualityMeasure(ABC):

    def __init__(self, name: str):
        self.name = name
        self.collected_values = []
        self.time_values = []


    @abstractmethod
    def calculate_quality(self, reference_image, deformed_image):
        """
        Calculate quality measure for an image.
        
        :param self: this object
        :param reference_image: reference image - base for quality measure
        :param deformed_image: deformed image - image that is assessed
        """
        pass


class MSE(QualityMeasure):

    def calculate_quality(self, reference_image, deformed_image):
        return mean_squared_error(reference_image, deformed_image)


class PSNR(QualityMeasure):

    def calculate_quality(self, reference_image, deformed_image):
        return peak_signal_noise_ratio(reference_image, deformed_image)


class SSIM(QualityMeasure):

    def calculate_quality(self, reference_image, deformed_image):
        return structural_similarity(reference_image, deformed_image, channel_axis=2)