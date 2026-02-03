from abc import ABC
from abc import abstractmethod
from skimage.metrics import structural_similarity, mean_squared_error, peak_signal_noise_ratio


class QualityMeasure(ABC):
    """Abstract class that allows to call functionality of the quality measure objects.

    Args:
        ABC: Python abc module helper class that provides a standard way to create an abstract class using inheritance.
    """

    def __init__(self, name: str):
        """Initializing method that allows to assign quality measure name
        used in the process of saving results.

        Args:
            name (str): Quality measure name.
        """
        self.name = name
        self.collected_values = []
        self.time_values = []
        self.average_time = 0.0


    @abstractmethod
    def calculate_quality(self, reference_image, deformed_image):
        """Calculate quality value for an image.

        Args:
            reference_image (ndarray): reference image - base for quality measure.
            deformed_image (ndarray): deformed image - image that is assessed in relation to reference image.
        """
        pass


class MSE(QualityMeasure):
    """MSE quality measure class.

    Args:
        QualityMeasure: Abstract parent class.
    """

    def __init__(self, name="mse"):
        """Initializing method that allows to assign quality measure name
        used in the process of saving results. Used here to add default value.

        Args:
            name (str, optional): Quality measure name. Defaults to "mse".
        """
        super().__init__(name)


    def calculate_quality(self, reference_image, deformed_image):
        """Used to override parent method with MSE functionality."""
        return mean_squared_error(reference_image, deformed_image)


class PSNR(QualityMeasure):
    """PSNR quality measure class.

    Args:
        QualityMeasure: Abstract parent class.
    """

    def __init__(self, name="psnr"):
        """Initializing method that allows to assign quality measure name
        used in the process of saving results. Used here to add default value.

        Args:
            name (str, optional): Quality measure name. Defaults to "psnr".
        """
        super().__init__(name)


    def calculate_quality(self, reference_image, deformed_image):
        """Used to override parent method with PSNR functionality."""
        return peak_signal_noise_ratio(reference_image, deformed_image)


class SSIM(QualityMeasure):
    """SSIM quality measure class.

    Args:
        QualityMeasure: Abstract parent class.
    """

    def __init__(self, name="ssim"):
        """Initializing method that allows to assign quality measure name
        used in the process of saving results. Used here to add default value.

        Args:
            name (str, optional): Quality measure name. Defaults to "ssim".
        """
        super().__init__(name)


    def calculate_quality(self, reference_image, deformed_image):
        """Used to override parent method with SSIM functionality."""
        return structural_similarity(reference_image, deformed_image, channel_axis=2)