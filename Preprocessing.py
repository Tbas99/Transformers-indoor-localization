import numpy as np

class Normalizer(object):
    """Normalize the RSS data according to formula (1)"""
    
    def __init__(self, rssMin):
        """
        Args:
            rssMin: The minimum observed rss value in the dataset
        """
        self.rssMin = rssMin

    def __call__(self, samples):
        normalizedSamples = np.where(
            samples == 100,
            0,
            (samples - self.rssMin))
        return normalizedSamples
