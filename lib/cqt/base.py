import numpy as np
from lib.sharedtypes import ExtractedFeature


class BaseCQT:
    def extract(self, audio_slice: np.ndarray) -> ExtractedFeature:
        raise NotImplementedError("Override this")
