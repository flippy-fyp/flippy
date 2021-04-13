from typing import Union, List
import numpy as np
from lib.sharedtypes import ExtractedFeature


class BaseCQT:
    def extract(
        self, audio: np.ndarray
    ) -> Union[ExtractedFeature, List[ExtractedFeature]]:
        """
        Offline: Returns np.ndarray of List of ExtractedFeature
        Online:  Returns Extracted Feature for a slice
        """
        raise NotImplementedError("Override this")
