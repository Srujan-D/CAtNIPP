import numpy as np


class MinMaxScaler:
    def __init__(
        self, expected_min=-1.0, expected_max=1.0, actual_min=None, actual_max=None
    ) -> None:
        self.expected_min = expected_min
        self.expected_max = expected_max
        self.expected_range = expected_max - expected_min
        self.actual_min = 0 # actual_min
        self.actual_max = 1.0 # actual_max
        if actual_min is not None and actual_max is not None:
            self.actual_range = actual_max - actual_min
        else:
            self.actual_range = None

    def fit(self, x: np.ndarray) -> None:
        self.actual_min = x.min(axis=0, keepdims=True)
        self.actual_max = x.max(axis=0, keepdims=True)
        self.actual_range = self.actual_max - self.actual_min

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        assert self.actual_range is not None
        try:
            standardized = (raw - self.actual_min) / self.actual_range
        except:
            print('------------in except block------------')
            print('-----raw-----', raw, raw.shape)
            print('-----self.actual_min-----', self.actual_min, self.actual_min.shape)
            print('-----self.actual_range-----', self.actual_range, self.actual_range.shape)
            quit()
        return self.expected_min + standardized * self.expected_range

    def postprocess(self, scaled: np.ndarray) -> np.ndarray:
        standardized = (scaled - self.expected_min) / self.expected_range
        return self.actual_min + standardized * self.actual_range
