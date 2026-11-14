import numpy as np
from collections import deque
import warnings

class SimpleMovingAvg1D:
    def __init__(self, window_size: int, vector_dim: int):
        self.window_size = window_size
        self.buffer = np.full((window_size, vector_dim), np.nan, dtype=np.float64)
        self.ptr = 0

    def update(self, new_vector: np.ndarray) -> np.ndarray:
        self.buffer[self.ptr] = new_vector
        self.ptr = (self.ptr + 1) % self.window_size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(self.buffer, axis=0)
    

class SimpleMovingAvg0D:
    def __init__(self, window_size: int):
        self.queue = deque(maxlen=window_size)

    def update(self, new_value: float) -> float:
        self.queue.append(float(new_value))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return float(np.nanmean(self.queue))