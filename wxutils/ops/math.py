import numpy as np
from collections import deque

class SimpleMovingAvg1D:
    def __init__(self, window_size, vector_dim=None):
        self.window_size = window_size
        self.queue = deque(maxlen=window_size)
        # Initialize a running sum vector of zeroes
        if vector_dim:
            self.running_sum = np.zeros(vector_dim, dtype=np.float64)
        else:
            self.running_sum = None
    
    def update(self, new_vector: np.ndarray) -> np.ndarray:
        if self.running_sum is None:
            N = len(new_vector)
            self.running_sum = np.zeros(N, dtype=np.float64)
        
        if len(self.queue) == self.window_size:
            self.running_sum -= self.queue[0]
            
        self.queue.append(new_vector)
        self.running_sum += new_vector
        
        return self.running_sum / len(self.queue)
    

class SimpleMovingAvg0D:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.queue = deque(maxlen=window_size)
        self.running_sum = 0.0

    def update(self, new_value: float) -> float:
        if len(self.queue) == self.window_size:
            self.running_sum -= self.queue[0]
            
        self.queue.append(new_value)
        self.running_sum += new_value
        
        return self.running_sum / len(self.queue)