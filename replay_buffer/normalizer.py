import numpy as np

class Normalizer:
    def __init__(self, dim, eps=1e-3, clip_range=200) -> None:
        self.N = 0
        self.sum = np.zeros(shape=(1, dim))
        self.sq_sum = np.zeros(shape=(1, dim)) # sum of the squares of the data
        self.eps = eps
        self.clip = clip_range
    
    def update_stats(self, new_data):
        assert new_data.shape[1] == self.sum.shape[1]
        new_data_sum = np.sum(new_data, axis=0)
        new_data_sq_sum = np.sum(new_data ** 2, axis=0)
        self.sum = self.sum + new_data_sum
        self.sq_sum = self.sq_sum + new_data_sq_sum
        self.N = self.N + new_data.shape[0]

    def normalize(self, data):
        data_cp = np.copy(data)
        mean = self.sum / self.N
        std = np.max(((self.sq_sum - (self.sum ** 2) / self.N) / self.N, np.full_like(self.sum, fill_value=self.eps)))
        return np.clip((data_cp - mean) / std, a_min=-self.clip, a_max=self.clip)
