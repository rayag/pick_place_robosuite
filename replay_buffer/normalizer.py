import numpy as np
import threading
from mpi4py import MPI

class Normalizer:
    def __init__(self, dim, eps=1e-3, clip_range=200) -> None:
        self.N_local = np.zeros(shape=(1)) # needs to be arrat because of the MPI reduce functions
        self.sum_local = np.zeros(shape=(1, dim))
        self.sq_sum_local = np.zeros(shape=(1, dim)) # sum of the squares of the data
        # global
        self.N_global = np.zeros(shape=(1))
        self.sum_global = np.zeros(shape=(1, dim))
        self.sq_sum_global = np.zeros(shape=(1, dim)) # sum of the squares of the data

        self.eps = np.full(shape=(1, dim), fill_value=eps)
        self.mean = np.zeros(shape=(1, dim))
        self.std = np.ones_like(self.mean)
        self.clip = clip_range
        self.lock = threading.Lock()
    
    def update_stats(self, new_data):
        assert new_data.shape[1] == self.sum_local.shape[1]
        new_data_sum = np.sum(new_data, axis=0)
        new_data_sq_sum = np.sum(new_data ** 2, axis=0)
        with self.lock:
            self.sum_local = self.sum_local + new_data_sum
            self.sq_sum_local = self.sq_sum_local + new_data_sq_sum
            self.N_local = self.N_local + new_data.shape[0]

    def set_mean_std(self, mean, std):
        self.mean = mean
        self.std = std

    def sync_stats(self, single_worker=False):
        with self.lock:
            loc_N = self.N_local.copy()
            loc_sum = self.sum_local.copy()
            loc_sq_sum = self.sq_sum_local.copy()
            self.N_local[...] = 0
            self.sum_local[...] = 0
            self.sq_sum_local[...] = 0
        if not single_worker:
            # sync stats across workers
            loc_N = _mpi_average(loc_N)
            loc_sum = _mpi_average(loc_sum)
            loc_sq_sum = _mpi_average(loc_sq_sum)
        # update global data
        self.N_global += loc_N
        self.sum_global += loc_sum
        self.sq_sum_global += loc_sq_sum
        if np.all(self.N_global) > 0:
            self.mean = self.sum_global / self.N_global
            self.std = np.sqrt(np.max([np.square(self.eps), 
                (self.sq_sum_global / self.N_global[0]) - np.square(self.sum_global / self.N_global[0])], axis=0))

 
    def normalize(self, data):
        data_cp = np.copy(data)
        return np.clip((data_cp - self.mean) / self.std, a_min=-self.clip, a_max=self.clip)

def _mpi_average(x):
    buf = np.zeros_like(x)
    MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
    buf /= MPI.COMM_WORLD.Get_size()
    return buf
