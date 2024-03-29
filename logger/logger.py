import numpy as np
import os
from mpi4py import MPI
class ProgressLogger:
    def __init__(self, path) -> None:
        self.progress_file_path = os.path.join(path, "progress.csv")
        self.output_file_path = os.path.join(path, 'output.txt')
        self.epoch_file_path = os.path.join(path, 'epoch.csv')
        self.capacity = 50
        self.returns = np.zeros(shape=(self.capacity,), dtype=np.float32)
        self.actor_loss = np.zeros(shape=(self.capacity,), dtype=np.float32)
        self.critic_loss = np.zeros(shape=(self.capacity,), dtype=np.float32)
        self.values = np.zeros(shape=(self.capacity,), dtype=np.float32)
        self.complete_episodes = np.zeros(shape=(self.capacity,), dtype=np.float32)
        self.it = 0
        if MPI.COMM_WORLD.Get_rank() == 0:
            with open(self.progress_file_path, 'w+') as f:
                f.write("returns,actor_loss,critic_loss,complete_episodes,values\n")
            with open(self.epoch_file_path, 'w+') as f:
                f.write("success_rate,misc\n")
    
    def add(self, ep_return, actor_loss, critic_loss, complete_episodes, value):
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.returns[self.it] = ep_return
            self.actor_loss[self.it] = actor_loss
            self.critic_loss[self.it] = critic_loss
            self.complete_episodes[self.it] = complete_episodes
            self.values[self.it] = value
            self.it += 1
            if self.it == self.capacity:
                with open(self.progress_file_path, "a") as f:
                    for i in range(self.it):
                        f.write(f"{self.returns[i]},{self.actor_loss[i]},{self.critic_loss[i]},{self.complete_episodes[i]},{self.values[i]}\n")
                self.it = 0

    def add_epoch_data(self, success_rate, misc=None):
        if MPI.COMM_WORLD.Get_rank() == 0:
            with open(self.epoch_file_path, "a") as f:
                f.write(f"{success_rate},{0 if misc is None else misc}\n")

    def print_and_log_output(self, output):
        print(output)
        if MPI.COMM_WORLD.Get_rank() == 0:
            with open(self.output_file_path, 'a') as f:
                f.write(output + "\n")

    def print_last_ten_runs_stat(self, current_iteration):
        if self.it >= 10:
            print(f"Runs: {current_iteration} Mean return from last 10 episodes: {np.mean(self.returns[self.it-10:self.it])} Mean Q: {np.mean(self.values[self.it-10:self.it])} Complete episodes: {self.complete_episodes[self.it - 1]}")