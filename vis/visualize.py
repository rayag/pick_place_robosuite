import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def visualise_from_custom_progress_file(path):
    path = os.path.join(path, 'progress.csv')
    df = pd.read_csv(path)
    iterations = df.size
    figure, axis = plt.subplots(2, 3, figsize=(15, 8))
    axis[0,0].plot(df['returns'], "-b", label="Raw")
    axis[0,0].plot(running_average(df['returns'].to_numpy(), n=10), '-.', color='red', label="Mean")
    axis[0,0].legend(loc="upper left")
    axis[0,0].set_xlabel("Iteration")
    axis[0,0].set_ylabel("Reward")
    axis[0,0].set_title(path)
    print(df.keys())

    try:
        axis[0,1].plot(df['actor_loss'])
        axis[0,1].legend(loc="upper left")
        axis[0,1].set_xlabel("Iteration")
        axis[0,1].set_ylabel("Actor Loss")
    except:
        print("Missing actor loss")
    try:
        axis[1,0].plot(df['critic_loss'])
        axis[1,0].legend(loc="upper left")
        axis[1,0].set_xlabel("Iteration")
        axis[1,0].set_ylabel("Critic Loss")
    except:
        print("Missing Critic loss")

    axis[1,1].plot(df['values'], "-b")
    axis[1,1].legend(loc="upper left")
    axis[1,1].set_xlabel("Iteration")
    axis[1,1].set_ylabel("Mean Q")

    axis[0,2].plot(df['complete_episodes'])
    axis[0,2].legend(loc="upper left")
    axis[0,2].set_xlabel("Iteration")
    axis[0,2].set_ylabel("Completed episodes")
    
    axis[1,2].plot(calc_percent(df['complete_episodes'].to_numpy()))
    axis[1,2].legend(loc="upper left")
    axis[1,2].set_xlabel("Iteration")
    axis[1,2].set_ylabel("Completed episodes")

    plt.show()

def running_average(x, n = 10):
    mavg = np.zeros_like(x, dtype=np.float32)
    for i in range(1, x.shape[0]+1):
        if i > n:
            mavg[i-1] = np.mean(x[i-n:i])
        else:
            mavg[i-1] = np.mean(x[:i])
    return mavg

def calc_percent(x, n = 100):
    ret = np.zeros_like(x)
    for i in range(x.shape[0]):
        if i > 100:
            complete_episodes = x[i] - x[i-n]
            ret[i] = complete_episodes / n
        else:
            complete_episodes = x[i] - x[0]
            ret[i] = complete_episodes / (i+1)
    return ret

def main():
    # visulize_from_progress_csv("/home/raya/ray_results/DDPG_PickPlaceGrabbedCan_2022-11-21_23-40-413uklke4y/progress.csv")
    visualise_from_custom_progress_file("./results/DDPG-HER-2022-12-18-15-14-06")

if __name__ == "__main__":
    main()