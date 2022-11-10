import pandas as pd
import matplotlib.pyplot as plt

def visulize_from_progress_csv(path):
    df = pd.read_csv(path)
    filtered = df[df['episode_reward_mean'].notnull()]
    plt.plot(filtered['episode_reward_mean'], "-b", label="mean")
    plt.plot(filtered['episode_reward_max'], "-r", label="max")
    plt.legend(loc="upper left")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.show()