import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visulize_from_progress_csv(path):
    df = pd.read_csv(path)
    filtered = df[df['episode_reward_mean'].notnull()]
    figure, axis = plt.subplots(2, 2, figsize=(10, 8))
    axis[0,0].plot(filtered['episode_reward_mean'], "-b", label="mean")
    axis[0,0].legend(loc="upper left")
    axis[0,0].set_xlabel("Iteration")
    axis[0,0].set_ylabel("Reward")
    axis[0,0].set_title(path)
    print(df.keys())

    try:
        axis[0,1].plot(df['training_iteration'], df['info/learner/default_policy/learner_stats/actor_loss'])
        axis[0,1].legend(loc="upper left")
        axis[0,1].set_xlabel("Iteration")
        axis[0,1].set_ylabel("Actor Loss")
    except:
        print("Missing actor loss")
    try:
        axis[1,0].plot(df['training_iteration'], df['info/learner/default_policy/learner_stats/critic_loss'])
        axis[1,0].legend(loc="upper left")
        axis[1,0].set_xlabel("Iteration")
        axis[1,0].set_ylabel("Critic Loss")
    except:
        print("Missing Critic loss")
    plt.show()

def visualise_from_custom_progress_file(path):
    df = pd.read_csv(path)
    iterations = df.size
    figure, axis = plt.subplots(2, 2, figsize=(10, 8))
    axis[0,0].plot(df['returns'], "-b")
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
    
    plt.show()

def main():
    # visulize_from_progress_csv("/home/raya/ray_results/DDPG_PickPlaceGrabbedCan_2022-11-21_23-40-413uklke4y/progress.csv")
    visualise_from_custom_progress_file("/home/rayageorgieva/uni/results/DDPG-2022-12-03-15-53-35/progress.csv")

if __name__ == "__main__":
    main()