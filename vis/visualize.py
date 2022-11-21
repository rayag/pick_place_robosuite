import pandas as pd
import matplotlib.pyplot as plt

def visulize_from_progress_csv(path):
    df = pd.read_csv(path)
    filtered = df[df['episode_reward_mean'].notnull()]
    figure, axis = plt.subplots(2, 2, figsize=(10, 8))
    axis[0,0].plot(filtered['episode_reward_mean'], "-b", label="mean")
    axis[0,0].legend(loc="upper left")
    axis[0,0].set_xlabel("Epoch")
    axis[0,0].set_ylabel("Reward")
    axis[0,0].set_title(path)
    print(df.keys())

    try:
        axis[0,1].plot(df['info/learner/default_policy/learner_stats/actor_loss'])
        axis[0,1].legend(loc="upper left")
        axis[0,1].set_xlabel("Epoch")
        axis[0,1].set_ylabel("Actor Loss")
    except:
        print("Missing actor loss")
    try:
        axis[1,0].plot(df['info/learner/default_policy/learner_stats/critic_loss'])
        axis[1,0].legend(loc="upper left")
        axis[1,0].set_xlabel("Epoch")
        axis[1,0].set_ylabel("Critic Loss")
    except:
        print("Missing Critic loss")
    plt.show()

def main():
    visulize_from_progress_csv("/home/raya/ray_results/DDPG_PickPlaceGrabbedCan_2022-11-21_16-12-59h9rgz9gp/progress.csv")

if __name__ == "__main__":
    main()