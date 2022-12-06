import pandas as pd
import matplotlib.pyplot as plt
import os

def visualise_from_custom_progress_file(path):
    path = os.path.join(path, 'progress.csv')
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

    axis[1,1].plot(df['values'], "-b")
    axis[1,1].legend(loc="upper left")
    axis[1,1].set_xlabel("Iteration")
    axis[1,1].set_ylabel("Mean Q")
    
    plt.show()

def main():
    # visulize_from_progress_csv("/home/raya/ray_results/DDPG_PickPlaceGrabbedCan_2022-11-21_23-40-413uklke4y/progress.csv")
    visualise_from_custom_progress_file("./results/DDPG-HER-2022-12-06-10-47-28")

if __name__ == "__main__":
    main()