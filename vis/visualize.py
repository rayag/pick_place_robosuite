import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def visualise_from_custom_progress_file(path):
    path = os.path.join(path, 'progress.csv')
    df = pd.read_csv(path)
    iterations = df.size
    figure, axis = plt.subplots(2, 3, figsize=(18, 8))
    
    axis[0,0].plot(df['returns'], "-b", label="Raw")
    axis[0,0].plot(running_average(df['returns'].to_numpy(), n=10), '-.', color='red', label="Mean")
    axis[0,0].legend(loc="upper left")
    axis[0,0].set_xlabel("Iteration")
    axis[0,0].set_ylabel("Reward")
    axis[0,0].set_title(path)
    print(df.keys())

    try:
        axis[0.1].set_axisbelow(True)
        axis[0,1].yaxis.grid(color='#E8E8E8', linestyle='dashed')
        axis[0,1].xaxis.grid(color='#E8E8E8', linestyle='dashed')
        axis[0,1].plot(df['actor_loss'])
        axis[0,1].legend(loc="upper left")
        axis[0,1].set_xlabel("Итерация")
        axis[0,1].set_ylabel("Грешка на актьора")
    except:
        print("Missing actor loss")

    try:
        ax = axis[1,0]
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='#E8E8E8', linestyle='dashed')
        ax.xaxis.grid(color='#E8E8E8', linestyle='dashed')
        ax.plot(df['critic_loss'])
        ax.legend(loc="upper left")
        ax.set_xlabel("Итерация")
        ax.set_ylabel("Грешка на критика")
    except:
        print("Missing Critic loss")

    try:
        axis[1,1].plot(df['values'], "-b")
        axis[1,1].legend(loc="upper left")
        axis[1,1].set_xlabel("Iteration")
        axis[1,1].set_ylabel("Mean Q")
    except:
        print("Missing value")

    try:
        axis[0,2].plot(df['complete_episodes'])
        axis[0,2].legend(loc="upper left")
        axis[0,2].set_xlabel("Iteration")
        axis[0,2].set_ylabel("Completed episodes")
        
        axis[1,2].plot(calc_percent(df['complete_episodes'].to_numpy()))
        axis[1,2].legend(loc="upper left")
        axis[1,2].set_xlabel("Iteration")
        axis[1,2].set_ylabel("Completed episodes")
    except:
        print("Missing complete episodes")
    
    plt.show()

def visualise_her_results(path):
    progress = os.path.join(path, 'progress.csv')
    epoch = os.path.join(path, "epoch.csv")
    df = pd.read_csv(progress)
    df_epoch = pd.read_csv(epoch)
    iterations = df.size
    epochs = df_epoch.size
    fig, axis = plt.subplots(2, 3, figsize=(15, 8))
    fig.subplots_adjust(right=0.9, left=0.1, top=0.9, bottom=0.1, wspace=0.4)
    axis[0,0].grid(color='#E8E8E8', linestyle='dashed')
    axis[0,0].plot(df_epoch['success_rate'] * 100, "-b")
    axis[0,0].plot(running_average(df_epoch['success_rate'].to_numpy() * 100, n=10), '-.', color='red', label="Mean")
    axis[0,0].set_xlabel("Епоха")
    axis[0,0].set_ylabel("Успеваемост %")
    # axis[0,0].set_title(path)

    try:
        axis[0,1].grid(color='#E8E8E8', linestyle='dashed')
        axis[0,1].plot(df['actor_loss'], color="#85C1E9")
        axis[0,1].legend(loc="upper left")
        axis[0,1].set_xlabel("Итерация")
        axis[0,1].set_ylabel("Грешка на актьора")
    except:
        print("Missing actor loss")
    try:
        axis[1,0].plot(df['critic_loss'])
        axis[1,0].legend(loc="upper left")
        axis[1,0].set_xlabel("Iteration")
        axis[1,0].set_ylabel("Critic Loss")
    except:
        print("Missing Critic loss")

    try:
        axis[1,1].plot(df['values'], "-b")
        axis[1,1].legend(loc="upper left")
        axis[1,1].set_xlabel("Iteration")
        axis[1,1].set_ylabel("Mean Q")
    except:
        print("Missing value")

    # axis[0,2].plot(df['complete_episodes'])
    # axis[0,2].legend(loc="upper left")
    # axis[0,2].set_xlabel("Iteration")
    # axis[0,2].set_ylabel("Completed episodes")

    axis[0,2].grid(color='#E8E8E8', linestyle='dashed')
    axis[0,2].plot(df['critic_loss'], color="#85C1E9")
    axis[0,2].legend(loc="upper left")
    axis[0,2].set_xlabel("Итерация")
    axis[0,2].set_ylabel("Грешка на критика")
    
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

def vis_comparison_epoch(lhs_path, rhs_path):
    lhs_epoch = pd.read_csv(os.path.join(lhs_path, "epoch.csv"))
    rhs_epoch = pd.read_csv(os.path.join(rhs_path, 'epoch.csv'))
    plt.plot(lhs_epoch['success_rate'] * 100, color="#85C1E9", label="k=4", linestyle='dashed')
    plt.plot(rhs_epoch['success_rate']  * 100, color="#E67E22", label="k=8")
    plt.grid(color='#E8E8E8', linestyle='dashed')
    plt.legend(loc="lower right")
    plt.ylabel("Успеваемост %")
    plt.xlabel("Епоха")
    plt.title("Сравнение на различни стойности на k")
    plt.show()

def vis_comparison_progress(lhs_path, rhs_path):
    lhs_epoch = pd.read_csv(os.path.join(lhs_path, "progress.csv"))
    rhs_epoch = pd.read_csv(os.path.join(rhs_path, 'progress.csv'))
    xs = np.array([x if x > -150 else x + 50 for x in rhs_epoch['returns']])
    xs = np.array([x if x < 150 else x - 50 for x in xs])
    plt.plot(xs, color="#85C1E9", label="Награда")
    plt.plot(running_average(xs), color="#E67E22", label="Пълзящо средно")
    plt.grid(color='#E8E8E8', linestyle='dashed')
    plt.legend(loc="upper left")
    plt.ylabel("Награда")
    plt.xlabel("Итерация")
    plt.title("Резултати от продължително трениране на DDPG+PER")
    plt.show()

def vis_reward():
    def f(x):
        if x < 0.07:
            return x
        else:
            return x * 100

    def f2(x):
        return 20 * x - 1
    
    xs = np.arange(0, 0.1, 0.0001)
    rewards = np.array([f(x) for x in xs])
    plt.plot(xs, rewards, color="#E67E22", label="награда f1")
    plt.plot(xs, xs, color="#85C1E9", linestyle='dashed', label="оригинална награда")
    plt.ylabel("Стойност на наградата")
    plt.xlabel("Стойност на наградата без промяна")
    plt.title("Сравнение между графиките на наградите")
    plt.grid(color='#E8E8E8', linestyle='dashed')
    plt.legend(loc="upper left")
    plt.show()

def main():
    # visulize_from_progress_csv("/home/raya/ray_results/DDPG_PickPlaceGrabbedCan_2022-11-21_23-40-413uklke4y/progress.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='directory of the progress file') 
    args = parser.parse_args()
    visualise_her_results(args.d)
    # vis_comparison_epoch("/home/rayageorgieva/uni/masters/pick_place_robosuite/results/DDPG-HER-2023-01-07-18-38-56/", "/home/rayageorgieva/uni/masters/pick_place_robosuite/results/DDPG-HER-reach-k-8-2023-02-08-11-12-25")
    # vis_comparison_progress("~/uni/results/tmp/ddpg-1/", "/home/rayageorgieva/uni/masters/pick_place_robosuite/results/DDPG--2023-02-08-09-11-57")
    # vis_comparison_progress("~/uni/results/tmp/ddpg-1/", "/home/rayageorgieva/uni/results/DDPG--2022-12-09-01-47-19")
    # vis_reward()

if __name__ == "__main__":
    main()