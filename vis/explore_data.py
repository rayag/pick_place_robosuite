import numpy as np
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os
import pandas as pd

from environment.pick_place_wrapper import PickPlaceWrapper, PICK_PLACE_DEFAULT_ENV_CFG, Task
from replay_buffer.normalizer import Normalizer

# DEMO_PATH = "/home/rayageorgieva/uni/masters/pick_place_robosuite/demo/low_dim.hdf5"
DEMO_PATH = "/home/rayageorgieva/uni/masters/pick_place_robosuite/demo/mh/low_dim.hdf5"

def inspect_env_data(visualize = False):
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['pick_only'] = True
    env_cfg['has_renderer'] = visualize
    env = PickPlaceWrapper()
    normalizer = Normalizer(env.obs_dim)
    with h5py.File(os.path.join("/home/rayageorgieva/uni/masters/pick_place_robosuite/results/DDPG-HER-2023-01-07-18-38-56/checkpoint_000135", 'normalizer_data.h5'), 'r') as f:
        normalizer.set_mean_std(f['obs_norm_mean'][()], f['obs_norm_std'][()])

    with h5py.File(DEMO_PATH, "r+") as f:
        demos = list(f['data'].keys())
        sum_steps = 0
        actions = np.empty(shape=(0, env.action_dim))
        observations = np.empty(shape=(0, env.obs_dim))

        for i in range(len(demos)):
            ep = demos[i]
            ep_obs = f["data/{}/obs_flat".format(ep)][()]
            ep_actions = f["data/{}/actions".format(ep)][()]
            actions = np.concatenate((actions, ep_actions))
            observations = np.concatenate((observations, ep_obs))

        observations_norm = normalizer.normalize(observations)
        print(f"OBS min: {np.min(observations, axis=0)}")
        print(f"OBS max: {np.max(observations, axis=0)}")
        print(f"OBS mean: {np.mean(observations, axis=0)}")
        print(f"OBS norm min: {np.min(observations_norm, axis=0)}")
        print(f"OBS norm max: {np.max(observations_norm, axis=0)}")
        print(f"OBS norm mean: {np.mean(observations_norm, axis=0)}")
        print()
        print(f"ACTION min: {np.min(actions, axis=0)}")
        print(f"ACTION max: {np.max(actions, axis=0)}")
        print(f"ACTION mean: {np.mean(actions, axis=0)}")
        print(f"ACTION std: {np.std(actions, axis=0)}")

        figure, axis = plt.subplots(2, 3, figsize=(16, 7))
        ax = axis[0,0]
        index = 35
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='#E8E8E8', linestyle='dashed')
        ax.xaxis.grid(color='#E8E8E8', linestyle='dashed')
        ax.set_facecolor(color='white')
        ax.hist(observations_norm[:, index], bins=20, color="#FFCC9999", rwidth=0.9)
        ax.set_xlabel('Стойност')
        ax.set_ylabel('Брой')
        ax.set_title('С нормализация X')
        ax.axvline(observations_norm[:, index].mean(), color='#CA6F1E', linestyle='dashed', linewidth=1)

        ax = axis[1,0]
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='#E8E8E8', linestyle='dashed')
        ax.xaxis.grid(color='#E8E8E8', linestyle='dashed')
        ax.set_facecolor(color='white')
        ax.hist(observations[:, index], bins=20, color="#70EBE299", rwidth=0.9, range=[-2,2])
        ax.set_xlabel('Стойност')
        ax.set_ylabel('Брой')
        ax.set_title('Без нормализация X')
        ax.axvline(observations[:, index].mean(), color='#21618C', linestyle='dashed', linewidth=1)

        ax = axis[0,1]
        index = 0
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='#E8E8E8', linestyle='dashed')
        ax.xaxis.grid(color='#E8E8E8', linestyle='dashed')
        ax.set_facecolor(color='white')
        ax.hist(observations_norm[:, index], bins=20, color="#FFCC9999", rwidth=0.9)
        ax.set_xlabel('Стойност')
        ax.set_ylabel('Брой')
        ax.set_title('С нормализация Y')
        ax.axvline(observations_norm[:, index].mean(), color='#CA6F1E', linestyle='dashed', linewidth=1)

        ax = axis[1,1]
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='#E8E8E8', linestyle='dashed')
        ax.xaxis.grid(color='#E8E8E8', linestyle='dashed')
        ax.set_facecolor(color='white')
        ax.hist(observations[:, index], bins=20, color="#70EBE299", rwidth=0.9, range=[-0.6, 0.5])
        ax.set_xlabel('Стойност')
        ax.set_ylabel('Брой')
        ax.set_title('Без нормализация Y')
        ax.axvline(observations[:, index].mean(), color='#21618C', linestyle='dashed', linewidth=1)

        ax = axis[0,2]
        index = 1
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='#E8E8E8', linestyle='dashed')
        ax.xaxis.grid(color='#E8E8E8', linestyle='dashed')
        ax.set_facecolor(color='white')
        ax.hist(observations_norm[:, index], bins=20, color="#FFCC9999", rwidth=0.9)
        ax.set_xlabel('Стойност')
        ax.set_ylabel('Брой')
        ax.set_title('С нормализация Z')
        ax.axvline(observations_norm[:, index].mean(), color='#CA6F1E', linestyle='dashed', linewidth=1)

        ax = axis[1,2]
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='#E8E8E8', linestyle='dashed')
        ax.xaxis.grid(color='#E8E8E8', linestyle='dashed')
        ax.set_facecolor(color='white')
        ax.hist(observations[:, index], bins=20, color="#70EBE299", rwidth=0.9, range=[-1,3])
        ax.set_xlabel('Стойност')
        ax.set_ylabel('Брой')
        ax.set_title('Без нормализация Z')
        ax.axvline(observations[:, index].mean(), color='#21618C', linestyle='dashed', linewidth=1)
        plt.show()

def explore_demos_steps():
    with h5py.File(DEMO_PATH, "r+") as f:
        demos = list(f['data'].keys())
        sum_steps = 0
        total_steps = 0
        for i in range(len(demos)):
            ep = demos[i]
            total_steps += len(f["data/{}/actions".format(ep)][()])
        print(f"Total steps {total_steps} Average steps per episode {total_steps / len(demos)}")

def explore_demos_rewards(horizon = 500):
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env = PickPlaceWrapper(task=Task.PICK_AND_PLACE)
    with h5py.File(DEMO_PATH, "r+") as f:
        demos = list(f['data'].keys())
        total_returns = 0
        total_dones = 0
        for i in range(len(demos)):
            ep = demos[i]
            actions = f["data/{}/actions".format(ep)][()]
            state = f["data/{}/states".format(ep)][()][0]
            env.reset_to(state)
            ep_return = 0

            for t in range(horizon):
                if t < len(actions):
                    act = actions[t]
                else:
                    act = np.zeros_like(actions[0])
                _, reward, done, _, _ = env.step(act)
                ep_return += reward
            total_returns += ep_return
            total_dones += 1
        print(f"Average retrun per episode {total_returns / len(demos)} success rate {total_dones / len(demos)}")

def plot():
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['has_renderer'] = True
    env_cfg['render_camera'] = "agentview"
    env = PickPlaceWrapper()
    with h5py.File(DEMO_PATH, "r+") as f:
        demos = list(f['data'].keys())
        ep = demos[0]
        actions = f["data/{}/actions".format(ep)][()]
        states = f["data/{}/states".format(ep)][()]
        env.reset_to(states[0])
        tmp = np.ones_like(actions[:,0])
        env.render()
        plt.hist2d(actions[:,1], actions[:, 0], color = 'blue', edgecolor = 'black',
        bins = int(180/5))
        plt.gca().invert_yaxis()
        plt.show()

def gather_initial_position():
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env = PickPlaceWrapper(env_config=env_cfg)
    xs = []
    ys = []
    for i in range(5000):
        obs, _ = env.reset()
        pos = env.extract_can_pos_from_obs(obs)
        xs.append(pos[0])
        ys.append(pos[1])
        if i % 100 == 0:
            print(f"i = {i}")
    df = pd.DataFrame(list(zip(xs, ys)), columns=['x', 'y'])
    df.to_csv("./results/init.csv")

def vis_initial_pos():
    # x = []
    # y=[]
    # with open('./data/init.csv') as file:
    #     for line in file:
    #         w = line.rstrip().split('\t')
    #         x.append(w[1])
    #         y.append(w[2])
    # df = pd.DataFrame(list(zip(x, y)), columns=['x', 'y'])
    # df.to_csv("./results/init.csv")

    df = pd.read_csv("./results/init.csv")
    print(df.columns)
    _, _, _, mesh = plt.hist2d(df['x'], df['y'], bins=[50,50], range=np.array([(-0.1, 0.25), (-0.45, -0.02)]))
    plt.colorbar(mesh)
    plt.title('Разпределение на началната позиция на предмета', fontsize=11, pad=10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def main():
    # vis_initial_pos()
    # inspect_env_data()
    explore_demos_rewards()

if __name__ == "__main__":
    main()