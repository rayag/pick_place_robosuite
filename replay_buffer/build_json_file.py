import h5py
import numpy as np
import os
import robosuite as suite

import ray._private.utils

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

from environment.pick_place_wrapper import PickPlaceWrapper, PICK_PLACE_DEFAULT_ENV_CFG
from replay_buffer.simple_replay_buffer import DEMO_PATH

if __name__ == "__main__":
    '''
    Code source: https://docs.ray.io/en/latest/rllib/rllib-offline.html#example-converting-external-experiences-to-batch-format
    '''
    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(
        os.path.join("/home/raya/uni/ray_test/data/", "demo-pick-only")
    )

    ctr_cfg = suite.load_controller_config(default_controller="OSC_POSE")
    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    env_cfg['controller_configs'] = ctr_cfg
    env_cfg['pick_only'] = True
    env = PickPlaceWrapper()

    # RLlib uses preprocessors to implement transforms such as one-hot encoding
    # and flattening of tuple and dict observations.
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    print("The preprocessor is", prep)
    with h5py.File(DEMO_PATH, "r+") as f:
        demos = list(f['data'].keys())
        print(f"Total episodes {len(demos)}")
        for i in range(len(demos)):
            print(f"Processing episode {i}...")
            ep = demos[i]
            eps_id = int(demos[i][5:])
            states = f["data/{}/states".format(ep)][()]
            acts = f["data/{}/actions".format(ep)][()]
            obs = env.reset_to(states[0])
            prev_action = acts[0]
            prev_reward = 0
            done = False
            t = 0
            while not done and t < states.shape[0]:
                action = acts[t]
                new_obs, rew, done, info = env.step(action)
                batch_builder.add_values(
                    t=t,
                    eps_id=eps_id,
                    agent_index=0,
                    obs=prep.transform(obs),
                    actions=action,
                    rewards=rew,
                    # prev_actions=prev_action,
                    # prev_rewards=prev_reward,
                    dones=done,
                    infos=info,
                    new_obs=prep.transform(new_obs),
                    weights=1
                )
                obs = new_obs
                prev_action = action
                prev_reward = rew
                t += 1
        writer.write(batch_builder.build_and_reset())