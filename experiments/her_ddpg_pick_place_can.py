import ray
import ray.rllib.agents.ddpg as DDPG
from ray.tune.logger import pretty_print

from callbacks.her_callbacks import HERCallbacks
from environment.pick_place_can_grabbed import PickPlaceGrabbedCan
from environment.pick_place_wrapper import PICK_PLACE_DEFAULT_ENV_CFG

def train():
    ray.shutdown()
    ray.init(
        num_cpus=8,
        num_gpus=1,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    config = DDPG.DEFAULT_CONFIG.copy()
    config['env'] = "Pendulum-v1"
    config['framework'] = "torch"
    config['twin_q'] = True
    config["num_gpus"] = 1
    config["num_workers"] = 2
    config["_fake_gpus"] = False
    config["horizon"] = 10
    config["learning_starts"] = 0
    config["evaluation_duration"] = 20
    config["input"] = {
        "sampler": 1
    }
    config["callbacks"] = HERCallbacks

    env_cfg = PICK_PLACE_DEFAULT_ENV_CFG
    # config['env_config'] = env_cfg
    algo = DDPG.DDPGTrainer(config=config)
    for i in range(10000):
        result = algo.train()
        if i % 20 == 0:
            print(pretty_print(result))
        if i % 100 == 0:
            checkpoint = algo.save()
            print("checkpoint saved at", checkpoint)
            print()
    checkpoint = algo.save()
    print("Last checkpoint saved at", checkpoint)

def main():
    train()

if __name__ == "__main__":
    main()