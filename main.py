import argparse
from config.global_config import GlobalConfig
from config.config_runner import ConfigRunner

def main():
    parser = argparse.ArgumentParser(
        prog = 'PickPlaceCanRobosuiteWithRl',
        description='Part of the masters thesis of Raya Georgieva'
    )

    parser.add_argument('-r', '--results-dir', default='./results', 
        help='Directory which will hold the results of the experiments')
    parser.add_argument('-dm', '--demo-dir', default='./demo',
        help='Directory holding data from demos')
    parser.add_argument('-e', '--experiment', choices=['ddpg-pick', 'ddpg-per-pick'],
        help='Experiment to run')
    parser.add_argument('-chkp', '--checkpoint', default=None,
        help='Checkpoint dir to load model from, should contain *.pth files')
    parser.add_argument('-a', '--action', choices=['train', 'rollout', 'rollout-helper', 'vis', 'eval'], required=True,
        help='Specifies whether the program will train or rollout agent')

    global_cfg = GlobalConfig(parser.parse_args())
    ConfigRunner.run(global_cfg)
    
if __name__ == "__main__":
    main()