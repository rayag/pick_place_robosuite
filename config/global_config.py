import argparse

class GlobalConfig:
    _instance = None

    def __init__(self, args: argparse.Namespace) -> None:
        if GlobalConfig._instance is None:
            self.results_dir = args.results_dir
            self.demo_dir = args.demo_dir
            self.experiment = args.experiment
            self.checkpoint_dir = args.checkpoint
            self.action = args.action

            GlobalConfig._instance = self

    @staticmethod
    def getInstance():
        return GlobalConfig._instance