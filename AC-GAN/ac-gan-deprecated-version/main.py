import os, sys
import yaml
import shutil
from argparse import ArgumentParser
from train import ACGANTrain

config_file = 'config.yaml'


def make_dirs(run_path):
    if os.path.exists(run_path):
        shutil.rmtree(run_path)
    os.makedirs(run_path, exist_ok=True)
    os.makedirs(os.path.join(run_path, 'images'))
    os.makedirs(os.path.join(run_path, 'ckpt'))


def main(override):
    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
    run_path = config['run_path']

    if os.path.exists(run_path) and not override:
        print("Run name exists! Please choose another name.")
        return

    make_dirs(run_path)
    shutil.copyfile(config_file, os.path.join(run_path, config_file))

    trainer = ACGANTrain(config)
    trainer.train()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', action='store_true', help="Override run name.")
    sys.argv = ['main.py', '-r']
    args = parser.parse_args()
    main(args.r)
