import argparse

parser = argparse.ArgumentParser(
    description='Parser for the Evaluation Process', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-c', '--config', default='',
                    help='defining the config file')
parser.add_argument('--dry_run', default=False,
                    help='define if the current execution be a dry run')
parser.add_argument('--wandb', default=False, help='enable wandb monitoring')

arguments = parser.parse_args()
arguments = vars(arguments)

config_file = arguments['config']
dry_run: bool = arguments['dry_run']
include_wandb: bool = arguments['wandb']


if '__main__' == __name__:
    print(arguments)
    