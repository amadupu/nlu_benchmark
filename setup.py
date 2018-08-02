import utils
import os

if __name__ == '__main__':

    # initialize directory structure

    os.makedirs(r'records/eval',exist_ok=True)
    os.makedirs(r'records/train',exist_ok=True)
    os.makedirs(r'model',exist_ok=True)
    os.makedirs(r'logs/train',exist_ok=True)
    os.makedirs(r'logs/eval', exist_ok=True)
    utils.clean_dir('records')
    utils.clean_dir('logs')

