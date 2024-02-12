
import argparse

def get_args():
    parser = argparse.ArgumentParser('T2S')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--epoch', default=1, type=int)
    return parser.parse_args()
