import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='Main')

    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--snr', type=float, default=10.)

    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--epoch', type=int, default=750,
                        help="total number of epochs")
    parser.add_argument('--load', type=int, default=0,
                        help='load trained model')
    parser.add_argument('--cuda', type=str, default=0,
                        help='gpu cuda device')

    args = parser.parse_args()
    return args
