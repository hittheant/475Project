from argparse import ArgumentParser
from load_data import load_data
from models import MaskClassifier

if __name__ == '__main__':
    parser = ArgumentParser(description='Face mask detecting model trainer')
    parser.add_argument('--run_name', type=str, default='debug', help='name of the current run (where runs are saved)')
    parser.add_argument('--data_dir', type=str, default='./archive', help='name of the data directory')
    parser.add_argument('--target_size', type=tuple, default=(600, 600), help='target size of data images')
    parser.add_argument('--faces', type=bool, default=False, help='without bounding boxes')

    args = parser.parse_args()

    (x_train, x_test, y_train, y_test) = load_data(args.data_dir, args.target_size, faces=True)

    model = MaskClassifier()
    model.train(x_train, y_train, x_test, y_test)
