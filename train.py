from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from load_data import load_data

if __name__ == '__main__':
    parser = ArgumentParser(description='Face mask detecting model trainer')
    parser.add_argument('--run_name', type=str, default='debug', help='name of the current run (where runs are saved)')
    parser.add_argument('--data_dir', type=str, default='./archive', help='name of the data directory')
    parser.add_argument('--target_size', type=tuple, default=(600, 600), help='target size of data images')

    args = parser.parse_args()

    data, labels = load_data(args.data_dir, args.target_size)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)