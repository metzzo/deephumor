import argparse

import torch

from evaluation.memory_profile import print_memory
from models.predict_cnn import setup_predict_cnn
from processing.create_debug_set import setup_create_debug_set
from processing.create_tuberlin import setup_create_tuberlin
from processing.pickle_to_csv import setup_pickle_to_csv
from processing.prepare_mnist import setup_prepare_mnist
from train_lstm import setup_train_lstm


def main():
    print_memory()

    from processing.preprocess import setup_preprocess
    from train_cnn import setup_train_cnn

    torch.manual_seed(42)

    use_cuda = torch.cuda.is_available()
    print("Uses CUDA: {0}".format(use_cuda))
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.empty_cache()

    operations = [
        setup_preprocess,
        setup_train_cnn,
        setup_pickle_to_csv,
        setup_create_debug_set,
        setup_prepare_mnist,
        setup_create_tuberlin,
        setup_predict_cnn,
        setup_train_lstm,
    ]

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument("--source", required=True, type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--loss', type=str)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='sgd')


    handlers = []
    for op in operations:
        handlers.append(op(parser=parser, group=group))

    args = parser.parse_args()

    for handler in handlers:
        handler(args=args, device=device)


if __name__ == '__main__':
    main()