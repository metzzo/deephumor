import argparse

from evaluation.memory_profile import print_memory
from processing.create_debug_set import setup_create_debug_set
from processing.pickle_to_csv import setup_pickle_to_csv
from processing.prepare_mnist import setup_prepare_mnist


def main():
    print_memory()

    from processing.preprocess import setup_preprocess
    from train import setup_train

    operations = [
        setup_preprocess,
        setup_train,
        setup_pickle_to_csv,
        setup_create_debug_set,
        setup_prepare_mnist
    ]

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument("--source", required=True, type=str)
    parser.add_argument("--target", type=str)

    handlers = []
    for op in operations:
        handlers.append(op(parser=parser, group=group))

    args = parser.parse_args()

    for handler in handlers:
        handler(args=args)


if __name__ == '__main__':
    main()