import argparse

from evaluation.memory_profile import print_memory


def main():
    print_memory()

    from processing.preprocess import setup_preprocess
    from train import setup_train

    operations = [
        setup_preprocess,
        setup_train
    ]

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument("--source", required=True, type=str)

    handlers = []
    for op in operations:
        handlers.append(op(parser=parser, group=group))

    args = parser.parse_args()

    for handler in handlers:
        handler(args=args)


if __name__ == '__main__':
    main()