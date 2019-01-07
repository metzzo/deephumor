import argparse


def main():
    from processing.preprocess import setup_preprocess

    operations = [
        setup_preprocess,
    ]

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    handlers = []
    for op in operations:
        handlers.append(op(parser=parser, group=group))

    args = parser.parse_args()

    for handler in handlers:
        handler(args=args)


if __name__ == '__main__':
    main()