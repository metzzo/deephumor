import torch


def print_memory():
    print('Memory allocated: {0} MB - Memory Cached: {1} MB'.format(
        torch.cuda.memory_allocated() / 1024**2,
        torch.cuda.memory_cached() / 1024 ** 2,
    ))
