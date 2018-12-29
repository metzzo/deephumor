
import numpy as np

from typing import List, Callable

from skimage.transform import resize

# All operations are functions that take and return numpy arrays
# See https://docs.python.org/3/library/typing.html#typing.Callable for what this line means
from torch.utils.data import Dataset

Op = Callable[[np.ndarray], np.ndarray]

def chain(ops: List[Op]) -> Op:
    '''
    Chain a list of operations together.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        for op_ in ops:
            sample = op_(sample)
        return sample

    return op


def type_cast(dtype: np.dtype) -> Op:
    '''
    Cast numpy arrays to the given type.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample.astype(dtype)

    return op


def vectorize() -> Op:
    '''
    Vectorize numpy arrays via "numpy.ravel()".
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.ravel(sample)

    return op


def hwc2chw() -> Op:
    '''
    Flip a 3D array with shape HWC to shape CHW.
    '''

    def op(arr: np.ndarray) -> np.ndarray:
        return np.transpose(arr, [2, 0, 1])

    return op


def chw2hwc() -> Op:
    '''
    Flip a 3D array with shape CHW to HWC.
    '''

    def op(arr: np.ndarray) -> np.ndarray:
        return np.transpose(arr, [1, 2, 0])

    return op


def add(val: float) -> Op:
    '''
    Add a scalar value to all array elements.
    '''

    def op(arr: np.ndarray) -> np.ndarray:
        return arr + val

    return op


def mul(val: float) -> Op:
    '''
    Multiply all array elements by the given scalar.
    '''

    def op(arr: np.ndarray) -> np.ndarray:
        return arr * val

    return op


def hflip() -> Op:
    '''
    Flip arrays with shape HWC horizontally with a probability of 0.5.
    '''

    def op(arr: np.ndarray) -> np.ndarray:
        if np.random.uniform(0.0, 1.0) < 0.5:
            return np.flip(arr, axis=1)
        else:
            return arr

    return op


def rcrop(sz: int, pad: int, pad_mode: str) -> Op:
    '''
    Extract a square random crop of size sz from arrays with shape HWC.
    If pad is > 0, the array is first padded by pad pixels along the top, left, bottom, and right.
    How padding is done is governed by pad_mode, which should work exactly as the 'mode' argument of numpy.pad.
    Raises ValueError if sz exceeds the array width/height after padding.
    '''

    def op(arr: np.ndarray) -> np.ndarray:
        if np.random.uniform(0.0, 1.0) < 0.75:
            return arr

        #skimage.io.imshow(arr)
        #skimage.io.show()

        if pad > 0:
            arr = np.pad(arr, pad_width=((pad, pad), (pad, pad), (0, 0)), mode=pad_mode)

        if sz > arr.shape[0] or sz > arr.shape[1]:
            raise ValueError()

        pos_x = int(np.random.uniform(0.0, arr.shape[0] - sz))
        pos_y = int(np.random.uniform(0.0, arr.shape[1] - sz))

        arr = arr[pos_x:pos_x + sz, pos_y:pos_y + sz]

        resized = np.array(resize(arr, (32, 32, 3), anti_aliasing=False))
        #skimage.io.imshow(resized)
        #skimage.io.show()
        #resized = np.array(resized)

        return resized

    return op
