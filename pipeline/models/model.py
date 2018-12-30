import numpy as np

from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    '''
    A machine learning model.
    '''

    @abstractmethod
    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple.
        The first value is always the number of input samples.
        If this value is 0, an arbitrary number of input samples is supported.
        '''

        pass

    @abstractmethod
    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple.
        '''

        pass

    @abstractmethod
    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data are the input data, with a shape compatible with input_shape().
        Labels are the corresponding target labels.
        Returns the training loss.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        pass

    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict target labels from input data.
        Data are the input data, with a shape compatible with input_shape().
        The label array has shape (n, output_shape()) with n being the number of input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''