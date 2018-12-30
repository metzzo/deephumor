import cv2

from pipeline.processing.ops import normalize_size
import numpy as np


def test_normalize_size_simple():
    op = normalize_size(32, 32)
    blank_image = np.zeros((64, 64, 3), np.uint8)
    result = op(blank_image)

    assert result.shape == (32, 32, 3)


def test_normalize_size_more_width():
    op = normalize_size(7, 3)

    input_image = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], np.uint8)
    expected_image = np.array([
        [0, 0, 1, 2, 3, 0, 0],
        [0, 0, 4, 5, 6, 0, 0],
        [0, 0, 7, 8, 9, 0, 0],
    ], np.uint8)

    result_image = op(input_image)

    assert (expected_image == result_image).all()


def test_normalize_size_more_height():
    op = normalize_size(3, 7)

    input_image = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], np.uint8)
    expected_image = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [0, 0, 0],
        [0, 0, 0],
    ], np.uint8)

    result_image = op(input_image)

    assert (expected_image == result_image).all()


def test_normalize_site_scale_down():
    op = normalize_size(2, 2)

    input_image = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], np.uint8)
    expected_image = np.array([
        [1, 1],
        [1, 1],
    ], np.uint8)

    result_image = op(input_image)

    assert (expected_image == result_image).all()


def test_normalize_site_scale_up():
    op = normalize_size(3, 3)

    input_image = np.array([
        [1, 1],
        [1, 1],
    ], np.uint8)
    expected_image = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], np.uint8)

    result_image = op(input_image)

    assert (expected_image == result_image).all()


def test_normalize_size_more_height_and_scale_down():
    op = normalize_size(3, 7)

    input_image = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ], np.uint8)
    expected_image = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
    ], np.uint8)

    result_image = op(input_image)

    assert (expected_image == result_image).all()


def test_normalize_size_more_width_and_scale_down():
    op = normalize_size(7, 3)

    input_image = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ], np.uint8)
    expected_image = np.array([
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
    ], np.uint8)

    result_image = op(input_image)

    assert (expected_image == result_image).all()
