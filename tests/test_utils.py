import pytest
from QDMpy.utils import *
import numpy as np


import numpy as np
from itertools import product


def test_generate_possible_dim():
    b_source = 10.0
    n = 5
    expected_shape = (n * n, 3)
    expected_source_dim = np.array(
        [
            [0.0, -90.0, 10.0],
            [0.0, -45.0, 10.0],
            [0.0, 0.0, 10.0],
            [0.0, 45.0, 10.0],
            [0.0, 90.0, 10.0],
            [90.0, -90.0, 10.0],
            [90.0, -45.0, 10.0],
            [90.0, 0.0, 10.0],
            [90.0, 45.0, 10.0],
            [90.0, 90.0, 10.0],
            [180.0, -90.0, 10.0],
            [180.0, -45.0, 10.0],
            [180.0, 0.0, 10.0],
            [180.0, 45.0, 10.0],
            [180.0, 90.0, 10.0],
            [270.0, -90.0, 10.0],
            [270.0, -45.0, 10.0],
            [270.0, 0.0, 10.0],
            [270.0, 45.0, 10.0],
            [270.0, 90.0, 10.0],
            [360.0, -90.0, 10.0],
            [360.0, -45.0, 10.0],
            [360.0, 0.0, 10.0],
            [360.0, 45.0, 10.0],
            [360.0, 90.0, 10.0],
        ]
    )

    # Verify expected output shape
    assert generate_possible_dim(b_source=b_source, n=n).shape == expected_shape

    # Verify source dim
    assert np.allclose(generate_possible_dim(b_source=b_source, n=n), expected_source_dim)


def test_rms():
    data = [1, 2, 3, 4, 5]
    expected_output = 3.3166247903554
    assert rms(data) == expected_output

    data = [0, 0, 0, 0, 0, 0]
    expected_output = 0
    assert rms(data) == expected_output

    data = [-1, 2, -3, 4, -5]
    expected_output = 3.3166247903554
    assert rms(data) == expected_output

    data = [1.1, 2.2, 3.3, 4.4, 5.5]
    expected_output = 3.6482872693909396
    assert rms(data) == expected_output

    data = [[1.1, 2.2, 3.3, 4.4, 5.5], [1.1, 2.2, 3.3, 4.4, 5.5], [1.1, 2.2, 3.3, 4.4, 5.5]]
    expected_output = 3.6482872693909396
    np.testing.assert_array_almost_equal(rms(data), expected_output)

    data = [data, data]
    np.testing.assert_array_almost_equal(rms(data), expected_output)


def test_rc2idx():
    # test case 1
    shape = (3, 3)
    idx = rc2idx((0, 0), shape)
    assert idx == 0

    # test case 2
    rc = np.array([[2, 3]])
    shape = (4, 5)
    idx = rc2idx(rc, shape)
    assert idx == 13

    # test case 3
    rc = np.array([[2, 3], [4, 3]])
    shape = (4, 5)
    idx_values = [14, 18]  # expected indices
    indices = rc2idx(rc, shape)
    assert np.array_equal(indices, idx_values)

    # test case 4
    rc = np.array([[2, 3], [4, 3], [12, 1]])
    shape = (14, 15)
    idx_values = [33, 63, 181]  # expected indices
    indices = rc2idx(rc, shape)
    assert np.array_equal(indices, idx_values)


# def test_idx2rc():
#     # Test case 1: Single index
#     idx = 5
#     shape = (3, 4)
#     expected_output = np.array([[1], [1]])
#     assert np.array_equal(idx2rc(idx, shape), expected_output)

#     # Test case 2: List of indices
#     idx = [5, 10]
#     shape = (3, 4)
#     expected_output = (np.array([[1], [1]]), np.array([[2], [2]]))
#     assert np.array_equal(idx2rc(idx, shape), expected_output)

#     # Test case 3: Numpy array of indices
#     idx = np.array([5, 10])
#     shape = (3, 4)
#     expected_output = (np.array([[1], [1]]), np.array([[2], [2]]))
#     assert np.array_equal(idx2rc(idx, shape), expected_output)

#     # Test case 4: Index out of bounds
#     idx = 20
#     shape = (3, 4)
#     with pytest.raises(IndexError):
#         idx2rc(idx, shape)
