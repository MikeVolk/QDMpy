import numpy as np
import pytest

from QDMpy._core.odmr import ODMR


@pytest.fixture
def odmr_data():
    data_array = np.random.rand(2, 2, 100, 100)
    scan_dimensions = np.array([10, 10])
    frequency_array = np.linspace(0, 10, 100)

    return ODMR(data_array, scan_dimensions, frequency_array)


def test_odmr_getitem_positive_polarization(odmr_data):
    data = odmr_data["+"]
    expected_data = odmr_data._raw_data[0]
    np.testing.assert_array_equal(data, expected_data)


def test_odmr_getitem_negative_polarization(odmr_data):
    data = odmr_data["-"]
    expected_data = odmr_data._raw_data[1]
    np.testing.assert_array_equal(data, expected_data)


def test_odmr_getitem_low_freq_range(odmr_data):
    data = odmr_data["<"]
    expected_data = odmr_data._raw_data[:, 0]
    np.testing.assert_array_equal(data, expected_data)


def test_odmr_getitem_high_freq_range(odmr_data):
    data = odmr_data[">"]
    expected_data = odmr_data._raw_data[:, 1]
    np.testing.assert_array_equal(data, expected_data)


def test_odmr_getitem_reshape(odmr_data):
    data = odmr_data["r"]
    expected_data = odmr_data._raw_data.reshape(
        odmr_data.n_pol, odmr_data.n_frange, *odmr_data._img_shape, odmr_data.n_freqs
    )
    np.testing.assert_array_equal(data, expected_data)


def test_odmr_getitem_specific_frequency(odmr_data):
    data = odmr_data["f1.0"]
    freq_idx = np.argmin(np.abs(odmr_data._frequencies - 1.0))
    expected_data = odmr_data._raw_data[..., freq_idx]
    np.testing.assert_array_equal(data, expected_data)


def test_odmr_getitem_numpy_slicing(odmr_data):
    data = odmr_data[:, :, 0:10]
    expected_data = odmr_data._raw_data[..., 0:10]
    np.testing.assert_array_equal(data, expected_data)
