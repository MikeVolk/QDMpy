import sys
import os
import numpy as np
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

sys.path.append(Path(os.path.dirname(__file__)).parent / "src")

from QDMpy._core.convert import b111_to_bxyz, get_unit_vector


class TestConversion(TestCase):
    def test_b111_to_bxyz(self) -> None:
        """
        A unit test for the b111_to_bxyz function.
        """
        # Define inputs and outputs for the function
        bmap = np.array([[1, 2], [3, 4]])
        pixel_size_in_m = 1.175e-06
        rotation_angle_in_degrees = 30
        direction_vector = None
        expected_output = np.array(
            [
                [[1.29063762, 1.74166891], [2.35723717, 2.19226836]],
                [[-0.25, 0.4330127], [-0.4330127, 0.25]],
                [[0.47505438, 0.47505438], [0.3588714, 0.3588714]],
            ]
        )
        # Call the function
        output = b111_to_bxyz(bmap, pixel_size_in_m, rotation_angle_in_degrees, direction_vector)

        # Check if the output is the same as expected output
        np.testing.assert_allclose(output, expected_output, rtol=1e-8)


def test_get_unit_vector():
    rotation_angle = np.pi / 4  # 45 degree
    direction_vector = np.array([0.5, 0.5, 0.5])  # 111 direction
    expected_output = np.array([0.70710678, 0.29289322, 0.64644661])
    assert np.allclose(
        get_unit_vector(rotation_angle, direction_vector), expected_output, atol=1e-8
    )

    rotation_angle = np.pi / 2  # 90 degree
    direction_vector = np.array([1, 0, 0])  # x-axis direction
    expected_output = np.array([0, -1, 0])
    assert np.allclose(
        get_unit_vector(rotation_angle, direction_vector), expected_output, atol=1e-8
    )
