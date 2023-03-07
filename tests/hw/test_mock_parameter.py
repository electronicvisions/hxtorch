import unittest
import hxtorch

class TestMockParameter(unittest.TestCase):
    """
    Tests the MockParameter struct and its usage.
    """

    def test_init(self):
        default_mock_parameter = hxtorch.MockParameter()
        custom_mock_parameter = hxtorch.MockParameter(
            gain=default_mock_parameter.gain,
            noise_std=default_mock_parameter.noise_std
        )
        self.assertEqual(custom_mock_parameter, default_mock_parameter)

    def test_repr(self):
        mock_parameter = hxtorch.MockParameter()
        self.assertRegex(repr(mock_parameter), r"MockParameter\(.*\)")

    def test_getter_setter(self):
        mock_parameter = hxtorch.MockParameter(gain=2000)
        with self.assertRaises(OverflowError):
            hxtorch.set_mock_parameter(mock_parameter)

        mock_parameter = hxtorch.MockParameter(gain=0.0043)
        hxtorch.set_mock_parameter(mock_parameter)
        self.assertEqual(hxtorch.get_mock_parameter(), mock_parameter)

    def test_measure(self):
        hxtorch.init_hardware(ann=True)
        mock_parameter = hxtorch.measure_mock_parameter()
        hxtorch.release_hardware()
        self.assertGreater(mock_parameter.gain, 0)
        self.assertLessEqual(mock_parameter.gain, 1)


if __name__ == '__main__':
    unittest.main()
