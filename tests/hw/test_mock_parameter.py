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


if __name__ == '__main__':
    unittest.main()
