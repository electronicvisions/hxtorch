import unittest
import os

import hxtorch
from hxtorch.examples import minimal


class MinimalExampleTest(unittest.TestCase):
    """
    Tests the minimal example.
    """

    def test_main(self) -> None:
        """
        Run the example.
        """
        minimal.main()


if __name__ == "__main__":
    unittest.main()
