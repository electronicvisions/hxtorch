"""
Test Yin-Yang dataset
"""
import unittest

from hxtorch.snn.datasets.yinyang import YinYangDataset


class YinYangDatasetTest(unittest.TestCase):
    """ Test YinYangDataset class """

    def test_item_dimensions(self):
        """ Test dataset and data dimension """
        dataset = YinYangDataset(size=5)
        for sample, _ in dataset:
            self.assertSequenceEqual(sample.shape, [4])
        self.assertEqual(len(dataset), 5)

    def test_sum(self):
        """
        Test that 0 and 2 (as well as 1 and 3) component of
        sample sum up to 1, since sample should be (x, y, 1-x, 1-y).
        """
        dataset = YinYangDataset(size=1)
        for sample, _ in dataset:
            self.assertEqual(sample[0]+sample[2], 1.)
            self.assertEqual(sample[1]+sample[3], 1.)

    def test_type(self):
        """ Test sample type is float and target type is int """
        dataset = YinYangDataset(size=1)
        for sample, target in dataset:
            for coord in sample:
                self.assertIsInstance(coord, float)
            self.assertIsInstance(target, int)


if __name__ == '__main__':
    unittest.main()
