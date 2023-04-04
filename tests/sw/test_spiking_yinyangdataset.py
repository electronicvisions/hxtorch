"""
Test Yin-Yang dataset
"""
import unittest

from hxtorch.spiking.datasets.yinyang import YinYangDataset


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

    def test_which_class(self):
        """ Test to which class a sample corresponds """
        dataset = YinYangDataset(size=1, r_small=0.1, r_big=0.5)
        dot_1, dot_2 = (0.26, 0.52), (0.74, 0.48)
        yin, yang = (0.76, 0.34), (0.23, 0.65)
        self.assertEqual(
            dataset.class_names[dataset.which_class(*dot_1)], "dot")
        self.assertEqual(
            dataset.class_names[dataset.which_class(*dot_2)], "dot")
        self.assertEqual(
            dataset.class_names[dataset.which_class(*yin)], "yin")
        self.assertEqual(
            dataset.class_names[dataset.which_class(*yang)], "yang")

    def test_dist_to_right_dot(self):
        """ Test distance calculation to right dot """
        dataset = YinYangDataset(size=1, r_small=0.1, r_big=0.5)
        dist = dataset.dist_to_right_dot(0.4, 0.1)
        self.assertAlmostEqual(dist, 0.531507, 6)

    def test_dist_to_left_dot(self):
        """ Test distance calculation to left dot """
        dataset = YinYangDataset(size=1, r_small=0.1, r_big=0.5)
        dist = dataset.dist_to_left_dot(0.4, 0.1)
        self.assertAlmostEqual(dist, 0.427200, 6)


if __name__ == '__main__':
    unittest.main()
