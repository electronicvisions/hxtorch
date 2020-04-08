import unittest
import torch
import hxtorch
import dlens_vx.hxcomm as hxcomm
import dlens_vx.sta as sta
import pyhaldls_vx as hal
import pygrenade_vx as grenade
import dlens_vx

dlens_vx.logger.default_config(level=dlens_vx.logger.LogLevel.INFO)

class TestCaseConv(object):
    def __init__(self, name, conv, weights, inputs, stride, size):
        self.name = name
        self.conv = conv
        self.weights = weights
        self.inputs = inputs
        self.stride = stride
        self.size = size

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class TestConv(unittest.TestCase):
    TESTS = set()

    @classmethod
    def setUpClass(cls):
        with hxcomm.ManagedConnection() as connection:
            sta.run(connection, sta.generate(sta.ExperimentInit())[0].done())
            chip = grenade.ChipConfig()
            hxtorch.init(chip, connection)

    @classmethod
    def tearDownClass(cls):
        hxtorch.release()

    @classmethod
    def generate_cases(cls):
        for test in cls.TESTS:
            def generate_test(case):
                def test_invocation(self):
                    result = case.conv(case.inputs, case.weights, stride=case.stride)
                    self.assertEqual(result.size(), torch.Size(case.size))
                    self.assertTrue(result.is_contiguous())
                return test_invocation

            test_method = generate_test(test)
            test_method.__name__ = 'test_' + test.name
            setattr(TestConv, test_method.__name__, test_method)

    def test_conv1d_construction(self):
        # data shape not matching conv1d
        with self.assertRaises(RuntimeError):
            data_in = torch.ones(5)
            weights_in = torch.empty((7, 1, 1))
            hxtorch.conv1d(data_in, weights_in, 2)

        # weights shape not matching conv1d
        with self.assertRaises(RuntimeError):
            data_in = torch.ones((5, 2, 1))
            weights_in = torch.empty((7, 1))
            hxtorch.conv1d(data_in, weights_in, 3)

        # in_channels not matching
        with self.assertRaises(RuntimeError):
            data_in = torch.ones((5, 2, 1))
            weights_in = torch.empty((7, 1, 1))
            hxtorch.conv1d(data_in, weights_in, 3)

        # match with in_channels = 2, out_channels = 3, N = 2, kernel_size = 5
        weights = torch.tensor([
            [[1,2,3,4,5], [6,7,8,9,10]],
            [[11,12,13,14,15], [16,17,18,19,20]],
            [[21,22,23,24,25], [26,27,28,29,30]]
        ], dtype=torch.float)
        inputs = torch.tensor([
            [[i for i in range(30)], [i + 30 for i in range(30)]],
            [[i for i in range(30)], [i + 30 for i in range(30)]]
        ], dtype=torch.float)
        result = hxtorch.conv1d(inputs, weights, 4)
        self.assertEqual(result.size(), torch.Size([2, 3, 7]))

    def test_conv1d_backward(self):
        weights_in = torch.ones((3, 2, 5))
        weights_in.requires_grad = True
        data_in = torch.ones((1, 2, 30))
        result = hxtorch.conv1d(data_in, weights_in, 4)
        self.assertEqual(result.size(), torch.Size([1, 3, 7]))
        loss = result.sum()
        loss.backward()

    def test_conv2d_backward(self):
        weights_in = torch.ones((3, 2, 5, 10))
        weights_in.requires_grad = True
        data_in = torch.ones((1, 2, 30, 60))
        result = hxtorch.conv2d(data_in, weights_in, [4, 8])
        self.assertEqual(result.size(), torch.Size([1, 3, 7, 7]))
        loss = result.sum()
        loss.backward()


def add_cases_conv1d(weights, inputs, stride, size, postfix=''):
    TestConv.TESTS.update({
        TestCaseConv('conv1d_torch' + postfix, torch.conv1d, weights, inputs, stride, size),
        TestCaseConv('conv1d_hxtorch' + postfix, hxtorch.conv1d, weights, inputs, stride, size)})

add_cases_conv1d(torch.rand((1, 1, 5), dtype=torch.float),
              torch.rand((1, 1, 30), dtype=torch.float),
              4,
              [1, 1, 7],
              '_batch1_outchannels1_inchannels1_kernel_larger_stride')

add_cases_conv1d(torch.rand((1, 3, 5), dtype=torch.float),
              torch.rand((2, 3, 30), dtype=torch.float),
              4,
              [2, 1, 7],
              '_batch2_outchannels1_inchannels3_kernel_larger_stride')

add_cases_conv1d(torch.rand((4, 3, 5), dtype=torch.float),
              torch.rand((2, 3, 30), dtype=torch.float),
              4,
              [2, 4, 7],
              '_batch2_outchannels4_inchannels3_kernel_larger_stride')

add_cases_conv1d(torch.rand((1, 1, 5), dtype=torch.float),
              torch.rand((1, 1, 30), dtype=torch.float),
              7,
              [1, 1, 4],
              '_batch1_outchannels1_inchannels1_kernel_smaller_stride')

add_cases_conv1d(torch.rand((1, 3, 5), dtype=torch.float),
              torch.rand((2, 3, 30), dtype=torch.float),
              7,
              [2, 1, 4],
              '_batch2_outchannels1_inchannels3_kernel_smaller_stride')

add_cases_conv1d(torch.rand((4, 3, 5), dtype=torch.float),
              torch.rand((2, 3, 30), dtype=torch.float),
              7,
              [2, 4, 4],
              '_batch2_outchannels4_inchannels3_kernel_smaller_stride')

def add_cases_conv2d(weights, inputs, stride, size, postfix=''):
    TestConv.TESTS.update({
        TestCaseConv('conv2d_torch' + postfix, torch.conv2d, weights, inputs, stride, size),
        TestCaseConv('conv2d_hxtorch' + postfix, hxtorch.conv2d, weights, inputs, stride, size)})

add_cases_conv2d(torch.rand((1, 1, 5, 10), dtype=torch.float),
              torch.rand((1, 1, 30, 60), dtype=torch.float),
              [4, 8],
              [1, 1, 7, 7],
              '_batch1_outchannels1_inchannels1_kernel_larger_stride')

add_cases_conv2d(torch.rand((1, 3, 5, 10), dtype=torch.float),
              torch.rand((2, 3, 30, 60), dtype=torch.float),
              [4, 8],
              [2, 1, 7, 7],
              '_batch2_outchannels1_inchannels3_kernel_larger_stride')

add_cases_conv2d(torch.rand((4, 3, 5, 10), dtype=torch.float),
              torch.rand((2, 3, 30, 60), dtype=torch.float),
              [4, 8],
              [2, 4, 7, 7],
              '_batch2_outchannels4_inchannels3_kernel_larger_stride')

add_cases_conv2d(torch.rand((1, 1, 5, 10), dtype=torch.float),
              torch.rand((1, 1, 30, 60), dtype=torch.float),
              [7, 14],
              [1, 1, 4, 4],
              '_batch1_outchannels1_inchannels1_kernel_smaller_stride')

add_cases_conv2d(torch.rand((1, 3, 5, 10), dtype=torch.float),
              torch.rand((2, 3, 30, 60), dtype=torch.float),
              [7, 14],
              [2, 1, 4, 4],
              '_batch2_outchannels1_inchannels3_kernel_smaller_stride')

add_cases_conv2d(torch.rand((4, 3, 5, 10), dtype=torch.float),
              torch.rand((2, 3, 30, 60), dtype=torch.float),
              [7, 14],
              [2, 4, 4, 4],
              '_batch2_outchannels4_inchannels3_kernel_smaller_stride')

TestConv.generate_cases()


if __name__ == '__main__':
    unittest.main()
