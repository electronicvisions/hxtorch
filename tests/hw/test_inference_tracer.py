import os
import unittest
import tempfile
import torch
import hxtorch


class TestInferenceTracer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hxtorch.init()

    @classmethod
    def tearDownClass(cls):
        hxtorch.release()

    def template(self, func):
        fd, path = tempfile.mkstemp()
        try:
            tracer = hxtorch.InferenceTracer(path)
            tracer.start()
            expectation = func()
            names = tracer.stop()
            self.assertEqual(names, expectation)
        finally:
            os.remove(path)

    @staticmethod
    def empty():
        return []

    @staticmethod
    def sequence():
        inputs = torch.zeros((20,256))

        weights_1 = torch.zeros((256,512))
        r1 = hxtorch.mac(inputs, weights_1)
        r2 = hxtorch.relu(r1)
        r3 = hxtorch.converting_relu(r2)
        weights_2 = torch.zeros((512,123))
        r4 = hxtorch.mac(r3, weights_2)
        hxtorch.relu(r4)
        return ["mac", "relu", "converting_relu", "mac", "relu"]

    @staticmethod
    def wrong_sequence():
        inputs = torch.zeros((20,256))

        weights_1 = torch.zeros((256,512))
        r1 = hxtorch.mac(inputs, weights_1)
        r2 = hxtorch.relu(r1)
        r3 = hxtorch.converting_relu(r2)
        r3 = r3 + 2.  # not traceable
        weights_2 = torch.zeros((512,123))
        r4 = hxtorch.mac(r3, weights_2)
        hxtorch.relu(r4)

    def test_empty(self):
        self.template(TestInferenceTracer.empty)

    def test_sequence(self):
        self.template(TestInferenceTracer.sequence)

    def test_wrong_sequence(self):
        with self.assertRaises(RuntimeError):
            self.template(TestInferenceTracer.wrong_sequence)

    def test_nested(self):
        fd, path = tempfile.mkstemp()
        nested_fd, nested_path = tempfile.mkstemp()
        try:
            inputs = torch.zeros((20,256))
            weights_1 = torch.zeros((256,512))
            weights_2 = torch.zeros((512,123))

            tracer = hxtorch.InferenceTracer(path)
            tracer.start()

            r1 = hxtorch.mac(inputs, weights_1)
            nested_tracer = hxtorch.InferenceTracer(nested_path)
            nested_tracer.start()
            r2 = hxtorch.relu(r1)
            r3 = hxtorch.converting_relu(r2)
            nested_names = nested_tracer.stop()
            r4 = hxtorch.mac(r3, weights_2)
            hxtorch.relu(r4)
            names = tracer.stop()

            expectation = ["mac", "relu", "converting_relu", "mac", "relu"]
            nested_expectation = ["relu", "converting_relu"]
            self.assertEqual(names, expectation)
            self.assertEqual(nested_names, nested_expectation)
        finally:
            os.remove(path)
            os.remove(nested_path)

    def test_single_relu_inference_trace(self):
        fd, path = tempfile.mkstemp()
        try:
            inputs = torch.arange(-128., 127.)

            tracer = hxtorch.InferenceTracer(path)
            tracer.start()
            res = hxtorch.relu(inputs)
            names = tracer.stop()

            expectation = ["relu"]
            self.assertEqual(names, expectation)

            traced_res = hxtorch.inference_trace(inputs, path)
            self.assertTrue(torch.equal(traced_res, res))
        finally:
            os.remove(path)


if __name__ == '__main__':
    unittest.main()
