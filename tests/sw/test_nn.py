import unittest
from unittest.mock import Mock
import torch
from torch.autograd import gradcheck
from dlens_vx import hal
from hxtorch import nn  as hxnn


class TestNN(unittest.TestCase):
    """
    Test the hxtorch.nn module.
    """

    def test_scale_input(self):
        """
        Test the scaling of the inputs.
        """
        x_in = torch.arange(8., 16., .5).view(2, 8)
        x_in.requires_grad = True
        x_scaled = hxnn.scale_input(x_in)
        self.assertLessEqual(x_scaled.max(), hal.PADIEvent.HagenActivation.max)

        self.assertTrue(gradcheck(hxnn.scale_input, x_in, eps=1e-2, atol=1e-1))

    def test_scale_weight(self):
        """
        Test the scaling of the weight.
        """
        w_in = torch.arange(-32, 32., .5).view(8, 16)
        w_in.requires_grad = True
        w_scaled = hxnn.scale_weight(w_in)
        self.assertLessEqual(w_scaled.abs().max(), hal.SynapseQuad.Weight.max)

        self.assertTrue(
            gradcheck(hxnn.scale_weight, w_in, eps=1e-2, atol=1e-1))

    def test_linear(self):
        """
        Test the Linear layer.
        """
        def matmul_side_effect(x, weights, **kwargs):
            return torch.matmul(x, weights)

        torch_layer = torch.nn.Linear(19, 2)
        hxtorch_layer = hxnn.Linear(
            19, 2, num_sends=3, wait_between_events=10,
            input_transform=lambda x: x,
            weight_transform=lambda w: torch.ones_like(w))
        hxtorch_layer._matmul = Mock(side_effect=matmul_side_effect)
        x_in = torch.arange(0., 19.).unsqueeze(0)
        x_out = torch_layer(x_in)
        x_out_hx = hxtorch_layer(x_in)

        call_args, call_kwargs = hxtorch_layer._matmul.call_args
        self.assertSequenceEqual(x_out.shape, x_out_hx.shape)
        self.assertListEqual(call_args[0].tolist(), x_in.tolist())
        self.assertListEqual(call_args[1].tolist(),
                             torch.ones((19, 2)).tolist())
        self.assertEqual(call_kwargs["num_sends"], 3)
        self.assertEqual(call_kwargs["wait_between_events"], 10)
        self.assertEqual(call_kwargs["mock"], False)

        # repr
        self.assertRegex(repr(hxtorch_layer), r'Linear\(.*, num_sends=3, wait')


if __name__ == '__main__':
    unittest.main()
