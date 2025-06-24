import unittest
import torch
from hxtorch.spiking import (Handle, LIFObservables, LIObservables,
    SynapseHandle, TensorHandle)


class TestHXHandle(unittest.TestCase):
    """
    Test the hxtorch.spiking.Handle and redundantly check for correct type
    aliasing of hxtorch.spiking.LIFObservables, hxtorch.spiking.LIObservables,
    hxtorch.spiking.SynapseHandle and hxtorch.spiking.TensorHandle.
    """

    def test_handle(self):
        """
        Test Handle
        """
        # artificial data
        spikes = torch.randint(0, 2, size=(20, 10, 2), dtype=bool)
        membrane_cadc = torch.rand(size=(20, 10, 2))
        current = torch.rand(size=(20, 10, 2))
        membrane_madc = torch.rand(size=(20, 10, 2))
        adaptation = torch.rand(size=(20, 10, 2))

        # test class and object construction
        handle = Handle(spikes=spikes, membrane_cadc=membrane_cadc,
            current=current, membrane_madc=membrane_madc,
            adaptation=adaptation)
        empty_handle = Handle('spikes', 'membrane_cadc', 'current',
            'membrane_madc', 'adaptation')
        self.assertTrue(isinstance(empty_handle, type(handle)))

        # test holds
        handle.spikes = None
        self.assertTrue(handle.holds("membrane_cadc"))
        self.assertFalse(handle.holds("spikes"))
        self.assertFalse(handle.holds("wrong_key"))

        # test clone
        handle_wrong = Handle('adaptation', 'membrane_madc', 'current',)
        handle_right = Handle('adaptation', 'membrane_madc', 'current',
            'membrane_cadc', 'spikes')
        self.assertRaises(AssertionError, handle_wrong.clone, handle)
        handle_right.clone(handle)
        self.assertEqual(handle_right, handle)

    def test_type_aliasing(self):
        """
        Assert correct type aliasing of hxtorch.spiking.LIFObservables,
        hxtorch.spiking.LIObservables, hxtorch.spiking.SynapseHandle and
        hxtorch.spiking.TensorHandle.
        """
        self.assertEqual(LIFObservables, type(Handle('spikes', 'membrane_cadc',
            'current', 'membrane_madc')))
        self.assertEqual(LIObservables, type(Handle('membrane_cadc', 'current',
            'membrane_madc')))
        self.assertEqual(SynapseHandle, type(Handle('graded_spikes')))
        self.assertEqual(TensorHandle, type(Handle('tensor')))


if __name__ == '__main__':
    unittest.main()
