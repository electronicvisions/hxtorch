import unittest
import torch
from hxtorch.spiking import NeuronHandle, ReadoutNeuronHandle, SynapseHandle


class TestHXHandle(unittest.TestCase):
    """
    Test the hxtorch.spiking.Handle
    """

    def test_neuronhandle(self):
        """
        Test NeuronHandle
        """
        # artificial data
        spikes = torch.randint(0, 2, size=(20, 10, 2), dtype=bool)
        v_cadc = torch.rand(size=(20, 10, 2))
        current = torch.rand(size=(20, 10, 2))
        v_madc = torch.rand(size=(20, 10, 2))

        # test put
        handle = NeuronHandle()
        handle.put(spikes, v_cadc, current, v_madc)
        self.assertTrue(torch.equal(spikes, handle.spikes))
        self.assertTrue(torch.equal(v_cadc, handle.v_cadc))
        self.assertTrue(torch.equal(current, handle.current))
        self.assertTrue(torch.equal(v_madc, handle.v_madc))

        handle = NeuronHandle()
        handle.put(
            v_madc=v_madc, v_cadc=v_cadc, current=current, spikes=spikes)
        self.assertTrue(torch.equal(spikes, handle.spikes))
        self.assertTrue(torch.equal(v_cadc, handle.v_cadc))
        self.assertTrue(torch.equal(current, handle.current))
        self.assertTrue(torch.equal(v_madc, handle.v_madc))

        handle = NeuronHandle()
        handle.put(v_cadc=v_cadc)
        self.assertTrue(handle.spikes is None)
        self.assertTrue(handle.v_madc is None)
        self.assertTrue(handle.current is None)
        self.assertTrue(torch.equal(v_cadc, handle.v_cadc))

        handle = NeuronHandle()
        handle.put(v_madc=v_madc)
        self.assertTrue(handle.spikes is None)
        self.assertTrue(handle.v_cadc is None)
        self.assertTrue(handle.current is None)
        self.assertTrue(torch.equal(v_madc, handle.v_madc))

        handle = NeuronHandle()
        handle.put(current=current)
        self.assertTrue(handle.spikes is None)
        self.assertTrue(handle.v_cadc is None)
        self.assertTrue(handle.v_madc is None)
        self.assertTrue(torch.equal(current, handle.current))

        handle = NeuronHandle()
        handle.put(spikes)
        self.assertTrue(torch.equal(spikes, handle.spikes))
        self.assertTrue(handle.v_cadc is None)
        self.assertTrue(handle.current is None)
        self.assertTrue(handle.v_madc is None)

        with self.assertRaises(AssertionError):
            handle.put(
                v_cadc=v_cadc, spikes=v_madc, wrong=v_cadc)
        with self.assertRaises(AssertionError):
            handle.put(v_cadc, spikes=v_cadc, wrong=v_cadc)
        with self.assertRaises(AssertionError):
            handle.put(v_cadc, v_madc, v_madc, v_madc, current)

        # test holds
        handle = NeuronHandle()
        handle.put(v_cadc=v_cadc)
        self.assertTrue(handle.holds("v_cadc"))
        self.assertFalse(handle.holds("spikes"))
        self.assertFalse(handle.holds("wrong_key"))

    def test_readoutneuronhandle(self):
        """
        Test ReadoutNeuronHandle
        """
        # artificial data
        v_cadc = torch.rand(size=(20, 10, 2))
        current = torch.rand(size=(20, 10, 2))
        v_madc = torch.rand(size=(20, 10, 2))

        # test put
        handle = ReadoutNeuronHandle()
        handle.put(v_cadc, current, v_madc)
        self.assertTrue(torch.equal(v_cadc, handle.v_cadc))
        self.assertTrue(torch.equal(current, handle.current))
        self.assertTrue(torch.equal(v_madc, handle.v_madc))

        handle = ReadoutNeuronHandle()
        handle.put(v_cadc=v_cadc, v_madc=v_madc, current=current)
        self.assertTrue(torch.equal(v_cadc, handle.v_cadc))
        self.assertTrue(torch.equal(current, handle.current))
        self.assertTrue(torch.equal(v_madc, handle.v_madc))

        handle = ReadoutNeuronHandle()
        self.assertTrue(handle.v_cadc is None)
        self.assertTrue(handle.current is None)
        self.assertTrue(handle.v_madc is None)

        handle = ReadoutNeuronHandle()
        with self.assertRaises(AssertionError):
            handle.put(v_cadc=v_cadc, v_madc=v_madc, spikes=v_cadc)
        with self.assertRaises(AssertionError):
            handle.put(v_madc, spikes=v_cadc)
        with self.assertRaises(AssertionError):
            handle.put(v_cadc, v_madc, v_cadc, current)

        # test holds
        handle = ReadoutNeuronHandle()
        handle.put(v_cadc=v_cadc)
        self.assertTrue(handle.holds("v_cadc"))
        self.assertFalse(handle.holds("wrong_key"))

    def test_synapsehandle(self):
        """
        Test SynapseHandle
        """
        # artificial data
        graded_spikes = torch.rand(size=(20, 10, 2))

        # test put
        handle = SynapseHandle()
        handle.put(graded_spikes)
        self.assertTrue(torch.equal(graded_spikes, handle.graded_spikes))

        handle = SynapseHandle()
        handle.put(graded_spikes=graded_spikes)
        self.assertTrue(torch.equal(graded_spikes, handle.graded_spikes))

        handle = SynapseHandle()
        self.assertTrue(handle.graded_spikes is None)

        handle = SynapseHandle()
        with self.assertRaises(AssertionError):
            handle.put(graded_spikes=graded_spikes, spikes=graded_spikes)
        with self.assertRaises(AssertionError):
            handle.put(graded_spikes, spikes=graded_spikes)
        with self.assertRaises(AssertionError):
            handle.put(graded_spikes, graded_spikes)

        # test holds
        handle = SynapseHandle()
        handle.put(graded_spikes=graded_spikes)
        self.assertTrue(handle.holds("graded_spikes"))
        self.assertFalse(handle.holds("wrong_key"))


if __name__ == '__main__':
    unittest.main()
