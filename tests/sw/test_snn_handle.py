import unittest
import torch
from hxtorch.snn import NeuronHandle, ReadoutNeuronHandle, SynapseHandle


class TestHXHandle(unittest.TestCase):
    """
    Test the hxtorch.snn.Handle
    """

    def test_neuronhandle(self):
        """
        Test NeuronHandle
        """
        # artificial data
        spikes = torch.randint(0, 2, size=(20, 10, 2), dtype=bool)
        v_cadc = torch.rand(size=(20, 10, 2))
        v_madc = torch.rand(size=(20, 10, 2))

        # test put
        handle = NeuronHandle()
        handle.put(spikes, v_cadc, v_madc)
        self.assertTrue(torch.equal(spikes, handle.spikes))
        self.assertTrue(torch.equal(v_cadc, handle.v_cadc))
        self.assertTrue(torch.equal(v_madc, handle.v_madc))

        handle = NeuronHandle()
        handle.put(v_madc=v_madc, v_cadc=v_cadc, spikes=spikes)
        self.assertTrue(torch.equal(spikes, handle.spikes))
        self.assertTrue(torch.equal(v_cadc, handle.v_cadc))
        self.assertTrue(torch.equal(v_madc, handle.v_madc))

        handle = NeuronHandle()
        handle.put(v_cadc=v_cadc)
        self.assertTrue(handle.spikes is None)
        self.assertTrue(handle.v_madc is None)
        self.assertTrue(torch.equal(v_cadc, handle.v_cadc))

        handle = NeuronHandle()
        handle.put(v_madc=v_madc)
        self.assertTrue(handle.spikes is None)
        self.assertTrue(handle.v_cadc is None)
        self.assertTrue(torch.equal(v_madc, handle.v_madc))

        handle = NeuronHandle()
        handle.put(spikes)
        self.assertTrue(torch.equal(spikes, handle.spikes))
        self.assertTrue(handle.v_cadc is None)
        self.assertTrue(handle.v_madc is None)

        with self.assertRaises(AssertionError):
            handle.put(
                v_cadc=v_cadc, spikes=v_madc, wrong=v_cadc)
        with self.assertRaises(AssertionError):
            handle.put(v_cadc, spikes=v_cadc, wrong=v_cadc)
        with self.assertRaises(AssertionError):
            handle.put(v_cadc, v_madc, v_madc, v_madc)

        # test holds
        handle = NeuronHandle()
        handle.put(v_cadc=v_cadc)
        self.assertTrue(handle.holds("v_cadc"))
        self.assertFalse(handle.holds("spikes"))
        self.assertFalse(handle.holds("wrong_key"))

        # test observable state
        handle = NeuronHandle()
        handle.put(spikes, v_cadc, v_madc)
        self.assertTrue(torch.equal(spikes, handle.observable_state))
        self.assertTrue(torch.equal(handle.spikes, handle.observable_state))
        handle.put(None, v_cadc)
        self.assertTrue(handle.observable_state is None)

    def test_readoutneuronhandle(self):
        """
        Test ReadoutNeuronHandle
        """
        # artificial data
        v_cadc = torch.rand(size=(20, 10, 2))
        v_madc = torch.rand(size=(20, 10, 2))

        # test put
        handle = ReadoutNeuronHandle()
        handle.put(v_cadc, v_madc)
        self.assertTrue(torch.equal(v_cadc, handle.v_cadc))
        self.assertTrue(torch.equal(v_madc, handle.v_madc))

        handle = ReadoutNeuronHandle()
        handle.put(v_cadc=v_cadc, v_madc=v_madc)
        self.assertTrue(torch.equal(v_cadc, handle.v_cadc))
        self.assertTrue(torch.equal(v_madc, handle.v_madc))

        handle = ReadoutNeuronHandle()
        self.assertTrue(handle.v_cadc is None)
        self.assertTrue(handle.v_madc is None)

        handle = ReadoutNeuronHandle()
        with self.assertRaises(AssertionError):
            handle.put(v_cadc=v_cadc, v_madc=v_madc, spikes=v_cadc)
        with self.assertRaises(AssertionError):
            handle.put(v_madc, spikes=v_cadc)
        with self.assertRaises(AssertionError):
            handle.put(v_cadc, v_madc, v_cadc)

        # test holds
        handle = ReadoutNeuronHandle()
        handle.put(v_cadc=v_cadc)
        self.assertTrue(handle.holds("v_cadc"))
        self.assertFalse(handle.holds("wrong_key"))

        # test observable state
        handle = ReadoutNeuronHandle()
        handle.put(v_cadc, v_madc)
        self.assertTrue(torch.equal(v_cadc, handle.observable_state))
        self.assertTrue(torch.equal(handle.v_cadc, handle.observable_state))
        handle.put(None)
        self.assertTrue(handle.observable_state is None)

    def test_synapsehandle(self):
        """
        Test SynapseHandle
        """
        # artificial data
        current = torch.rand(size=(20, 10, 2))

        # test put
        handle = SynapseHandle()
        handle.put(current)
        self.assertTrue(torch.equal(current, handle.current))

        handle = SynapseHandle()
        handle.put(current=current)
        self.assertTrue(torch.equal(current, handle.current))

        handle = SynapseHandle()
        self.assertTrue(handle.current is None)

        handle = SynapseHandle()
        with self.assertRaises(AssertionError):
            handle.put(current=current, spikes=current)
        with self.assertRaises(AssertionError):
            handle.put(current, spikes=current)
        with self.assertRaises(AssertionError):
            handle.put(current, current)

        # test holds
        handle = SynapseHandle()
        handle.put(current=current)
        self.assertTrue(handle.holds("current"))
        self.assertFalse(handle.holds("wrong_key"))

        # test observable state
        handle = SynapseHandle()
        handle.put(current)
        self.assertTrue(torch.equal(current, handle.observable_state))
        self.assertTrue(
            torch.equal(handle.current, handle.observable_state))
        handle.put(None)

    def test_observable_state(self):
        """
        Test observable state
        """
        # artificial data
        data = torch.rand(size=(20, 10, 2))

        # test put
        handle = NeuronHandle(data)

        self.assertTrue(torch.equal(handle.observable_state, data))
        self.assertTrue(torch.equal(handle.spikes, data))
        self.assertEqual(handle.observable_state_identifier, "spikes")

        handle.observable_state_identifier = "v_cadc"
        self.assertTrue(handle.observable_state is None)
        self.assertEqual(handle.observable_state_identifier, "v_cadc")


if __name__ == '__main__':
    unittest.main()
