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
        membrane = torch.rand(size=(20, 10, 2))

        # test put
        handle = NeuronHandle()
        handle.put(spikes, membrane)
        self.assertTrue(torch.equal(spikes, handle.spikes))
        self.assertTrue(torch.equal(membrane, handle.membrane))

        handle = NeuronHandle()
        handle.put(membrane=membrane, spikes=spikes)
        self.assertTrue(torch.equal(spikes, handle.spikes))
        self.assertTrue(torch.equal(membrane, handle.membrane))

        handle = NeuronHandle()
        handle.put(membrane=membrane)
        self.assertTrue(handle.spikes is None)
        self.assertTrue(torch.equal(membrane, handle.membrane))

        handle = NeuronHandle()
        handle.put(spikes)
        self.assertTrue(torch.equal(spikes, handle.spikes))
        self.assertTrue(handle.membrane is None)

        with self.assertRaises(AssertionError):
            handle.put(
                membrane=membrane, spikes=membrane, wrong=membrane)
        with self.assertRaises(AssertionError):
            handle.put(membrane, spikes=membrane, wrong=membrane)
        with self.assertRaises(AssertionError):
            handle.put(membrane, membrane, membrane)

        # test holds
        handle = NeuronHandle()
        handle.put(membrane=membrane)
        self.assertTrue(handle.holds("membrane"))
        self.assertFalse(handle.holds("spikes"))
        self.assertFalse(handle.holds("wrong_key"))

        # test observable state
        handle = NeuronHandle()
        handle.put(spikes, membrane)
        self.assertTrue(torch.equal(spikes, handle.observable_state))
        self.assertTrue(torch.equal(handle.spikes, handle.observable_state))
        handle.put(None, membrane)
        self.assertTrue(handle.observable_state is None)

    def test_readoutneuronhandle(self):
        """
        TestXReadoutNeuronHandle
        """
        # artificial data
        membrane = torch.rand(size=(20, 10, 2))

        # test put
        handle = ReadoutNeuronHandle()
        handle.put(membrane)
        self.assertTrue(torch.equal(membrane, handle.membrane))

        handle = ReadoutNeuronHandle()
        handle.put(membrane=membrane)
        self.assertTrue(torch.equal(membrane, handle.membrane))

        handle = ReadoutNeuronHandle()
        self.assertTrue(handle.membrane is None)

        handle = ReadoutNeuronHandle()
        with self.assertRaises(AssertionError):
            handle.put(membrane=membrane, spikes=membrane)
        with self.assertRaises(AssertionError):
            handle.put(membrane, spikes=membrane)
        with self.assertRaises(AssertionError):
            handle.put(membrane, membrane)

        # test holds
        handle = ReadoutNeuronHandle()
        handle.put(membrane=membrane)
        self.assertTrue(handle.holds("membrane"))
        self.assertFalse(handle.holds("wrong_key"))

        # test observable state
        handle = ReadoutNeuronHandle()
        handle.put(membrane)
        self.assertTrue(torch.equal(membrane, handle.observable_state))
        self.assertTrue(torch.equal(handle.membrane, handle.observable_state))
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

        handle.observable_state_identifier = "membrane"
        self.assertTrue(handle.observable_state is None)
        self.assertEqual(handle.observable_state_identifier, "membrane")


if __name__ == '__main__':
    unittest.main()
