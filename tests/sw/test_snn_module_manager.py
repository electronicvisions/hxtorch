"""
Test Modules and Node classes
"""
import unittest
import torch
import hxtorch.snn as snn
from hxtorch.snn.backend.nodes import Node
from hxtorch.snn.backend.module_manager import ModuleManager


class TestNode(unittest.TestCase):
    """ Test Nodes """

    def test_create_node(self):
        """
        Test node can be created properly.
        """
        module = snn.HXModule(snn.Instance(), None)
        in_handles = (snn.NeuronHandle(),)
        out_handle = snn.NeuronHandle()

        # Create Node
        node = Node(module, in_handles, out_handle)
        self.assertEqual(node.module, module)
        self.assertEqual(node.input_handle, in_handles)
        self.assertEqual(node.output_handle, out_handle)

        # Pre and post nodes are empty upon construction
        self.assertEqual(node.pre, [])
        self.assertEqual(node.post, [])

        # No grenade descriptor assigned yet
        self.assertIsNone(node.descriptor)

    def test_set_handle(self):
        """ Test handle assignment """
        module = snn.HXModule(snn.Instance(), None)
        in_handles1 = (snn.NeuronHandle(),)
        out_handle1 = snn.NeuronHandle()
        node = Node(module, in_handles1, out_handle1)

        # New handles
        in_handles2 = (snn.NeuronHandle(),)
        out_handle2 = snn.NeuronHandle()

        # Update
        node.set_handles(in_handles2, out_handle2)
        self.assertEqual(node.input_handle, in_handles2)
        self.assertEqual(node.output_handle, out_handle2)
        self.assertNotEqual(node.input_handle, in_handles1)
        self.assertNotEqual(node.output_handle, out_handle1)


class TestModuleManager(unittest.TestCase):
    """ Test Modules object """

    def test_add(self):
        """ Test add module """
        modules = ModuleManager()
        self.assertEqual(len(modules), 0)

        # Test modules
        module1 = snn.HXModule(None, None)
        module2 = snn.HXModule(None, None)

        # Test handles
        in_handle1 = snn.NeuronHandle()
        out_handle1 = snn.NeuronHandle()
        in_handle2 = snn.NeuronHandle()
        out_handle2 = snn.NeuronHandle()
        in_handle3 = snn.NeuronHandle()
        out_handle3 = snn.NeuronHandle()

        # Add
        node1 = modules.add(module1, in_handle1, out_handle1)
        self.assertEqual(len(modules), 1)
        self.assertEqual(node1.input_handle, (in_handle1,))
        self.assertEqual(node1.output_handle, out_handle1)

        # Add second
        node2 = modules.add(module2, in_handle2, out_handle2)
        self.assertEqual(len(modules), 2)
        self.assertNotEqual(node2, node1)

        # Add again with same module but different handles -> should update
        node3 = modules.add(module1, in_handle3, out_handle3)
        self.assertEqual(len(modules), 2)
        self.assertEqual(node1, node3)
        self.assertEqual(node3.input_handle, (in_handle3,))
        self.assertEqual(node3.output_handle, out_handle3)

    def test_clear(self):
        """ Test clear removes nodes """
        modules = ModuleManager()
        self.assertEqual(len(modules), 0)

        # Test modules
        module1 = snn.HXModule(None, None)
        module2 = snn.HXModule(None, None)

        # Test handles
        in_handle1 = snn.NeuronHandle()
        out_handle1 = snn.NeuronHandle()
        in_handle2 = snn.NeuronHandle()
        out_handle2 = snn.NeuronHandle()

        # Add
        modules.add(module1, in_handle1, out_handle1)
        modules.add(module2, in_handle2, out_handle2)
        self.assertEqual(len(modules), 2)

        # Clear
        modules.clear()
        self.assertEqual(len(modules), 0)

    def test_leafs(self):
        """ Test get correct leafs """
        modules = ModuleManager()

        # Test modules
        module1 = snn.HXModule(None, None)
        module2 = snn.HXModule(None, None)
        module3 = snn.HXModule(None, None)
        module4 = snn.HXModule(None, None)
        # Test handles
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        handle5 = snn.NeuronHandle()

        # Add Modules
        modules.add(module1, handle1, handle2)
        modules.add(module2, handle2, handle3)
        leaf_node1 = modules.add(module3, handle3, handle4)
        # Connect
        modules.connect_nodes()

        # Get leaf
        leafs = modules.leafs()
        self.assertEqual(len(leafs), 1)
        self.assertEqual(leafs[0], leaf_node1)

        # Several leaf nodes
        leaf_node2 = modules.add(module4, handle3, handle5)
        # Connect
        modules.connect_nodes()

        # Get leafs
        leafs = modules.leafs()
        self.assertEqual(len(leafs), 2)
        self.assertEqual(leafs[0], leaf_node1)
        self.assertEqual(leafs[1], leaf_node2)

    def test_inputs(self):
        modules = ModuleManager()

        # Test modules
        module1 = snn.HXModule(None, None)
        module2 = snn.HXModule(None, None)
        module3 = snn.HXModule(None, None)
        module4 = snn.HXModule(None, None)

        # Test handles
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        handle5 = snn.NeuronHandle()
        handle6 = snn.NeuronHandle()

        # Add Modules
        in_node1 = modules.add(module1, handle1, handle2)
        modules.add(module2, (handle2, handle6), handle3)
        modules.add(module3, handle3, handle4)
        # Connect
        modules.connect_nodes()

        # Get inputs
        inputs = modules.inputs()
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0], in_node1)

        # Several inputs nodes
        in_node2 = modules.add(module4, handle5, handle6)
        # Connect
        modules.connect_nodes()

        # Get inputs
        inputs = modules.inputs()
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0], in_node1)
        self.assertEqual(inputs[1], in_node2)

    def test_connect_nodes(self):
        """ Test existing nodes are connected correctly """
        modules = ModuleManager()
        ins = snn.Instance()
        # Test modules
        module1 = snn.Neuron(10, ins, None)
        module2 = snn.Synapse(10, 10, ins, None)
        module3 = snn.Neuron(10, ins, None)
        module4 = snn.Synapse(10, 10, ins, None)
        # Test handles
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        handle5 = snn.NeuronHandle()
        # Add and connect
        node1 = modules.add(module1, handle1, handle2)
        node2 = modules.add(module2, handle2, handle3)
        node3 = modules.add(module3, (handle3, handle5), handle4)
        node4 = modules.add(module4, handle4, handle5)

        # Connect
        modules.connect_nodes()

        # Test
        self.assertEqual(node1.pre, [])
        self.assertEqual(node1.post, [node2])
        self.assertEqual(node2.pre, [node1])
        self.assertEqual(node2.post, [node3])
        self.assertEqual(node3.pre, [node2, node4])
        self.assertEqual(node3.post, [node4])
        self.assertEqual(node4.pre, [node3])
        self.assertEqual(node4.post, [node3])

    def test_get_node(self):
        """ Test get_node returns the correct node """
        modules = ModuleManager()
        ins = snn.Instance()
        # Test modules
        module1 = snn.Neuron(10, ins, None)
        module2 = snn.Synapse(10, 10, ins, None)
        # Test handles
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        # Add and connect
        node1 = modules.add(module1, handle1, handle2)
        node2 = modules.add(module2, handle2, handle3)

        # Test
        node1_ret = modules.get_node(module1)
        self.assertEqual(node1, node1_ret)
        node2_ret = modules.get_node(module2)
        self.assertEqual(node2, node2_ret)

    def test_module_exists(self):
        """ Test module exists """
        modules = ModuleManager()
        ins = snn.Instance()
        # Test modules
        module1 = snn.Synapse(10, 10, ins, None)
        module2 = snn.Neuron(10, ins, None)
        # Test handles
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        # Add and connect
        modules.add(module1, handle1, handle2)
        modules.add(module2, handle3, handle4)
        # Check if exists
        self.assertTrue(modules.module_exists(module1))
        self.assertFalse(modules.module_exists(snn.Neuron(10, ins, None)))

    def test_pre_pop(self):
        """ Test get all pre-populations """
        modules = ModuleManager()
        ins = snn.Instance()
        # Test modules
        module0 = snn.Synapse(10, 10, ins, None)
        module1 = snn.Neuron(10, ins, None)
        module2 = snn.Synapse(10, 10, ins, None)
        module3 = snn.Neuron(10, ins, None)
        # Test handles
        handle0 = snn.NeuronHandle()
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        # Add and connect
        modules.add(module0, handle0, handle1)
        node1 = modules.add(module1, handle1, handle2)
        node2 = modules.add(module2, handle2, handle3)
        node3 = modules.add(module3, handle3, handle4)
        modules.connect_nodes()
        # Get pre populations
        pre_pop = modules.pre_populations(node3)
        self.assertEqual(len(pre_pop), 1)
        self.assertEqual(pre_pop[0], node1)
        pre_pop = modules.pre_populations(node2)
        self.assertEqual(len(pre_pop), 1)
        self.assertEqual(pre_pop[0], node1)

        # Multiple pre-populations
        modules = ModuleManager()
        ins = snn.Instance()
        # Test modules
        module1 = snn.Neuron(10, ins, None)
        module2 = snn.Synapse(10, 10, ins, None)
        module3 = snn.Neuron(10, ins, None)
        module4 = snn.Synapse(10, 10, ins, None)
        module5 = snn.Neuron(10, ins, None)
        # Test handles
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        handle5 = snn.NeuronHandle()
        handle6 = snn.NeuronHandle()
        handle7 = snn.NeuronHandle()
        # Add and connect
        node1 = modules.add(module1, handle1, handle2)
        modules.add(module2, handle2, handle3)
        node3 = modules.add(module3, handle4, handle5)
        modules.add(module4, handle5, handle6)
        node5 = modules.add(module5, (handle3, handle6), handle7)
        modules.connect_nodes()
        # Get pre-populations
        pre_pops = modules.pre_populations(node5)
        self.assertEqual(len(pre_pops), 2)
        self.assertTrue(node1 in pre_pops)
        self.assertTrue(node3 in pre_pops)

        # Recurrent
        modules = ModuleManager()
        ins = snn.Instance()
        # Test modules
        module1 = snn.Neuron(10, ins, None)
        module2 = snn.Synapse(10, 10, ins, None)
        module3 = snn.Neuron(10, ins, None)
        module4 = snn.Synapse(10, 10, ins, None)
        # Test handles
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        handle5 = snn.NeuronHandle()
        # Add and connect
        node1 = modules.add(module1, handle1, handle2)
        node2 = modules.add(module2, handle2, handle3)
        node3 = modules.add(module3, (handle3, handle5), handle4)
        modules.add(module4, handle4, handle5)
        modules.connect_nodes()

        # Get pre-populations
        pre_pops = modules.pre_populations(node3)
        self.assertEqual(len(pre_pops), 2)
        self.assertTrue(node1 in pre_pops)
        self.assertTrue(node3 in pre_pops)

    def test_post_pop(self):
        """ Test get all post-populations """
        # Test Forward
        modules = ModuleManager()
        ins = snn.Instance()
        # Test modules
        module0 = snn.Synapse(10, 10, ins, None)
        module1 = snn.Neuron(10, ins, None)
        module2 = snn.Synapse(10, 10, ins, None)
        module3 = snn.Neuron(10, ins, None)
        # Test handles
        handle0 = snn.NeuronHandle()
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        # Add and connect
        node1 = modules.add(module0, handle0, handle1)
        node2 = modules.add(module1, handle1, handle2)
        modules.add(module2, handle2, handle3)
        node4 = modules.add(module3, handle3, handle4)
        modules.connect_nodes()
        # Get post population
        post_pops = modules.post_populations(node1)
        self.assertEqual(len(post_pops), 1)
        self.assertEqual(post_pops, [node2])
        post_pops = modules.post_populations(node2)
        self.assertEqual(len(post_pops), 1)
        self.assertEqual(post_pops, [node4])

        # Test recurrent
        modules = ModuleManager()
        ins = snn.Instance()
        # Test modules
        module1 = snn.Neuron(10, ins, None)
        module2 = snn.Synapse(10, 10, ins, None)
        module3 = snn.Neuron(10, ins, None)
        module4 = snn.Synapse(10, 10, ins, None)
        # Test handles
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        handle5 = snn.NeuronHandle()
        # Add and connect
        node1 = modules.add(module1, handle1, handle2)
        node2 = modules.add(module2, handle2, handle3)
        node3 = modules.add(module3, (handle3, handle5), handle4)
        modules.add(module4, handle4, handle5)
        modules.connect_nodes()
        # Get post population (one by design)
        post_pops = modules.post_populations(node2)
        self.assertEqual(len(post_pops), 1)
        self.assertEqual(post_pops, [node3])
        post_pops = modules.post_populations(node3)
        self.assertEqual(len(post_pops), 1)
        self.assertEqual(post_pops, [node3])

    def test_set_dropout_mask(self) -> None:
        """ Test dropout masks are set properly """
        # Test neuron followed by dropout
        modules = ModuleManager()
        ins = snn.Instance()
        # Test modules
        module1 = snn.Neuron(10, ins, None)
        module2 = snn.BatchDropout(10, 0.5, ins, None)
        # Test handles
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        # Add and connect
        modules.add(module1, handle1, handle2)
        modules.add(module2, handle2, handle3)
        modules.connect_nodes()

        # Masks should still be None
        self.assertIsNone(module1.mask)
        self.assertIsNone(module2.mask)

        # Set dropout masks
        modules._set_dropout_mask()
        # Masks should now be set
        self.assertIsNotNone(module1.mask)
        self.assertTrue(torch.equal(module1.mask, module2.mask))

        # Test dropout not proceded by neuron
        modules = ModuleManager()
        ins = snn.Instance()
        # Test modules
        module1 = snn.Synapse(10, 10, ins, None)
        module2 = snn.BatchDropout(10, 0.5, ins, None)
        # Test handles
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        # Add and connect
        modules.add(module1, handle1, handle2)
        modules.add(module2, handle2, handle3)
        modules.connect_nodes()

        # Masks should still be None
        self.assertIsNone(module2.mask)

        # Set dropout masks -> should raise
        with self.assertRaises(AssertionError):
            modules._set_dropout_mask()

    def test_forward_input_nodes(self):
        """ Test if input nodes are forwarded properly """
        # Test neuron followed by dropout
        ins = snn.Instance(mock=True)
        # Test modules
        module1 = snn.Synapse(10, 10, ins, lambda x: x)
        module2 = snn.Neuron(10, ins, lambda x: x)
        # Test handles
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()

        modules = ins.modules

        # Add and connect
        node1 = modules.add(module1, handle1, handle2)
        node2 = modules.add(module2, handle2, handle3)

        # Test
        self.assertEqual(node1.input_handle, (handle1,))
        self.assertEqual(len(modules), 2)
        self.assertEqual(len(modules._input_populations), 0)

        # Forward input nodes
        modules._inject_input_nodes(ins)
        modules._forward_input_nodes()

        # Test
        in_pops = list(modules._input_populations.values())
        self.assertEqual(len(modules), 3)
        self.assertEqual(in_pops[0].post, [])
        self.assertEqual(len(in_pops), 1)
        self.assertEqual(in_pops[0].input_handle, (handle1,))
        self.assertEqual((in_pops[0].output_handle,), node1.input_handle)
        self.assertNotEqual(node1.input_handle, (handle1,))
        self.assertEqual(node1.output_handle, handle2)

        # Forward again input nodes
        modules._inject_input_nodes(ins)
        modules._forward_input_nodes()

        # Test -> nothing should have changed besides intermediate handle
        in_pops = list(modules._input_populations.values())
        self.assertEqual(len(modules), 3)
        self.assertEqual(in_pops[0].post, [])
        self.assertEqual(len(in_pops), 1)
        self.assertEqual((in_pops[0].output_handle,), node1.input_handle)
        self.assertNotEqual(node1.input_handle, (handle1,))
        self.assertEqual(node1.output_handle, handle2)

        # Test connect
        modules.connect_nodes()
        self.assertEqual(in_pops[0].pre, [])
        self.assertEqual(in_pops[0].post, [node1])
        self.assertEqual(node1.pre, [in_pops[0]])
        self.assertEqual(node1.post, [node2])
        self.assertEqual(node2.pre, [node1])
        self.assertEqual(node2.post, [])

    def test_update_node(self):
        """ Test nodes handles are updated properly """
        modules = ModuleManager()
        ins = snn.Instance()
        # Test modules
        module = snn.Neuron(10, ins, None)
        # Test handles
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        # Add and connect
        node_pre = modules.add(module, handle1, handle2)

        # Check pre update
        self.assertEqual(node_pre.input_handle, (handle1,))
        self.assertEqual(node_pre.output_handle, handle2)

        # Update handles
        node_post = modules._update_node(module, handle3, handle4)

        # Check post update
        self.assertEqual(node_post.input_handle, (handle3,))
        self.assertEqual(node_post.output_handle, handle4)
        self.assertEqual(node_pre, node_post)

    def test_ordered(self):
        """ Test modules are returned in correct order """
        # Simple feed-forward
        modules = ModuleManager()
        ins = snn.Instance()
        # Test modules
        module1 = snn.Neuron(10, ins, None)
        module2 = snn.Synapse(10, 10, ins, None)
        module3 = snn.Neuron(10, ins, None)
        module4 = snn.Synapse(10, 10, None, ins)
        # Test handles
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        handle5 = snn.NeuronHandle()
        # Add and connect
        node1 = modules.add(module1, handle1, handle2)
        node2 = modules.add(module2, handle2, handle3)
        node3 = modules.add(module3, handle3, handle4)
        node4 = modules.add(module4, handle4, handle5)

        # Connect
        modules.connect_nodes()

        # Order
        ordered_nodes = modules.ordered()

        # Test order
        target_order = [node1, node2, node3, node4]
        self.assertEqual(target_order, ordered_nodes)

        # Simple feed-forward, ordered wrong
        modules = ModuleManager()
        ins = snn.Instance()
        # Test modules
        module1 = snn.Neuron(10, ins, None)
        module2 = snn.Synapse(10, 10, ins, None)
        module3 = snn.Neuron(10, ins, None)
        module4 = snn.Synapse(10, 10, ins, None)
        # Test handles
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        handle5 = snn.NeuronHandle()
        # Add and connect
        node2 = modules.add(module2, handle2, handle3)
        node3 = modules.add(module3, handle3, handle4)
        node4 = modules.add(module4, handle4, handle5)
        node1 = modules.add(module1, handle1, handle2)

        # Connect
        modules.connect_nodes()

        # Order
        ordered_nodes = modules.ordered()

        # Test order
        target_order = [node1, node2, node3, node4]
        self.assertEqual(target_order, ordered_nodes)

        # Feedforward multiple pre
        modules = ModuleManager()
        ins = snn.Instance()

        # Test modules
        module1 = snn.Neuron(10, ins, None)
        module2 = snn.Synapse(10, 10, ins, None)
        module3 = snn.Neuron(10, ins, None)
        module4 = snn.Synapse(10, 10, ins, None)
        # Test handles
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        handle5 = snn.NeuronHandle()
        handle6 = snn.NeuronHandle()
        # Add and connect
        node1 = modules.add(module1, handle1, handle2)
        node2 = modules.add(module2, handle3, handle4)
        node3 = modules.add(module3, (handle2, handle4), handle5)
        node4 = modules.add(module4, handle5, handle6)

        # Connect
        modules.connect_nodes()

        # Order
        ordered_nodes = modules.ordered()

        # Test order
        target_order = [node1, node2, node3, node4]
        self.assertEqual(target_order, ordered_nodes)

        # Feedforward Diamond
        modules = ModuleManager()
        ins = snn.Instance()
        # Test modules
        module1 = snn.Neuron(10, ins, None)
        module2 = snn.Synapse(10, 10, ins, None)
        module3 = snn.Neuron(10, ins, None)
        module4 = snn.Synapse(10, 10, ins, None)
        module5 = snn.Synapse(10, 10, ins, None)
        module6 = snn.Neuron(10, ins, None)
        module7 = snn.Synapse(10, 10, ins, None)
        module8 = snn.Neuron(10, ins, None)
        # Test handles
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        handle5 = snn.NeuronHandle()
        handle6 = snn.NeuronHandle()
        handle7 = snn.NeuronHandle()
        handle8 = snn.NeuronHandle()
        handle9 = snn.NeuronHandle()
        # Add and connect
        node1 = modules.add(module1, handle1, handle2)
        node2 = modules.add(module2, handle2, handle3)
        node3 = modules.add(module3, handle3, handle4)
        node4 = modules.add(module4, handle4, handle5)
        node5 = modules.add(module5, handle2, handle6)
        node6 = modules.add(module6, handle6, handle7)
        node7 = modules.add(module7, handle7, handle8)
        node8 = modules.add(module8, (handle5, handle8), handle9)

        # Connect
        modules.connect_nodes()

        # Order
        ordered_nodes = modules.ordered()

        # Test order
        target_order = [node1, node2, node5, node3, node6, node4, node7, node8]
        self.assertEqual(target_order, ordered_nodes)


if __name__ == "__main__":
    unittest.main()
