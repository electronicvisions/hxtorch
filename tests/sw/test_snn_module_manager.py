"""
Test Modules and Node classes
"""
import unittest
import networkx as nx
import torch
import hxtorch.snn as snn
from hxtorch.snn.backend.module_manager import ModuleManager


class TestModuleManager(unittest.TestCase):
    """ Test Modules object """

    def test_add_node(self):
        """ Test add module """
        # Test add two connected nodes
        modules = ModuleManager()
        module1 = snn.HXModule(None, None)
        module2 = snn.HXModule(None, None)
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        # One module
        self.assertEqual(len(modules.nodes), 0)
        modules.add_node(module1, (handle1,), handle2)
        self.assertEqual(len(modules.nodes), 1)
        self.assertEqual(len(modules._open_sources), 1)
        self.assertEqual(len(modules._open_targets), 1)
        # Add another
        modules.add_node(module2, (handle2,), handle3)
        self.assertEqual(len(modules.nodes), 2)
        self.assertEqual(len(modules._open_sources), 1)
        self.assertEqual(len(modules._open_targets), 1)

        # Test two sources
        modules = ModuleManager()
        module1 = snn.HXModule(None, None)
        module2 = snn.HXModule(None, None)
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        self.assertEqual(len(modules.nodes), 0)
        modules.add_node(module1, (handle1, handle2,), handle3)
        self.assertEqual(len(modules.nodes), 1)
        self.assertEqual(len(modules._open_sources), 2)
        self.assertEqual(len(modules._open_targets), 1)
        # Test add another
        modules.add_node(module2, (handle1,), handle4)
        self.assertEqual(len(modules.nodes), 2)
        self.assertEqual(len(modules._open_sources), 2)
        self.assertEqual(len(modules._open_targets), 2)

        # Test recurrence
        modules = ModuleManager()
        module1 = snn.HXModule(None, None)
        module2 = snn.HXModule(None, None)
        module3 = snn.HXModule(None, None)
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        self.assertEqual(len(modules.nodes), 0)
        modules.add_node(module1, (handle1,), handle2)
        modules.add_node(module2, (handle2,), handle3)
        modules.add_node(module3, (handle3,), handle4)
        modules.add_node(module1, (handle4,), handle2)
        self.assertEqual(len(modules.nodes), 3)
        self.assertEqual(len(modules._open_sources), 1)
        self.assertEqual(len(modules._open_targets), 0)

    def test_add_wrapper(self):
        """ Test add wrapper module """
        modules = ModuleManager()
        module1 = snn.HXModule(None, None)
        module2 = snn.HXModule(None, None)
        wrapper = snn.HXModuleWrapper(None, [module1, module2], None)
        self.assertEqual(len(modules.wrappers), 0)
        modules.add_wrapper(wrapper)
        self.assertEqual(len(modules.wrappers), 1)
        self.assertEqual(modules.wrappers[wrapper], "w_0")

    def test_get_module_by_id(self):
        """ Test get_module_by_id returns the module """
        modules = ModuleManager()
        module1 = snn.HXModule(None, None)
        module2 = snn.HXModule(None, None)
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        # Add and connect
        modules.add_node(module1, (handle1,), handle2)
        modules.add_node(module2, (handle2,), handle3)
        # Test
        module1_ret = modules.get_module_by_id(0)
        module2_ret = modules.get_module_by_id(1)
        self.assertEqual(module1_ret, module1)
        self.assertEqual(module2_ret, module2)

    def test_get_id_by_module(self):
        """ Test get_id_by_module returns the correct id """
        modules = ModuleManager()
        module1 = snn.HXModule(None, None)
        module2 = snn.HXModule(None, None)
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        # Add and connect
        modules.add_node(module1, (handle1,), handle2)
        modules.add_node(module2, (handle2,), handle3)
        # Test
        id1_ret = modules.get_id_by_module(module1)
        id2_ret = modules.get_id_by_module(module2)
        self.assertEqual(id1_ret, 0)
        self.assertEqual(id2_ret, 1)

    def test_clear(self):
        """ Test clear removes nodes """
        modules = ModuleManager()
        self.assertEqual(len(modules.nodes), 0)
        module1 = snn.HXModule(None, None)
        module2 = snn.HXModule(None, None)
        in_handle1 = snn.NeuronHandle()
        out_handle1 = snn.NeuronHandle()
        in_handle2 = snn.NeuronHandle()
        out_handle2 = snn.NeuronHandle()
        modules.add_node(module1, (in_handle1,), out_handle1)
        modules.add_node(module2, (in_handle2,), out_handle2)
        self.assertEqual(len(modules.nodes), 2)
        modules.clear()
        self.assertEqual(len(modules.nodes), 2)
        self.assertEqual(len(modules.graph.nodes()), 0)

    def test_handle_inputs(self):
        """ Test handle_inputs """
        modules = ModuleManager()
        experiment = snn.Experiment(mock=True)
        module1 = snn.Synapse(10, 11, None, None)
        module2 = snn.Synapse(12, 13, None, None)
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        # Add and connect
        modules.add_node(module1, (handle1,), handle2)
        modules.add_node(module2, (handle2, handle3), handle4)
        modules._handle_inputs(experiment)
        self.assertEqual(len(modules._inputs), 2)
        self.assertEqual(set([module1, module2]), set(modules._inputs.keys()))

        # Clear -> input modules should be reused
        modules.clear()
        self.assertEqual(set([module1, module2]), set(modules._inputs.keys()))
        modules.add_node(module1, (handle1,), handle2)
        modules.add_node(module2, (handle2, handle3), handle4)
        modules._handle_inputs(experiment)
        self.assertEqual(len(modules._inputs), 2)
        self.assertEqual(set([module1, module2]), set(modules._inputs.keys()))

    def test_handle_dropout_mask(self) -> None:
        """ Test dropout masks are set properly """
        # Test neuron followed by dropout
        modules = ModuleManager()
        experiment = snn.Experiment(mock=True)
        module1 = snn.Neuron(10, experiment, None)
        module2 = snn.BatchDropout(10, 0.5, experiment, None)
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        # Add and connect
        modules.add_node(module1, (handle1,), handle2)
        modules.add_node(module2, (handle2,), handle3)
        modules.hw_graph = nx.DiGraph(modules.graph)
        modules._handle_inputs(experiment)
        self.assertIsNone(module1.mask)
        self.assertIsNone(module2.mask)
        # Set dropout masks
        modules._handle_dropout_mask()
        self.assertIsNotNone(module1.mask)
        self.assertTrue(torch.equal(module1.mask, module2.mask))

        # Test dropout not proceded by neuron
        modules = ModuleManager()
        module1 = snn.Synapse(10, 10, None, None)
        module2 = snn.BatchDropout(10, 0.5, None, None)
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        modules.add_node(module1, (handle1,), handle2)
        modules.add_node(module2, (handle2,), handle3)
        modules.hw_graph = nx.DiGraph(modules.graph)
        modules._handle_inputs(experiment)
        self.assertIsNone(module2.mask)
        # Set dropout masks -> should raise
        with self.assertRaises(TypeError):
            modules._handle_dropout_mask()

    def test_handle_wrappers(self) -> None:
        """ Test wrappers are handles properly """
        modules = ModuleManager()
        module1 = snn.HXModule(None, None)
        module2 = snn.HXModule(None, None)
        module3 = snn.HXModule(None, None)
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        # Add and connect
        modules.add_node(module1, (handle1,), handle2)
        modules.add_node(module2, (handle2,), handle3)
        modules.add_node(module3, (handle3,), handle4)
        modules.add_node(module1, (handle4,), handle2)
        wrapper = snn.HXModuleWrapper(
            None, [module1, module2, module3], None)
        modules.add_wrapper(wrapper)
        self.assertEqual(len(modules.graph.nodes()), 4)
        modules._handle_wrappers()
        self.assertEqual(len(modules.graph.nodes()), 2)
        self.assertEqual(set(modules.graph.nodes()), set(["w_0", "s_0_0"]))
        self.assertEqual(modules.graph.nodes["w_0"]["sources"], [handle1])
        self.assertEqual(
            modules.graph.nodes["w_0"]["targets"], [handle2, handle3, handle4])

    def test_order(self) -> None:
        """ Test ordering of modules """
        # Test raises if cycle is present
        modules = ModuleManager()
        module1 = snn.HXModule(None, None)
        module2 = snn.HXModule(None, None)
        module3 = snn.HXModule(None, None)
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        # Add and connect
        modules.add_node(module1, (handle1,), handle2)
        modules.add_node(module2, (handle2,), handle3)
        modules.add_node(module3, (handle3,), handle4)
        modules.add_node(module1, (handle4,), handle2)
        with self.assertRaises(ValueError):
            modules._order()

        # Test without wrapper
        modules = ModuleManager()
        module1 = snn.HXModule(None, None)
        module2 = snn.HXModule(None, None)
        module3 = snn.HXModule(None, None)
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        handle5 = snn.NeuronHandle()
        # Add and connect
        modules.add_node(module1, (handle1,), handle2)
        modules.add_node(module2, (handle2, handle3), handle4)
        modules.add_node(module3, (handle4,), handle5)
        nodes = modules._order()
        self.assertEqual(len(nodes), 3)
        self.assertEqual(
            nodes, [(module1, (handle1,), handle2),
                    (module2, (handle2, handle3), handle4),
                    (module3, (handle4,), handle5)])
        # Reverse source order
        modules = ModuleManager()
        # Add and connect
        modules.add_node(module1, (handle1,), handle2)
        modules.add_node(module2, (handle3, handle2), handle4)
        modules.add_node(module3, (handle4,), handle5)
        nodes = modules._order()
        self.assertEqual(len(nodes), 3)
        self.assertEqual(
            nodes, [(module1, (handle1,), handle2),
                    (module2, (handle3, handle2), handle4),
                    (module3, (handle4,), handle5)])

        # Test with wrapper
        modules = ModuleManager()
        module1 = snn.HXModule(None, None)
        module2 = snn.HXModule(None, None)
        module3 = snn.HXModule(None, None)
        module4 = snn.HXModule(None, None)
        module5 = snn.HXModule(None, None)
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        handle5 = snn.NeuronHandle()
        handle6 = snn.NeuronHandle()
        # Add and connect
        modules.add_node(module1, (handle1,), handle2)
        modules.add_node(module2, (handle2,), handle3)
        modules.add_node(module3, (handle3,), handle4)
        modules.add_node(module4, (handle4,), handle5)
        modules.add_node(module2, (handle5,), handle3)
        modules.add_node(module5, (handle4,), handle6)
        wrapper = snn.HXModuleWrapper(
            None, [module2, module3, module4], None)
        modules.add_wrapper(wrapper)
        modules._handle_wrappers()
        nodes = modules._order()
        self.assertEqual(len(nodes), 3)
        self.assertEqual(
            nodes, [(module1, (handle1,), handle2),
                    (wrapper, (handle2,), (handle3, handle4, handle5)),
                    (module5, (handle4,), handle6)])

    def test_has_module(self):
        """ Test if manager has module """
        modules = ModuleManager()
        module1 = snn.HXModule(None, None)
        module2 = snn.HXModule(None, None)
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        # Add and connect
        modules.add_node(module1, (handle1,), handle2)
        modules.add_node(module2, (handle3,), handle4)
        self.assertTrue(modules.has_module(module1))
        self.assertFalse(modules.has_module(snn.HXModule(None, None)))

    def test_print(self):
        """ Test manager is printed properly """
        modules = ModuleManager()
        module1 = snn.Synapse(10, 12, None, None)
        module2 = snn.HXModule(None, None)
        module3 = snn.HXModule(None, None)
        module4 = snn.HXModule(None, None)
        module5 = snn.HXModule(None, None)
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        handle5 = snn.NeuronHandle()
        handle6 = snn.NeuronHandle()
        modules.add_node(module1, (handle1,), handle2)
        modules.add_node(module2, (handle2,), handle3)
        modules.add_node(module3, (handle3,), handle4)
        modules.add_node(module4, (handle4,), handle5)
        modules.add_node(module2, (handle5,), handle3)
        modules.add_node(module5, (handle4,), handle6)
        wrapper = snn.HXModuleWrapper(
            None, [module2, module3, module4], None)
        modules.add_wrapper(wrapper)
        modules.hw_graph = nx.DiGraph(modules.graph)
        modules._handle_inputs(snn.Experiment())
        modules._handle_wrappers()
        print(modules)

    def test_get_populations(self):
        """ Test get populations """
        experiment = snn.Experiment()
        modules = ModuleManager()
        module1 = snn.InputNeuron(12, experiment=experiment)
        module2 = snn.Synapse(10, 12, experiment=experiment, func=None)
        module3 = snn.Neuron(10, experiment=experiment, func=None)
        module4 = snn.BatchDropout(10, 0.5, experiment=experiment)
        handle1 = snn.NeuronHandle()
        handle2 = snn.NeuronHandle()
        handle3 = snn.NeuronHandle()
        handle4 = snn.NeuronHandle()
        handle5 = snn.NeuronHandle()
        modules.add_node(module1, (handle1,), handle2)
        modules.add_node(module2, (handle2,), handle3)
        modules.add_node(module3, (handle3,), handle4)
        modules.add_node(module4, (handle4,), handle5)
        modules.hw_graph = nx.DiGraph(modules.graph)
        source_pops = modules.source_populations(module2)
        target_pops = modules.target_populations(module2)
        self.assertEqual(source_pops, [module1])
        self.assertEqual(target_pops, [module3])


if __name__ == "__main__":
    unittest.main()
