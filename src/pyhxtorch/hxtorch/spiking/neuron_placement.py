"""
Defining neuron placement allocator
"""
# pylint: disable=no-member, invalid-name
from typing import Dict, List, Union, Optional
from dlens_vx_v3 import halco


# TODO: Issue: 4007
class NeuronPlacement:
    # TODO: support multi compartment issue #3750
    """
    Tracks assignment of pyNN IDs of HXNeuron based populations to the
    corresponding hardware entities, i.e. LogicalNeuronOnDLS.
    """
    _id_2_ln: Dict[int, halco.LogicalNeuronOnDLS]
    _used_an: List[halco.AtomicNeuronOnDLS]

    def __init__(self):
        self._id_2_ln = {}
        self._used_an = []

    def register_id(
            self,
            neuron_id: Union[List[int], int],
            shape: halco.LogicalNeuronCompartments,
            placement_constraint: Optional[
                Union[List[halco.LogicalNeuronOnDLS],
                      halco.LogicalNeuronOnDLS]] = None):
        """
        Register a new ID to placement
        :param neuron_id: pyNN neuron ID to be registered
        :param shape: The shape of the neurons on hardware.
        :param placement_constraint: A logical neuron or a list of logical
            neurons, each corresponding to one ID in `neuron_id`.
        """
        if not (hasattr(neuron_id, "__iter__")
                and hasattr(neuron_id, "__len__")):
            neuron_id = [neuron_id]
        if placement_constraint is not None:
            self._register_with_logical_neurons(
                neuron_id, placement_constraint)
        else:
            self._register(neuron_id, shape)

    def _register_with_logical_neurons(
            self,
            neuron_id: Union[List[int], int],
            placement_constraint: Union[
                List[halco.LogicalNeuronOnDLS], halco.LogicalNeuronOnDLS]):
        """
        Register neurons with global IDs `neuron_id` with given logical neurons
        `logical_neuron`.
        :param neuron_id: An int or list of ints corresponding to the global
            IDs of the neurons.
        :param placement_constraint: A logical neuron or a list of logical
            neurons, each corresponding to one ID in `neuron_id`.
        """
        if not isinstance(placement_constraint, list):
            placement_constraint = [placement_constraint]
        for idx, ln in zip(neuron_id, placement_constraint):
            if set(self._used_an).intersection(set(ln.get_atomic_neurons())):
                raise ValueError(
                    f"Cannot register LogicalNeuron {ln} since at "
                    + "least one of its compartments are already "
                    + "allocated.")
            self._used_an += ln.get_atomic_neurons()
            assert idx not in self._id_2_ln
            self._id_2_ln[idx] = ln

    def _register(
            self,
            neuron_id: Union[List[int], int],
            shape: halco.LogicalNeuronCompartments):
        """
        Register neurons with global IDs `neuron_id` with creating logical
        neurons implicitly.
        :param neuron_id: An int or list of ints corresponding to the global
            IDs of the neurons.
        :param shape: The shape of the neurons on hardware.
        """
        for idx in neuron_id:
            placed = False
            for anchor in halco.iter_all(halco.AtomicNeuronOnDLS):
                if anchor in self._used_an:
                    continue
                ln = halco.LogicalNeuronOnDLS(shape, anchor)
                fits = True
                for compartment in ln.get_placed_compartments().values():
                    if bool(set(self._used_an) & set(compartment)):
                        fits = False
                if fits:
                    for compartment in ln.get_placed_compartments() \
                            .values():
                        for an in compartment:
                            self._used_an.append(an)
                    self._id_2_ln[idx] = ln
                    placed = True
                    break
            if not placed:
                raise ValueError(f"Cannot register ID {idx}")

    def id2logicalneuron(self, neuron_id: Union[List[int], int]) \
            -> Union[List[halco.LogicalNeuronOnDLS], halco.LogicalNeuronOnDLS]:
        """
        Get hardware coordinate from pyNN int.
        :param neuron_id: pyNN neuron int
        """
        try:
            return [self._id_2_ln[idx] for idx in neuron_id]
        except TypeError:
            return self._id_2_ln[neuron_id]
