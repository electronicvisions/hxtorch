"""
Defining tensor handles able to hold references to tensors for lazy assignment
after hardware data acquisition
"""
from typing import List, Optional
from collections import OrderedDict
import torch


class HandleMeta(type):

    """ Meta class to create Handle types with properties """

    def __new__(mcs, clsname, bases, dict_) -> None:
        """
        Create a new instance of the given class.
        """
        for item in dict_["_carries"]:

            def getter(self, name: str = item) -> torch.Tensor:
                return self.data[name]

            def setter(self, data: torch.Tensor, name: str = item) -> None:
                self.data[name] = data

            dict_.update({item: property(fget=getter, fset=setter)})

        return super().__new__(mcs, clsname, bases, dict_)


class TensorHandle(metaclass=HandleMeta):

    """
    Base class for HX tensor handles. New tensor handles have to be derived
    from this class. The name of tensors the tensor handle 'carries' has to be
    indicated in the class member '_carries'. For all elements in this list a
    property is created implicitly.
    """

    _carries: List[str] = []

    def __init__(self) -> None:
        """
        Instantiate a new HX handle holding references to torch tensors.
        """
        self.data = OrderedDict()

    def holds(self, name: str) -> bool:
        """
        Checks whether the tensor handle already holds a tensor with key
        `name`.

        :param name: Key of reference to tensor.
        :return: Returns a bool indicating whether the data present at key
            `name` is not None.
        """
        return self.data.get(name) is not None

    def put(self, *tensors, **kwargs) -> None:
        """
        Fill the tensor handle with actual data given by tensors or kwargs. If
        tensors are given as positional arguments the tensors are assigned in
        the order given by class member '_carries'. Keyword arguments are
        assigned to the corresponding key. Therefore, the key has to be in
        '_carries.'

        :param tensors: Tensors which are assigned to the tensor handle. The
            given tensors are associated with the elements in '_carries' in
            successive order.

        :keyword param kwargs: Assigns the items in kwargs to the elements in
            the tensor handle associated with the corresponding kwargs keys.
        """
        keys = list(self.data.keys())
        assert len(keys) >= len(tensors), \
            "Encountered more 'tensors' than keys in the handle."

        for i, tensor in enumerate(tensors):
            self.data[keys[i]] = tensor

        for key, value in kwargs.items():
            assert key in keys, "Encountered unknown key."
            self.data[key] = value

    def clone(self, handle: "TensorHandle") -> None:
        """
        Copy contents for `handle` to `this` handle. `This` handle must contain
        a data field for eacht data field in `handle`.

        :param handle: The handle to clone.
        """
        for key, value in handle.data.items():
            self.data[key] = value

    def clear(self) -> None:
        """
        Set all data in handle to 'None'.
        """
        for key in self.data.keys():
            self.data[key] = None


class NeuronHandle(TensorHandle):

    """ Specialization for HX neuron observables """

    _carries = ["spikes", "membrane_cadc", "current", "membrane_madc"]

    def __init__(self, spikes: Optional[torch.Tensor] = None,
                 membrane_cadc: Optional[torch.Tensor] = None,
                 current: Optional[torch.Tensor] = None,
                 membrane_madc: Optional[torch.Tensor] = None) -> None:
        """
        Instantiate a neuron handle able to hold spike and membrane tensors.

        :param spikes: Optional spike tensor.
        :param membrane_cadc: Optional membrane tensor, holding CADC
            recordings.
        :param current: Optional current tensor, holding synaptic current.
        :param membrane_madc: Optional membrane tensor, holding MADC
            recordings.
        """
        super().__init__()
        self.spikes = spikes
        self.membrane_cadc = membrane_cadc
        self.current = current
        self.membrane_madc = membrane_madc


class ReadoutNeuronHandle(TensorHandle):

    """ Specialization for HX neuron observables """

    _carries = ["membrane_cadc", "current", "membrane_madc"]

    def __init__(self, membrane_cadc: Optional[torch.Tensor] = None,
                 current: Optional[torch.Tensor] = None,
                 membrane_madc: Optional[torch.Tensor] = None) -> None:
        """
        Instantiate a readout neuron handle able to hold MADC, current, and
        CADC data. This handle defines the CADC values as the observable state,
        this can be changed by deriving this class.

        :param membrane_cadc: Optional membrane tensor, holding CADC
            recordings.
        :param current: Optional current tensor, holding synaptic current.
        :param membrane_madc: Optional membrane tensor, holding MADC
            recordings.
        """
        super().__init__()
        self.membrane_cadc = membrane_cadc
        self.current = current
        self.membrane_madc = membrane_madc


class SynapseHandle(TensorHandle):

    """ Specialization for HX synapses """

    _carries = ["graded_spikes"]

    def __init__(self, graded_spikes: Optional[torch.Tensor] = None) -> None:
        """
        Instantiate a synapse handle able to hold a graded spikes tensors as
        input to neurons.

        :param graded_spikes: Optional tensor holding graded spikes.
        """
        super().__init__()
        self.graded_spikes = graded_spikes
