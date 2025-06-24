"""
Defining tensor handles able to hold references to tensors for lazy assignment
after hardware data acquisition
"""
from abc import ABC
from typing import Type, Optional
from dataclasses import make_dataclass, is_dataclass, astuple, field, fields
import torch

handle_register = {}


class Handle(ABC):
    """
    Factory for classes which are to be used as custom handles for observable
    data, depending on the specific observables a module deals with.
    """

    # pylint: disable=invalid-name
    def __new__(cls, *args, **kwargs):
        """
        Instantiate a new HX handle holding references to torch tensors.
        Can be used to directly initialize custom HandleClass with values
        via `kwargs` or pass attribute keys only via `args` as strings
        for construction of a dummy object.
        The latter is only to be used for type generation.
        """
        if args and kwargs and set(args).difference(set(kwargs.keys())):
            raise Exception("Ambiguous Handle construction mode",
                            set(args), set(kwargs.keys()))
        if args:
            args_sorted = tuple(sorted(args))
            attributes = [(key, Optional[torch.Tensor], field(default=None))
                          for key in args_sorted]
        else:
            kwargs = dict(sorted(kwargs.items()))
            attributes = [(key, Optional[torch.Tensor], field(default=value))
                          for key, value in kwargs.items()]

        handle_name = cls.__name__ + "_" \
            + "_".join([str(attr[0]) for attr in attributes])
        doc = "Handle for " + ", ".join([str(attr[0]) for attr in attributes])

        def __eq__(this, other):
            return (is_dataclass(this) and is_dataclass(other)
                    and all(torch.equal(this_element, other_element) if
                            isinstance(this_element, torch.Tensor)
                            and isinstance(other_element, torch.Tensor) else
                            this_element == other_element for
                            (this_element, other_element) in
                            zip(astuple(this), astuple(other))))

        def holds(self, name: str) -> bool:
            """
            Checks whether the tensor handle already holds a tensor with key
            `name`.

            :param name: Key of reference to tensor.
            :return: Returns a bool indicating whether the data present at key
                `name` is not None.
            """
            return (name in [f.name for f in fields(self)]
                    and getattr(self, name) is not None)

        def clone(self, handle) -> None:
            """
            Overwrite contents from `this` with contents from `handle`.

            :param handle: The handle to clone.
            """
            assert ([f.name for f in fields(self)]
                    == [f.name for f in fields(handle)])
            for f in fields(handle):
                setattr(self, f.name, getattr(handle, f.name))

        HandleClass = make_dataclass(
            handle_name, attributes, eq=True, namespace={
                "__eq__": __eq__, "holds": holds, "clone": clone})
        HandleClass.__doc__ = doc
        HandleClass.__str__ = lambda self: HandleClass.__name__ + ": \n\t" \
            + "\n\t".join([str(key) + " = " + str(value)
                          for (key, value) in kwargs.items()])

        # Check if handle class is already existing
        if handle_name in handle_register and args:
            return handle_register[handle_name](*((None,) * len(args)))
        if handle_name in handle_register and kwargs:
            return handle_register[handle_name](**kwargs)
        if handle_name not in handle_register:
            handle_register[handle_name] = HandleClass

        if args and not kwargs:
            return HandleClass(*((None,) * len(args)))
        return HandleClass(**kwargs)


TensorHandle: Type = type(Handle('tensor'))
LIFObservables: Type = type(Handle('spikes', 'membrane_cadc', 'current',
                                   'membrane_madc'))
LIObservables: Type = type(Handle('membrane_cadc', 'current',
                                  'membrane_madc'))
SynapseHandle: Type = type(Handle('graded_spikes'))
