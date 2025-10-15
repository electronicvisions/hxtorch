"""
Defining basic types to create hw-executable instances
"""
# pylint: disable=no-member, invalid-name
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Tuple,
    Dict,
)
import pylogging as logger

from dlens_vx_v3 import lola
import pygrenade_vx as grenade

from hxtorch.core.experiment import BaseExperiment
from hxtorch.spiking.backend.module_manager import ModuleManager
from hxtorch.spiking.modules.types.population import Population


if TYPE_CHECKING:
    from hxtorch.spiking.handle import Handle
    from hxtorch.spiking.modules.hx_module import HXBaseModule
    from hxtorch.spiking.modules.hx_module_wrapper import HXModuleWrapper


class Experiment(BaseExperiment):

    """ Experiment class for describing experiments on hardware """

    _population_types = Population

    def __init__(
        self,
        *args,
        mock: bool = False,
        dt: float = 1e-6,
        inter_batch_entry_wait: int = 0,
        **kwargs,
    ):
        super().__init__(
            inter_batch_entry_wait,
            *args,
            **kwargs,
        )
        self.modules = ModuleManager()
        self.mock = mock
        self.dt = dt
        self.log = logger.get("hxtorch.spiking.Experiment")
        self.runtime_in_s = 0

    @property
    def batch_size(self) -> int:
        sizes = [
            handle.spikes.shape[1] for handle in self.modules.input_data()]
        assert all(sizes)
        return sizes[0]

    @batch_size.setter
    def batch_size(self, value):
        pass

    def get_source_handle(self, module) -> None:
        """ Generate external input events """
        return [
            e["handle"] for _, _, e in self.modules.graph.in_edges(
                self.modules.get_id_by_module(module), data=True)].pop()

    def connect(self, module: HXBaseModule,
                input_handles: Tuple[Handle, ...],
                output_handle: Handle) -> Handle:
        """
        Add an module to the experiment and connect it to other experiment
        modules via input and output handles.

        :param module: The HXModule to add to the experiment.
        :param input_handles: The TensorHandle serving as input to the module
            (its obsv_state).
        :param output_handle: The TensorHandle outputted by the module,
            serving as input to subsequent HXModules.
        """
        return self.modules.add_node(module, input_handles, output_handle)

    def connect_wrapper(self, wrapper: HXModuleWrapper):
        """
        Add a wrapper module to the experiment and assign it to the experiments
        modules. In the PyTorch graph the individual module functions assigned
        to the wrapper are then bypassed and only the wrapper's `forward_func`
        is considered when building the PyTorch graph. This functionality is of
        interest if several modules have cyclic dependencies and need to be
        represented by one PyTorch function.

        :param wrapper: The HWModuleWrapper to add to the experiment.
        """
        # Unique modules
        assert len(set(wrapper.modules)) == len(wrapper.modules)

        # Check if modules are already existent
        for other_wrapper in self.modules.wrappers:
            if other_wrapper.contains(wrapper.modules):
                raise ValueError(
                    "You tried to register a wrapper with a group of modules "
                    + "that are partially registered in another group.")

        self.modules.add_wrapper(wrapper)

    def post_mapping_hook(self):
        for module in self.modules.nodes:
            if hasattr(module, "override_hw_params"):
                module.override_hw_params(self.snippets[-2])

    def run(self, runtime: float | None):
        """
        Executes the experiment in mock or on hardware using the information
        added to the experiment for a time given by `runtime` and returns a
        dict of hardware data represented as PyTorch data types.

        :param runtime: The runtime of the experiment on hardware in ms.

        :returns: Returns the data map as dict, where the keys are the
            population descriptors and values are tuples of values returned by
            the corresponding module's `post_process` method.
        """
        if not self.mock:
            self.runtime_in_s = runtime * self.dt

        # Preprocess layer
        self.modules.pre_process(self)

        # In mock-mode nothing to do here
        if self.mock:
            return None

        results = super().run(self.runtime_in_s)

        # TODO: Extend to more execution instances
        self._last_run_chip_configs = results.execution_instances.get(
            grenade.common.ExecutionInstanceOnExecutor()
        ).pre_execution_chips

        return results

    @property
    def last_run_chip_configs(self) -> Dict[
        grenade.common.ChipOnConnection,
        lola.Chip
    ]:
        return self._last_execution_instances
