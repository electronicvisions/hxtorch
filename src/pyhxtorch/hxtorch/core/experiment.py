"""
Defining basic types to create hw-executable instances
"""
from __future__ import annotations
from typing import (
    Any,
    Dict,
    Tuple,
    Optional,
)
import pylogging as logger

from dlens_vx_v3 import hal
import pygrenade_vx as grenade

from hxtorch import _runtime


class BaseExperiment(grenade.network.abstract.frontend.Experiment):

    def __init__(
        self,
        inter_batch_entry_wait: int,
        *args,
        **kwargs,
    ) -> None:
        self.log = logger.get("hxtorch.core.BaseExperiment")

        self.inter_batch_entry_wait = inter_batch_entry_wait
        self.inter_batch_entry_routing_disabled = True  # pylint: disable=invalid-name
        self._last_run_chip_configs = None
        self.ppu_symbols_read = {}
        self.batch_size = None

        super().__init__(*args, **kwargs)
        self.hooks = {}
        self._an_offset = 0

    def reset(self):
        super().reset()
        self.ppu_symbols_read = {}

    def generate_runtimes(self, runtime) -> Dict[
            grenade.common.TimeDomainOnTopology,
            grenade.common.TimeDomainRuntimes]:
        """
        :param runtime: The runtime of the experiment on hardware in s.
        """
        assert self.batch_size is not None, \
            "Batch size must be set before generating runtimes."
        runtime_in_clocks = int(
            runtime * int(hal.Timer.Value.fpga_clock_cycles_per_us) * 1e6)
        return {
            grenade.common.TimeDomainOnTopology():
            grenade.network.abstract.ClockCycleTimeDomainRuntimes(
                self.batch_size * [grenade.common.Time(runtime_in_clocks)],
                self.inter_batch_entry_wait,
                self.inter_batch_entry_routing_disabled,
            )}

    def add_placement_constraint(
        self,
        size: int,
        placement_constraint
    ):
        # TODO: Make sure this works for partitioned network
        if placement_constraint is not None:
            permutation = self.mapper.neuron_permutation
            ans = [an for ln in placement_constraint
                   for an in ln.get_atomic_neurons()]
            for i, nrn in enumerate(ans):
                old_an_idx = permutation.index(nrn)
                permutation[old_an_idx] = permutation[i + self._an_offset]
                permutation[i + self._an_offset] = nrn
            self.mapper.neuron_permutation = permutation
        self._an_offset += size

    def post_mapping_hook(self):
        """
        Hook to be executed after mapping, but before execution. Can be used to
        set hardware parameters that depend on the mapping.
        """

    # pylint: disable=arguments-renamed
    def run(
        self,
        runtime: Optional[int],
    ) -> Dict[grenade.network.PopulationOnNetwork, Tuple[Any, ...]]:
        """
        Executes the experiment in mock or on hardware using the information
        added to the experiment for a time given by `runtime` and returns a
        dict of hardware data represented as PyTorch data types.

        :param runtime: The runtime of the experiment on hardware in ms.

        :returns: Returns the data map as dict, where the keys are the
            population descriptors and values are tuples of values returned by
            the corresponding module's `post_process` method.
        """
        self.reset()
        self.add_snippet(0, runtime, _runtime.executor)  # pylint: disable=protected-access

        self.post_mapping_hook()

        if _runtime.executor is None:  # pylint: disable=protected-access
            raise RuntimeError("Executor not initialized.")

        super().run(_runtime.executor)  # pylint: disable=protected-access
        results = self.snippets[-2].output_data

        if results.execution_instances\
                .contains(grenade.common.ExecutionInstanceOnExecutor()) \
                and results.execution_instances.get(
                    grenade.common.ExecutionInstanceOnExecutor()) \
                .read_ppu_symbols \
                and results.execution_instances.get(
                    grenade.common.ExecutionInstanceOnExecutor())\
                .read_ppu_symbols[0]:
            self.ppu_symbols_read = results.execution_instances\
                .get(grenade.common.ExecutionInstanceOnExecutor())\
                .read_ppu_symbols[0]

        return results
