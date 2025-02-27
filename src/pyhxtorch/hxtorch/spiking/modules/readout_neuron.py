"""
Implementing SNN modules
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Tuple, Type, Optional, List, Union
import pylogging as logger

import torch

from dlens_vx_v3 import lola, halco

import hxtorch.spiking.functional as F
from hxtorch.spiking.handle import ReadoutNeuronHandle, SynapseHandle
from hxtorch.spiking.morphology import Morphology
from hxtorch.spiking.modules.neuron import Neuron
from hxtorch.spiking.modules.types import ModuleParameterType
if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment
    from hxtorch.spiking.observables import HardwareObservables
    from hxtorch.spiking.execution_instance import ExecutionInstance

log = logger.get("hxtorch.spiking.modules")


class ReadoutNeuron(Neuron):
    """
    Readout neuron layer

    Caveat:
    For execution on hardware, this module can only be used in conjunction with
    a preceding Synapse module.
    """

    output_type: Type = ReadoutNeuronHandle

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self, size: int,
                 experiment: Experiment,
                 leak: ModuleParameterType = 80,
                 tau_mem: ModuleParameterType = 10e-6,
                 tau_syn: ModuleParameterType = 10e-6,
                 i_synin_gm: ModuleParameterType = 500,
                 membrane_capacitance: ModuleParameterType = 63,
                 synapse_dac_bias: ModuleParameterType = 600,
                 execution_instance: Optional[ExecutionInstance] = None,
                 enable_cadc_recording: bool = True,
                 enable_cadc_recording_placement_in_dram: bool = False,
                 enable_madc_recording: bool = False,
                 record_neuron_id: Optional[int] = None,
                 placement_constraint: Optional[
                     List[halco.LogicalNeuronOnDLS]] = None,
                 trace_offset: Union[Dict[halco.AtomicNeuronOnDLS, float],
                                     torch.Tensor, float] = 0.,
                 trace_scale: Union[Dict[halco.AtomicNeuronOnDLS, float],
                                    torch.Tensor, float] = 1.,
                 cadc_time_shift: int = 0, shift_cadc_to_first: bool = False,
                 interpolation_mode: str = "linear",
                 neuron_structure: Optional[Morphology] = None,
                 **extra_params) -> None:
        """
        Initialize a ReadoutNeuron. This module creates a population of non-
        spiking neurons of size `size` and is equivalent to Neuron when its
        spiking mask is disabled for all neurons.

        The readout neuron is parameterized by the `ModuleParameterType`d
        parameters:
            leak, tau_mem, tau_syn, i_synin_gm, membrane_capacitance,
            synapse_dac_bias
        More infos to the respective parameters on BSS-2 can be found in
        `calix.spiking.neuron.NeuronCalibTarget`. If the parameters are not
        given as `ParameterType`, they are implicitly converted to
        `HXParameter` which provides the same value to the BSS-2 calibration
        (`param.hardware_value`) (and thus the hardware operation state) as to
        the numerical model (`param.model_value`) defined in `forward_func`.
        `MixedHXModelParameter` and `HXTransformedModelParameter` allow using
        different values on BSS-2 and in the numerics. This is useful if the
        dynamic range on hardware and in the numerical model  differ. If so,
        the trace and weight scaling parameters need to be set accordingly in
        order to translate the weights to their corresponding hardware value
        and the hardware measurements into the dynamic range used in the
        numerics.

        :param size: Size of the population.
        :param experiment: Experiment to register the module in.
        :param execution_instance: Execution instance to place to.
        :param leak: The leak potential. Defaults to HXParameter(80).
        :param tau_syn: The synaptic time constant in s. Defaults to
            HXParameter(10e-6).
        :param tau_mem: The membrane time constant in s. Defaults to
            HXParameter(10e-6).
        :param i_synin_gm: A hardware parameter adjusting the hardware neuron
            -specific synaptic efficacy. Defaults to HXParameter(500).
        :param membrane_capacitance: The capacitance of the membrane. The
            available range is 0 to approximately 2.2 pF, represented as 0 to
            63 LSB.
        :param synapse_dac_bias: Synapse DAC bias current that is desired. Can
            be lowered in order to reduce the amplitude of a spike at the input
            of the synaptic input OTA. This can be useful to avoid saturation
            when using larger synaptic time constants. Defaults to
            HXParameter(600).
        :param enable_cadc_recording: Enables or disables parallel sampling of
            the populations membrane trace via the CADC. A maximum sample rate
            of 1.7us is possible.
        :param enable_cadc_recording_placement_in_dram: Whether to place CADC
            recording data into DRAM (period ~6us) or SRAM (period ~2us).
        :param enable_madc_recording: Enables or disables the recording of the
            neurons `record_neuron_id` membrane trace via the MADC. Only a
            single neuron can be recorded. This membrane traces is samples with
            a significant higher resolution as with the CADC.
        :param record_neuron_id: The in-population neuron index of the neuron
            to be recorded with the MADC. This has only an effect when
            `enable_madc_recording` is enabled.
        :param placement_constraint: An optional list of logical neurons
            defining where to place the module`s neurons on hardware.
        :param trace_offset: The value by which the measured CADC traces are
            shifted before the scaling is applied. If this offset is given as
            float the same value is applied to all neuron traces in this
            population. One can also provide a torch tensor holding one offset
            for each individual neuron in this population. The corresponding
            tensor has to be of size `size`. Further, the offsets can be
            supplied in a dictionary where the keys are the hardware neuron
            coordinates and the values are the offsets, i.e.
            Dict[AtomicNeuronOnDLS, float]. The dictionary has to provide one
            coordinate for each hardware neuron represented by this population,
            but might also hold neuron coordinates that do not correspond to
            this layer. The layer-specific offsets are then picked and applied
            implicitly.
        :param trace_scale: The value by which the measured CADC traces are
            scaled after the offset is applied. If this scale is given as
            float all neuron traces are scaled with the same value population.
            One can also provide a torch tensor holding one scale for each
            individual neuron in this population. The corresponding tensor has
            to be of size `size`. Further, the scales can be supplied in a
            dictionary where the keys are the hardware neuron coordinates and
            the values are the scales, i.e. Dict[AtomicNeuronOnDLS, float]. The
            dictionary has to provide one coordinate for each hardware neuron
            represented by this population, but might also hold neuron
            coordinates that do not correspond to this layer. The layer-
            specific scales are then picked and applied implicitly.
        :param cadc_time_shift: An integer indicating by how many time steps
            the CADC values are shifted in time. A positive value shifts later
            CADC samples to earlier times and vice versa for a negative value.
        :param shift_cadc_to_first: A boolean indicating that the first
            measured CADC value is used as an offset. Note, this disables the
            param `trace_offset`.
        :param interpolation_mode: The method used to interpolate the measured
            CADC traces onto the given time grid.
        :param neuron_structure: Structure of the neuron. If not supplied a
            single neuron circuit is used.
        """
        super().__init__(
            size,
            experiment,
            leak=leak,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            i_synin_gm=i_synin_gm,
            membrane_capacitance=membrane_capacitance,
            synapse_dac_bias=synapse_dac_bias,
            execution_instance=execution_instance,
            enable_spike_recording=False,
            enable_cadc_recording=enable_cadc_recording,
            enable_cadc_recording_placement_in_dram=(
                enable_cadc_recording_placement_in_dram
            ),
            enable_madc_recording=enable_madc_recording,
            record_neuron_id=record_neuron_id,
            placement_constraint=placement_constraint,
            trace_offset=trace_offset, trace_scale=trace_scale,
            cadc_time_shift=cadc_time_shift,
            shift_cadc_to_first=shift_cadc_to_first,
            interpolation_mode=interpolation_mode,
            neuron_structure=neuron_structure,
            **extra_params)

    def configure_hw_entity(self, neuron_id: int,
                            neuron_block: lola.NeuronBlock,
                            coord: halco.LogicalNeuronOnDLS) \
            -> lola.NeuronBlock:
        """
        Configures a neuron in the given module with its specific properties.
        The neurons digital event outputs are enabled according to the given
        spiking mask.

        TODO: Additional parameterization should happen here, i.e. with
              population-specific parameters.

        :param neuron_id: In-population neuron index.
        :param neuron_block: The neuron block hardware entity.
        :param coord: Coordinate of neuron on hardware.
        :returns: Configured neuron block.
        """
        self._neuron_structure.implement_morphology(coord, neuron_block)
        self._neuron_structure.disable_spiking(coord, neuron_block)

        if neuron_id == self._record_neuron_id:
            self._neuron_structure.enable_madc_recording(
                coord, neuron_block, self._madc_readout_source)

        return neuron_block

    def post_process(self, hw_data: HardwareObservables, runtime: float) \
            -> Tuple[Optional[torch.Tensor], ...]:
        """
        User defined post process method called as soon as population-specific
        hardware observables are returned. This function has to convert the
        data types returned by grenade into PyTorch tensors. This function can
        be overridden by the user if non-default grenade-PyTorch data type
        conversion is required.
        Note: This function should return Tuple[Optional[torch.Tensor], ...],
              like (cadc or madc,). This should match the
              ReadoutTensorHandle signature.

        :param hw_data: A ``HardwareObservables`` instance holding the
            population's recorded hardware observables.
        :param runtime: The requested runtime of the experiment on hardware in
            s.

        :returns: Returns a tuple of optional torch.Tensors holding the
            hardware data (madc or cadc,)
        """
        # No spikes here
        assert not self._enable_spike_recording
        _, cadc, madc = super().post_process(hw_data, runtime)

        return cadc, madc

    # pylint: disable=redefined-builtin
    def forward_func(self, *input: SynapseHandle,
                     hw_data: Optional[Tuple[torch.Tensor]] = None) \
            -> ReadoutNeuronHandle:
        assert all(handle.graded_spikes is not None for handle in input)
        return ReadoutNeuronHandle(*F.cuba_li_integration(
            tuple(handle.graded_spikes for handle in input),
            leak=self.leak.model_value,
            tau_syn=self.tau_syn.model_value,
            tau_mem=self.tau_mem.model_value,
            hw_data=hw_data,
            dt=self.experiment.dt))


class ReadoutNeuronExp(ReadoutNeuron):
    """
    Neuron layer with exponential Euler intergration scheme.
    Synaptic and memebrane time constant are required to be provided
    as HXTransformedModelParameter(exp(-dt/tau), -dt/ln(tau)).
    This ensures that the correct model and hardware values are being used.
    """

    # pylint: disable=redefined-builtin
    def forward_func(self, *input: SynapseHandle,
                     hw_data: Optional[Tuple[torch.Tensor]] = None) \
            -> ReadoutNeuronHandle:
        return ReadoutNeuronHandle(*F.exp_cuba_li_integration(
            tuple(handle.graded_spikes for handle in input),
            leak=self.leak.model_value,
            tau_syn_exp=self.tau_syn.model_value,
            tau_mem_exp=self.tau_mem.model_value,
            hw_data=hw_data))
