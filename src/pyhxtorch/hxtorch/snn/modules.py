"""
Implementing SNN modules
"""
from typing import Any, Callable, Dict, Tuple, Type, Optional, Union, List
from functools import partial
import inspect
import numpy as np

import torch
import torch.nn.functional as nnF
from torch.nn.parameter import Parameter

from dlens_vx_v3 import lola, hal, halco
import pygrenade_vx as grenade

from _hxtorch._snn import DataHandle, SpikeHandle, CADCHandle, MADCHandle  # pylint: disable=import-error
import hxtorch
import hxtorch.snn.functional as F
from hxtorch.snn.transforms import weight_transforms
from hxtorch.snn.handle import (
    TensorHandle, NeuronHandle, ReadoutNeuronHandle, SynapseHandle)

log = hxtorch.logger.get("hxtorch.snn.modules")


class HXModule(torch.nn.Module):
    """
    PyTorch module supplying basic functionality for building SNNs on HX.
    """

    output_type: Type = TensorHandle

    def __init__(self, instance,
                 func: Union[Callable, torch.autograd.Function]) -> None:
        """
        :param instance: Instance to append layer to.
        :param func: Callable function implementing the module's forward
            functionallity or a torch.autograd.Function implementing the
            module's forward and backward operation.
            TODO: Inform about func args
        """
        super().__init__()

        self._func = func
        self.instance = instance
        self.extra_args: Tuple[Any] = tuple()
        self.extra_kwargs: Dict[str, Any] = dict()
        self.size: int = None
        self._changed_since_last_run = True

    @property
    def changed_since_last_run(self) -> bool:
        """
        Getter for changed_since_last_run.

        :returns: Boolean indicating wether module changed since last run.
        """
        return self._changed_since_last_run

    def reset_changed_since_last_run(self) -> None:
        """
        Reset changed_since_last_run. Sets the corresponding flag to false.
        """
        self._changed_since_last_run = False

    def post_process(self, hw_spikes: Optional[DataHandle],
                     hw_cadc: Optional[DataHandle],
                     hw_madc: Optional[DataHandle]) \
            -> Tuple[Optional[torch.Tensor], ...]:
        """
        This methods needs to be overridden for every derived module that
        demands hardware observables.

        :returns: Hardware data represented as torch.Tensors. Note that
            torch.Tensors are required here to enable gradient flow.
        """
        raise NotImplementedError()

    def _is_autograd_fn(self) -> bool:
        """
        Determines whether the used function `func` is an autograd function or
        not.

        :returns: Returns whether `func` is an autograd function.
        """
        return isinstance(self._func, torch.autograd.function.FunctionMeta)

    def prepare_func(self, hw_data: torch.Tensor) -> Callable:
        """
        Strips `param` and `hw_result` arguments from self._func. This allows
        having a more general signature of self._func in `exec_forward`.

        :param hw_result: HW observables returned from grenade.

        :returns: Returns the memeber 'func(..., params=..., hw_result=...)'
            stripped down to 'func(...).
        """
        self.extra_kwargs.update({"hw_data": hw_data})
        if self._is_autograd_fn():
            class LocalAutograd(self._func):
                pass
            local_func = LocalAutograd
            signature = inspect.signature(local_func.forward)
            for key, value in self.extra_kwargs.items():
                if key in signature.parameters:
                    local_func.forward = partial(
                        local_func.forward, **{key: value})
            return local_func.apply

        signature = inspect.signature(self._func)
        local_func = self._func
        for key, value in self.extra_kwargs.items():
            if key in signature.parameters:
                local_func = partial(local_func, **{key: value})

        return local_func

    # Allow redefinition of builtin in order to be consistent with PyTorch
    # pylint: disable=redefined-builtin
    def forward(self, *input: Union[Tuple[TensorHandle], TensorHandle]) \
            -> TensorHandle:
        """
        Forward method registering layer operation in given instance. Input and
        output references will hold corresponding data as soon as 'hxtorch.run'
        in executed.

        :param input: Reference to TensorHandle holding data tensors as soon
            as required.

        :returns: Returns a Reference to TensorHandle holding result data
            asociated with this layer after 'hxtorch.run' is executed.
        """
        output = self.output_type()
        self.instance.connect(self, input, output)

        return output

    # Allow redefinition of builtin in order to be consistent with PyTorch
    # pylint: disable=redefined-builtin
    def exec_forward(self, input: Union[Tuple[TensorHandle], TensorHandle],
                     output: TensorHandle,
                     hw_data: Optional[Tuple[torch.Tensor]] = tuple()) -> None:
        """
        Inject hardware observables into TensorHandles or execute forward in
        mock-mode.
        """
        # Need tuple to allow for multiple input
        if not isinstance(input, tuple):
            input = (input,)
        input = tuple(handle.observable_state for handle in input)

        if self._is_autograd_fn() and not self.instance.mock:

            class LocalAutograd(self._func):
                @staticmethod
                def forward(  # pylint: disable=dangerous-default-value
                        ctx, *data, extra_kwargs=self.extra_kwargs):
                    ctx.extra_kwargs = extra_kwargs
                    ctx.save_for_backward(
                        *data, *hw_data if hw_data is not None else None)
                    return hw_data

            out = LocalAutograd.apply(*input, *self.extra_args)
        else:
            out = self.prepare_func(hw_data)(*input, *self.extra_args)

        # We need to unpack into `Handle.put` otherwise we get Tuple[Tuple]
        # in `put`, however, func should not be limited to return type 'tuple'
        out = (out,) if not isinstance(out, tuple) else out
        output.put(*out)


class Synapse(HXModule):  # pylint: disable=abstract-method
    """
    Synapse layer

    Caveat:
    For execution on hardware, this module can only be used in conjuction with
    a subsequent Neuron module.
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor
    output_type: Type = SynapseHandle

    # pylint: disable=too-many-arguments
    def __init__(self, in_features: int, out_features: int, instance,
                 func: Union[Callable, torch.autograd.Function] = nnF.linear,
                 device: str = None, dtype: Type = None,
                 transform: Callable = weight_transforms.linear_saturating) \
            -> None:
        """
        TODO: Think about what to do with device here.

        :param in_features: Size of input dimension.
        :param out_features: Size of output dimension.
        :param device: Device to execute on. Only considered in mock-mode.
        :param dtype: Data type of weight tensor.
        :param instance: Instance to append layer to.
        :param func: Callable function implementing the module's forward
            functionallity or a torch.autograd.Function implementing the
            module's forward and backward operation. Required function args:
                [input (torch.Tensor), weight (torch.Tensor)]
        """
        super().__init__(instance=instance, func=func)

        self.in_features = in_features
        self.out_features = out_features
        self.size = out_features

        self.weight = Parameter(
            torch.empty(
                (out_features, in_features), device=device, dtype=dtype))
        self._weight_old = torch.zeros_like(self.weight.data)
        self.weight_transform = transform

        self.reset_parameters(1.0e-3, 1. / np.sqrt(in_features))
        self.extra_args = (self.weight, None)  # No bias

    @property
    def changed_since_last_run(self) -> bool:
        """
        Getter for changed_since_last_run.

        :returns: Boolean indicating wether module changed since last run.
        """
        self._weight_old.to(self.weight.data.device)
        return not torch.equal(self.weight.data,
                               self._weight_old.to(self.weight.data.device)) \
            or self._changed_since_last_run

    def reset_changed_since_last_run(self) -> None:
        """
        Reset changed_since_last_run. Sets the corresponding flag to false.
        """
        self._weight_old = torch.clone(self.weight.data)
        return super().reset_changed_since_last_run()

    def reset_parameters(self, mean: float, std: float) -> None:
        """
        Resets the synapses weights by reinitalization using torch.nn.normal_
        with mean=`mean` and std=`std`.

        :param mean: The mean of the normal distribution used to initialize the
            weights from.
        :param std: The standard deviation of the normal distribution used to
            initialized the weights from.
        """
        torch.nn.init.normal_(self.weight, mean=mean, std=std)

    def register_hw_entity(self) -> None:
        """
        Add the synapse layer to the instances projections.
        """
        self.instance.register_projection(self)

    def add_to_network_graph(self, pre: grenade.PopulationDescriptor,
                             post: grenade.PopulationDescriptor,
                             builder: grenade.NetworkBuilder) \
            -> Tuple[grenade.ProjectionDescriptor, ...]:
        """
        Adds the projection to a grenade network builder by providing the
        population descriptor of the corresponding pre and post population.
        Note: This creates one inhibitory and one excitatory population on
        hardware in order to represent signed hardware weights.

        :param pre: Population descriptor of pre-population.
        :param post: Population descriptor of post-population.
        :param builder: Greande netowrk builder to add projection to.

        :returns: A tuple of grenade ProjectionDescriptors holding the
            descriptors for the excitatory and inhibitory projection.
        """

        weight_transformed = self.weight_transform(
            torch.clone(self.weight.data))

        weight_exc = torch.clone(weight_transformed)
        weight_inh = torch.clone(weight_transformed)
        weight_exc[weight_exc < 0.] = 0
        weight_inh[weight_inh >= 0.] = 0

        # TODO: Make sure this doesn't require rerouting
        connections_exc = hxtorch.snn.weight_to_connection(weight_exc)  # pylint: disable=no-member
        connections_inh = hxtorch.snn.weight_to_connection(weight_inh)  # pylint: disable=no-member

        projection_exc = grenade.Projection(
            grenade.Projection.ReceptorType.excitatory,
            connections_exc, pre, post)
        projection_inh = grenade.Projection(
            grenade.Projection.ReceptorType.inhibitory,
            connections_inh, pre, post)

        exc_descriptor = builder.add(projection_exc)
        inh_descriptor = builder.add(projection_inh)
        log.TRACE(f"Added projection '{self}' to grenade graph.")

        return exc_descriptor, inh_descriptor


class Neuron(HXModule):
    """
    Neuron layer

    Caveat:
    For execution on hardware, this module can only be used in conjuction with
    a preceding Synapse module.
    """

    output_type: Type = NeuronHandle

    # TODO: Integrate into API
    _madc_readout_source: hal.NeuronConfig.ReadoutSource = \
        hal.NeuronConfig.ReadoutSource.membrane
    _cadc_readout_source: lola.AtomicNeuron.Readout.Source \
        = lola.AtomicNeuron.Readout.Source.membrane

    # pylint: disable=too-many-arguments
    def __init__(self, size: int, instance: "Instance",
                 func: Union[Callable, torch.autograd.Function] = F.LIF,
                 params: Optional[Union[F.LIFParams, F.LIParams]] = None,
                 enable_spike_recording: bool = True,
                 enable_cadc_recording: bool = True,
                 enable_madc_recording: bool = False,
                 record_neuron_id: Optional[int] = None,
                 trace_offset: Union[torch.Tensor, float] = 0.,
                 trace_scale: Union[torch.Tensor, float] = 1.,
                 interpolation_mode: str = "linear") -> None:
        """
        Initialize a Neuron. This module creates a population of spiking
        neurons of size `size`. This module has a internal spiking mask, which
        allows to disable the event ouput and spiking recording of specific
        neurons within the layer. This is particularly useful for dropout.

        :param size: Size of the population.
        :param instance: Instance to append layer to.
        :param func: Callable function implementing the module's forward
            functionallity or a torch.autograd.Function implementing the
            module's forward and backward operation. Defaults to `LIF`.
        :param params: Neuron Parameters in case of mock neuron integration of
            for backward path. If func does have a param argument the params
            object will get injected automatically.
        :param enable_spike_recording: Boolean flag to enable or disable spike
            recording. Note, this does not disable the event out put of
            neurons. The event output has to be disabled via `mask`.
        :param enable_cadc_recording: Enables or disables parallel sampling of
            the populations membrane trace via the CADC. A maximum sample rate
            of 1.7us is possible.
        :param enable_madc_recording: Enables or disables the recording of the
            neurons `record_neuron_id` membrane trace via the MADC. Only a
            single neuron can be recorded. This membrane traces is samples with
            a significant higher resolution as with the CADC.
        :param record_neuron_id: The in-population neuron index of the neuron
            to be recorded with the MADC. This has only an effect when
            `enable_madc_recording` is enabled.
        """
        super().__init__(instance=instance, func=func)

        self.size = size
        self.params = params
        self.extra_kwargs.update({"params": params})

        self._enable_spike_recording = enable_spike_recording
        self._enable_cadc_recording = enable_cadc_recording
        self._enable_madc_recording = enable_madc_recording
        self._record_neuron_id = record_neuron_id
        self._mask: Optional[torch.Tensor] = None
        self.unit_ids: Optional[np.ndarray] = None

        self.offset = trace_offset
        self.scale = trace_scale
        self.interpolation_mode = interpolation_mode

    def register_hw_entity(self) -> None:
        """
        Infere neuron ids on hardware and register them.
        """
        self.unit_ids = np.arange(
            self.instance.id_counter, self.instance.id_counter + self.size)
        self.instance.neuron_placement.register_id(self.unit_ids)
        self.instance.id_counter += self.size
        # self.instance.populations.add(self.instance.modules.get_node(self))
        self.instance.register_population(self)

        if self._enable_madc_recording:
            if self.instance.has_madc_recording:
                raise RuntimeError(
                    "Another HXModule already registered MADC recording. "
                    + "MADC recording is only enabled for a "
                    + "single neuron.")
            self.instance.has_madc_recording = True
        log.TRACE(f"Registered hardware  entity '{self}'.")

    @property
    def mask(self) -> None:
        """
        Getter for spike mask.

        :returns: Returns the current spike mask.
        """
        return self._mask

    @mask.setter
    def mask(self, mask: torch.Tensor) -> None:
        """
        Setter for the spike mask.

        :param mask: Spike mask. Must be of shape `(self.size,)`.
        """
        # Mark dirty
        self._changed_since_last_run = True
        self._mask = mask

    @staticmethod
    def create_default_hw_entity() -> lola.AtomicNeuron:
        """
        At the moment, the default neuron is loaded from grenade's ChipConfig
        object, which holds the atomic neurons configured as a calibration is
        loaded in `hxtorch.hardware_init()`.

        TODO: - Needed?
              - Maybe this can return a default neuron, when pop-specific
                calibration is needed.
        """
        return lola.AtomicNeuron()

    def configure_hw_entity(self, neuron_id: int,
                            atomic_neuron: lola.AtomicNeuron) \
            -> lola.AtomicNeuron:
        """
        Configures a neuron in the given layer with its specific properties.
        The neurons digital event outputs are enabled according to the given
        spiking mask.

        TODO: Additional parameterization should happen here, i.e. with
              population-specific parameters.

        :param neuron_id: In-population neuron index.
        :param atomic_neuron: The neurons hardware entity representing the
            neuron with index `neuron_id` on hardware.

        :returns: Returns the AtomicNeuron with population-specific
            configurations appended.
        """
        # configure spike recording
        atomic_neuron.event_routing.analog_output = \
            atomic_neuron.EventRouting.AnalogOutputMode.normal
        atomic_neuron.event_routing.enable_digital = self.mask[neuron_id]

        # configure madc
        if neuron_id == self._record_neuron_id:
            atomic_neuron.readout.enable_amplifier = True
            atomic_neuron.readout.enable_buffered_access = True
            atomic_neuron.readout.source = self._madc_readout_source

        return atomic_neuron

    def add_to_network_graph(self, builder: grenade.NetworkBuilder) \
            -> grenade.PopulationDescriptor:
        """
        Add the layer's neurons to grenades network builder. If
        `enable_spike_recording` is enabled the neuron's spikes are recorded
        according to the layer's spiking mask. If no spiking mask is given all
        neuron spikes will be recorded. Note, the event output of the neurons
        are configured in `configure_hw_entity`.
        If `enable_cadc_recording` is enabled the populations neuron's are
        registered for CADC membrane recording.
        If `enable_madc_recording` is enabled the neuron with in-population
        index `record_neuron_id` will be recording via the MADC. Note, since
        the MADC can only record a single neuron on hardware, other Neuron
        layers registering also MADC recording might overwrite the setting
        here.

        :param builder: Grenade's network builder to add the layer's population
            to.
        :returns: Returns the builder with the population added.
        """
        # Create neuron mask if none is given (no dropout)
        if self._mask is None:
            self._mask = np.ones_like(self.unit_ids, dtype=bool)

        # Enable record spikes according to neuron mask
        if self._enable_spike_recording:
            enable_record_spikes = np.ones_like(self.unit_ids, dtype=bool)
        else:
            enable_record_spikes = np.zeros_like(self.unit_ids, dtype=bool)

        # get neuron coordinates
        coords: List[halco.AtomicNeuronOnDLS] = \
            self.instance.neuron_placement.id2atomicneuron(self.unit_ids)

        # create grenade population
        gpopulation = grenade.Population(coords, list(enable_record_spikes))

        # add to builder
        descriptor = builder.add(gpopulation)

        if self._enable_cadc_recording:
            for in_pop_id, unit_id in enumerate(self.unit_ids):
                neuron = grenade.CADCRecording.Neuron(
                    descriptor, in_pop_id, self._cadc_readout_source)
                self.instance.cadc_recording[unit_id] = neuron

        # No recording registered -> return
        if not self._enable_madc_recording:
            return descriptor

        # add MADC recording
        # NOTE: If two populations register MADC reordings grenade should
        #       throw in the following
        madc_recording = grenade.MADCRecording()
        madc_recording.population = descriptor
        madc_recording.source = self._madc_readout_source
        madc_recording.index = int(self._record_neuron_id)
        builder.add(madc_recording)
        log.TRACE(f"Added population '{self}' to grenade graph.")

        return descriptor

    @staticmethod
    def add_to_input_generator(layer: HXModule,
                               builder: grenade.InputGenerator) -> None:
        pass

    def post_process(self, hw_spikes: Optional[SpikeHandle],
                     hw_cadc: Optional[CADCHandle],
                     hw_madc: Optional[MADCHandle]) \
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

        :param hw_spikes: A SpikeHandle holding the population's spikes
            recorded by grenade as a sparse tensor. This data can be ignored
            for this readout neuron.
        :param hw_cadc: The CADCHandle holding the CADC membrane readout
            events in a sparse tensor.
        :param hw_madc: The MADCHandle holding the MADC membrane readout
            events in a sparse tensor.

        :returns: Returns a tuple of optional torch.Tensors holding the
            hardware data (madc or cadc,)
        """
        spikes, cadc, madc = None, None, None

        # Get cadc samples
        if self._enable_cadc_recording:
            cadc = hw_cadc.to_dense(
                self.instance.dt, mode=self.interpolation_mode)
            # Offset and scale
            cadc *= self.scale
            cadc -= self.offset

        # Get spikes
        if self._enable_spike_recording:
            spikes = hw_spikes.to_dense(self.instance.dt).float()

        # Get madc trace
        if self._enable_madc_recording:
            raise NotImplementedError(
                "MADCHandle to dense torch Tensor is not implemented yet.")

        return spikes, cadc, madc


class ReadoutNeuron(Neuron):
    """
    Readout neuron layer

    Caveat:
    For execution on hardware, this module can only be used in conjuction with
    a preceding Synapse module.
    """

    output_type: Type = ReadoutNeuronHandle

    # pylint: disable=too-many-arguments
    def __init__(self, size: int, instance: "Instance",
                 func: Union[Callable, torch.autograd.Function] = F.LI,
                 params: Optional[F.LIParams] = None,
                 enable_cadc_recording: bool = True,
                 enable_madc_recording: bool = False,
                 record_neuron_id: Optional[int] = None,
                 trace_offset: Union[torch.Tensor, float] = 0.,
                 trace_scale: Union[torch.Tensor, float] = 1.,
                 interpolation_mode: str = "linear") -> None:
        """
        Initialize a ReadoutNeuron. This module creates a population of non-
        spiking neurons of size `size` and is equivalent to Neuron when its
        spiking mask is disabled for all neurons.

        :param size: Size of the population.
        :param instance: Instance to append layer to.
        :param func: Callable function implementing the module's forward
            functionallity or a torch.autograd.Function implementing the
            module's forward and backward operation. Defaults to `LI`.
        :param params: Neuron Parameters in case of mock neuron integration of
            for backward path. If func does have a param argument the params
            object will get injected automatically.
        """
        super().__init__(
            size, instance, func, params, False, enable_cadc_recording,
            enable_madc_recording, record_neuron_id, trace_offset, trace_scale,
            interpolation_mode)

    def configure_hw_entity(self, neuron_id: int,
                            atomic_neuron: lola.AtomicNeuron) \
            -> lola.AtomicNeuron:
        """
        Configures a neuron in the given layer with its specific properties.

        :param neuron_id: In-population neuron index.
        :param atomic_neuron: The neurons hardware entity representing the
            neuron with index `neuron_id` on hardware.

        :returns: Returns the AtomicNeuron with population-specific
            configurations appended.
        """
        # configure spike recording
        atomic_neuron.event_routing.analog_output = \
            atomic_neuron.EventRouting.AnalogOutputMode.normal
        atomic_neuron.event_routing.enable_digital = False

        # disable threshold comparator
        atomic_neuron.threshold.enable = False

        return atomic_neuron

    def post_process(self, hw_spikes: Optional[SpikeHandle],
                     hw_cadc: Optional[CADCHandle],
                     hw_madc: Optional[MADCHandle]) \
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

        :param hw_spikes: A SpikeHandle holding the population's spikes
            recorded by grenade as a sparse tensor. This data can be ignored
            for this readout neuron.
        :param hw_cadc: The CADCHandle holding the CADC membrane readout
            events in a sparse tensor.
        :param hw_madc: The MADCHandle holding the MADC membrane readout
            events in a sparse tensor.

        :returns: Returns a tuple of optional torch.Tensors holding the
            hardware data (madc or cadc,)
        """
        cadc, madc = None, None

        # Get cadc samples
        if self._enable_cadc_recording:
            cadc = hw_cadc.to_dense(
                self.instance.dt, mode=self.interpolation_mode)
            # Offset and scale
            cadc *= self.scale
            cadc -= self.offset

        # Get madc trace
        if self._enable_madc_recording:
            raise NotImplementedError(
                "MADCHandle to dense torch Tensor is not implemented yet.")

        return cadc, madc


class BatchDropout(HXModule):  # pylint: disable=abstract-method
    """
    Batch dropout layer

    Caveat:
    In-place operations on TensorHandles are not supported. Must be placed
    after a neuron layer, i.e. Neuron.
    """

    output_type: Type = NeuronHandle

    def __init__(self, size: int, dropout: float, instance,
                 func: Union[
                     Callable, torch.autograd.Function] = F.batch_dropout) \
            -> None:
        """
        Initialize BatchDropout layer. This layer disables spiking neurons in
        the previous spiking Neuron layer with a probability of `dropout`.
        Note, `size` has to be equal to the size in the corresponding spiking
        layer. The spiking mask is maintained for the whole batch.

        :param size: Size of the population this dropout layer is applied to.
        :param dropout: Probability that a neuron in the precessing layer gets
            disabled during training.
        :param instance: Instance to append layer to.
        :param func: Callable function implementing the module's forward
            functionallity or a torch.autograd.Function implementing the
            module's forward and backward operation. Defaults to
            `batch_dropout`.
        """
        super().__init__(instance=instance, func=func)

        self.size = size
        self._dropout = dropout
        self._mask: Optional[torch.Tensor] = None

    def set_mask(self) -> None:
        """
        Creates a new random dropout mask, applied to the spiking neurons in
        the previous module.
        If `module.eval()` dropout will be disabled.

        :returns: Returns a random boolen spike mask of size `self.size`.
        """
        if self.training:
            self.mask = (torch.rand(self.size) > self._dropout)
        else:
            self.mask = torch.ones(self.size).bool()
        self.extra_args = (self._mask,)

        return self._mask

    @property
    def mask(self) -> None:
        """
        Getter for spike mask.

        :returns: Returns the current spike mask.
        """
        return self._mask

    @mask.setter
    def mask(self, mask: torch.Tensor) -> None:
        """
        Setter for the spike mask.

        :param mask: Spike mask. Must be of shape `(self.size,)`.
        """
        # Mark dirty
        self._changed_since_last_run = True
        self._mask = mask


class InputNeuron(HXModule):
    """
    Spike source generating spikes at the times [ms] given in the spike_times
    array.
    """
    output_type: Type = NeuronHandle

    def __init__(self, size: int, instance) -> None:
        """
        Instanziate a INputNeuron. This module serves as an External
        Population for input injection and is created within `snn.Instance`
        if not present in the considerd model.
        This module performes an identity mapping when `forward` is called.

        :param size: Number of input neurons.
        :param instance: Instance to which this module is assigned.
        """
        super().__init__(instance, func=F.input_neuron)
        self.size = size

    def register_hw_entity(self) -> None:
        """
        Register instance in member `instance`.
        """
        self.instance.register_population(self)

    def add_to_network_graph(self, builder: grenade.NetworkBuilder) \
            -> grenade.PopulationDescriptor:
        """
        Adds instance to grenade's network builder.

        :param builder: Grenade network builder to add extrenal population to.
        :returns: External population descriptor.
        """
        # create grenade population
        gpopulation = grenade.ExternalPopulation(self.size)
        # add to builder
        descriptor = builder.add(gpopulation)
        log.TRACE(f"Added Input Population: {self}")

        return descriptor

    def add_to_input_generator(self, input: NeuronHandle,  # pylint: disable=redefined-builtin
                               builder: grenade.InputGenerator) -> None:
        """
        Add the neurons events represented by this instance to grenades input
        generator.

        :param input: Dense spike tensor. These spikes are implictely converted
            to spike times. TODO: Allow sparse tensors.
        :param builder: Grenade's input generator to append the events to.
        """
        if isinstance(input, tuple):
            assert len(input) == 1
            input, = input  # unpack

        # tensor to spike times
        # maybe support sparse input tensor?
        # TODO: Expects ms relative. Align to time handling.
        spike_times = hxtorch.snn.tensor_to_spike_times(  # pylint: disable=no-member
            input.spikes, dt=self.instance.dt / 1e-3)
        descriptor = self.instance.modules.get_node(self).descriptor
        builder.add(spike_times, descriptor)

    def post_process(self, hw_spikes: Optional[DataHandle],
                     hw_cadc: Optional[DataHandle],
                     hw_madc: Optional[DataHandle]) \
            -> Tuple[Optional[torch.Tensor], ...]:
        pass