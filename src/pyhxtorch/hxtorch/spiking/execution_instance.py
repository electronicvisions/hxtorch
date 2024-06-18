"""
Definition of ExecutionInstance, wrapping grenade.common.ExecutionInstanceID,
and providing functionality for chip instance configuration
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

from dlens_vx_v3 import lola, sta
import pygrenade_vx as grenade
import pylogging as logger
from _hxtorch_core import init_hardware, release_hardware

# pylint: disable=import-error, no-name-in-module
from calix import calibrate
from calix.spiking import SpikingCalibTarget, SpikingCalibOptions
from calix.spiking.neuron import NeuronCalibTarget

from hxtorch.spiking.neuron_placement import NeuronPlacement
from hxtorch.spiking.utils import calib_helper
if TYPE_CHECKING:
    from hxtorch.spiking.modules import HXModule


class ExecutionInstances(set):
    """ List of ExecutionInstances """

    @property
    def chips(self) -> Dict[grenade.common.ExecutionInstanceID, lola.Chip]:
        """
        Getter for chips.

        :return: The Chip objects for each instance in a
            dict[ExecutionInstanceID, Chip]
        """
        return {inst.ID: inst.chip for inst in self}

    @property
    def cadc_recordings(self) -> Dict[
            grenade.common.ExecutionInstanceID,
            List[grenade.network.CADCRecording]]:
        """
        Getter for CADC recordings.

        :return: The CADC recordings for each instance in a
            dict[ExecutionInstanceID, CADCRecording]
        """
        return {inst.ID: inst.cadc_recordings() for inst in self
                if len(inst.cadc_neurons) != 0}

    @property
    def playback_hooks(self) -> Dict[
            grenade.common.ExecutionInstanceID,
            grenade.signal_flow.ExecutionInstanceHooks]:
        """
        Getter for all playback hooks assigned to each execution instance ID.

        :return: The playback hooks for each instance in a
            dict[ExecutionInstanceID, ExecutionInstanceHooks]
        """
        return {inst.ID: inst.generate_playback_hooks() for inst in self}


class BaseExecutionInstance(ABC):
    """ ExecutionInstance base class """

    def __init__(self) -> None:
        self.chip: Optional[lola.Chip] = None
        self._id = grenade.common.ExecutionInstanceID(id(self))
        self.cadc_neurons: Optional[
            Dict[int, grenade.network.CADCRecording.Neuron]] = {}
        self.modules: List[HXModule] = None

    def __hash__(self) -> int:
        return hash(self._id)

    def __repr__(self):
        return f"{self.ID}"

    def __str__(self):
        return f"ExecutionInstance(ID={int(self.ID)})"

    @property
    def ID(self):  # pylint: disable=invalid-name
        return self._id

    @abstractmethod
    def calibrate(self) -> None:
        """ Handle the calibration of the instance """

    @abstractmethod
    def cadc_recordings(self) -> grenade.network.CADCRecording:
        """
        Return the instance's ``CADCRecording`` object, holding all neurons
        that are to be recorded in this instance.

        :return: The ``grenade.network.CADCRecoding`` object
        """

    @abstractmethod
    def generate_playback_hooks(self) \
            -> grenade.signal_flow.ExecutionInstanceHooks:
        """
        Generate a ``ExecutionInstanceHooks`` object for the given
        execution instance, injected in ``grenade.run``.

        :return: The execution instance's playback hook.
        """


class ExecutionInstance(BaseExecutionInstance):

    def __init__(
            self,
            calib_path: Optional[Union[Path, str]] = None,
            calib_cache_dir: Optional[Union[Path, str]] = None,
            input_loopback: bool = False) -> None:
        """
        :param input_loopback: Record input spikes and use them for gradient
            calculation. Depending on link congestion, this may or may not be
            beneficial for the calculated gradient's precision.
        """
        super().__init__()

        self.log = logger.get(f"hxtorch.spiking.execution_instance.{self}")

        # Load chip objects
        self.calib_path = calib_path
        self.calib = None
        self.chip = None
        if calib_path is not None:
            self.chip = self.load_calib(self.calib_path)
        self.calib_cache_dir = calib_cache_dir

        self.input_loopback = input_loopback
        self.record_cadc_into_dram: Optional[bool] = None

        # State
        self.id_counter = 0
        self.has_madc_recording = False
        self.neuron_placement: NeuronPlacement = NeuronPlacement()

        # Static configs
        self.injection_pre_static_config = None
        self.injection_pre_realtime = None
        self.injection_post_realtime = None
        self.injection_inside_realtime_begin = None
        self.injection_inside_realtime = None
        self.injection_inside_realtime_end = None

    def load_calib(self, calib_path: Optional[Union[Path, str]] = None):
        """
        Load a calibration from path ``calib_path`` and apply to the
        experiment's chip object. If no path is specified a nightly calib is
        applied.

        :param calib_path: The path to the calibration. It None, the nightly
            calib is loaded.
        :return: Returns the chip object for the given calibration.
        """
        assert calib_path is not None
        self.log.INFO(f"Loading calibration from {calib_path}")
        if str(calib_path).endswith(".pkl"):
            self.calib = calib_helper.calib_from_calix_native(calib_path)
            self.chip = self.calib.to_chip()
        else:
            self.chip = calib_helper.chip_from_file(calib_path)
        self.calib_path = calib_path
        return self.chip

    def calibrate(self):
        """
        Manage calibration of chip instance. In case a calibration path is
        provided, parameters for the modules are loaded if possible. If no
        calibration path is given, calibration targets are attempted to be
        loaded from the modules parameter objects and the chip will be
        calibrated accordingly. If no parameter changes are detected, the chip
        will not be recalibrated.
        """
        if not any(m.calib_changed_since_last_run() for m in self.modules
                   if hasattr(m, "calib_changed_since_last_run")):
            return

        execute_calib = False
        if not self.calib_path:
            self.log.TRACE("No calibration path present. Try to infer "
                           + "parameters for calibrations")
            # gather calibration information
            target = \
                SpikingCalibTarget(
                    neuron_target=NeuronCalibTarget.DenseDefault)
            # initialize `synapse_dac_bias` and `i_synin_gm` as `None` to allow
            # check for different values in different populations
            target.neuron_target.synapse_dac_bias = None
            target.neuron_target.i_synin_gm = np.array([None, None])
            # if any neuron module has params, use for calibration
            for module in self.modules:
                if hasattr(module, "calibration_from_params"):
                    self.log.INFO(f"Add calib params of '{module}'.")
                    neurons = self.neuron_placement.id2logicalneuron(
                        module.unit_ids)
                    module.calibration_from_params(target, neurons)
                    execute_calib = True

        # otherwise use nightly calibration
        if not self.calib_path and not execute_calib:
            self.log.INFO(
                "No chip object present and no parameters for calibration "
                + "provided. Using chip object with default nightly calib.")
            self.chip = self.load_calib(
                calib_helper.nightly_calix_native_path())

        if not execute_calib:
            # Make sure experiment holds chip config
            assert self.chip is not None
            # EA: TODO: Probably we should do this in each case
            if self.calib is None:
                self.log.WARN(
                    "Tried to infer params from calib but no readable "
                    "calibration present. This might be because a coco binary "
                    "was indicated as calibration file. Skipped.")
            else:
                self.log.INFO(
                    "Try to infer params from loaded calibration file...")
                for module in self.modules:
                    if hasattr(module, "params_from_calibration"):
                        neurons = self.neuron_placement.id2logicalneuron(
                            module.unit_ids)
                        module.params_from_calibration(
                            self.calib.target, neurons)
        else:
            release_hardware()
            self.log.INFO("Calibrating...")
            self.calib = calibrate(
                target, SpikingCalibOptions(), self.calib_cache_dir)
            dumper = sta.PlaybackProgramBuilderDumper()
            self.calib.apply(dumper)
            self.chip = sta.convert_to_chip(dumper.done())
            init_hardware()
            self.log.INFO("Calibration finished... ")
        self.log.TRACE(f"Prepared static config of {self}.")

    def cadc_recordings(self) -> grenade.network.CADCRecording:
        """
        Return the instance's ``CADCRecording`` object, holding all neurons
        that are to be recorded in this instance.

        :return: The ``grenade.network.CADCRecoding`` object
        """
        assert len(self.cadc_neurons)
        cadc_recording = grenade.network.CADCRecording()
        if self.record_cadc_into_dram:
            cadc_recording.placement_on_dram = True
        cadc_recording.neurons = [nrn[0] for nrn in self.cadc_neurons.values()]
        return cadc_recording

    def generate_playback_hooks(self) \
            -> grenade.signal_flow.ExecutionInstanceHooks:
        """
        Handle config injected into grenade (not supported yet).

        :return: Returns the execution instance's (empty) playback hooks
            injected into ``grenade.run``.
        """
        pre_static_config = sta.PlaybackProgramBuilder()
        pre_realtime = sta.PlaybackProgramBuilder()
        inside_realtime_begin = sta.PlaybackProgramBuilder()
        inside_realtime = sta.AbsoluteTimePlaybackProgramBuilder()
        inside_realtime_end = sta.PlaybackProgramBuilder()
        post_realtime = sta.PlaybackProgramBuilder()

        if self.injection_pre_static_config is not None:
            pre_static_config.copy_back(self.injection_pre_static_config)

        if self.injection_pre_realtime is not None:
            pre_realtime.copy_back(self.injection_pre_realtime)

        if self.injection_inside_realtime_begin is not None:
            inside_realtime_begin.copy_back(
                self.injection_inside_realtime_begin)

        if self.injection_inside_realtime is not None:
            inside_realtime.copy(self.injection_inside_realtime)

        if self.injection_inside_realtime_end is not None:
            inside_realtime_end.copy_back(self.injection_inside_realtime_end)

        if self.injection_post_realtime is not None:
            post_realtime.copy_back(self.injection_post_realtime)

        return grenade.signal_flow.ExecutionInstanceHooks(
            pre_static_config, pre_realtime, inside_realtime_begin,
            inside_realtime, inside_realtime_end, post_realtime)
