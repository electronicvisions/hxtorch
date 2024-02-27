"""
Definition of ExecutionInstance, wrapping grenade.common.ExecutionInstanceID,
and providing functionality for chip instance configuration
"""
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod
from pathlib import Path

from dlens_vx_v3 import lola, sta
import pygrenade_vx as grenade
import pylogging as logger

from hxtorch.spiking.neuron_placement import NeuronPlacement
from hxtorch.spiking.utils import calib_helper


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
    def prepare_static_config(self) -> None:
        """ Prepare the static configuration of the instance """

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
            input_loopback: bool = False) -> None:
        """
        :param input_loopback: Record input spikes and use them for gradient
            calculation. Depending on link congestion, this may or may not be
            beneficial for the calculated gradient's precision.
        """
        super().__init__()

        self.log = logger.get(f"hxtorch.spiking.execution_instance.{self}")

        self._calib_path = calib_path
        if calib_path is not None:
            self.chip = self.load_calib(calib_path)

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
        # If no calib path is given we load spiking nightly calib
        self.log.INFO(f"Loading calibration from {calib_path}")
        self.chip = calib_helper.chip_from_file(calib_path)
        return self.chip

    def prepare_static_config(self):
        """ Prepare the static configuration of the instance """
        # If chip is still None we load default nightly calib
        if self.chip is None:
            self.log.INFO(
                "No chip object present. Using chip object with default "
                + "nightly calib.")
            self.chip = self.load_calib(calib_helper.nightly_calib_path())

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
