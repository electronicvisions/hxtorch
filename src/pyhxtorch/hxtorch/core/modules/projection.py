""" Define base population and projection classes for BSS-2 """
from __future__ import annotations

from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Union,
    List,
    Tuple,
    Dict,
    Literal,
    NamedTuple,
)

import numpy as np

from pyhalco_hicann_dls_vx_v3 import CompartmentOnLogicalNeuron

import pygrenade_vx as grenade
import pygrenade_vx.network.abstract as gabstract
import pygrenade_vx.common as gcommon
from pygrenade_vx.network.abstract.frontend import (
    ExperimentElement,
    ExperimentSnippet,
)

from hxtorch import logger
from hxtorch.core.plasticity_rule import PlasticityRule
from hxtorch.core.modules.hx_module import HXBaseModule
from hxtorch.core.parameter import ParameterType
from hxtorch.core.experiment import BaseExperiment

if TYPE_CHECKING:
    from pyhalco_hicann_dls_vx_v3 import DLSGlobal
    from hxtorch.core.modules.population import Population as BasePopulation

ModuleParameterType = Union[ParameterType, float, int]
ReceptorType = Literal["excitatory", "inhibitory"]
ReceptorMode = Literal["excitatory", "inhibitory", "signed"]


class ProjectionConnection(NamedTuple):
    idx_pre: int
    idx_post: int
    weight: float


# pylint: disable=abstract-method
class Projection(
    HXBaseModule,
    ExperimentElement,
):
    """ Base class for projections on BSS-2 """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    log = logger.get("hxtorch.core.modules.Projection")

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        in_features: int,
        out_features: int,
        experiment: BaseExperiment,
        chip_coordinate: Dict[
            grenade.common.ExecutionInstanceID,
            DLSGlobal
        ] | None = None,
        plasticity_rule: PlasticityRule | None = None,
        receptor: ReceptorType | Tuple[ReceptorType, ReceptorType]
            | List[ReceptorType] | None = None,
    ) -> None:
        """
        :param experiment: Experiment to append layer to.
        :param in_features: Size of input dimension.
        :param out_features: Size of output dimension.
        :param experiment: Experiment to append layer to.
        :param chip_coordinate: Chip coordinate this module is placed on.
        :param plasticity_rule: Plasticity rule adjusting this synapse.
        :param: receptor: Receptor type of the synapse. Can be 'excitatory',
            'inhibitory' or ('excitatory', inhibitory') for a signed synapse.
        """
        ExperimentElement.__init__(self, experiment)
        super().__init__(experiment, chip_coordinate)
        self.in_features = in_features
        self.out_features = out_features
        self._plasticity_rule = plasticity_rule
        self.plasticity_rule_descriptor = None

        def _normalize_receptor_mode(
            value: ReceptorType | Tuple[ReceptorType, ReceptorType]
            | List[ReceptorType] | None,
            arg_name: str,
        ) -> ReceptorMode | None:
            """Normalize receptor selector into single mode or signed 'both'."""
            if value is None:
                return None

            if isinstance(value, str):
                if value in {"excitatory", "inhibitory"}:
                    return value
                raise ValueError(
                    "Unsupported receptor value. Use 'excitatory' or "
                    "'inhibitory', or pass receptor=('excitatory', "
                    "'inhibitory') for signed behavior."
                )

            if isinstance(value, (list, tuple)):
                receptor_set = set(value)
                if receptor_set == {"excitatory", "inhibitory"}:
                    return "signed"
                if receptor_set == {"excitatory"}:
                    return "excitatory"
                if receptor_set == {"inhibitory"}:
                    return "inhibitory"
                raise ValueError(
                    f"Unsupported {arg_name} collection. Use ['excitatory'], "
                    "['inhibitory'], or ['excitatory', 'inhibitory']."
                )

            raise ValueError(
                f"Unsupported {arg_name} type. Use str, list, or tuple."
            )

        receptor_mode = _normalize_receptor_mode(receptor, "receptor")

        # One receptor means unsigned. Both receptors means signed.
        signed_flag = receptor_mode not in {"excitatory", "inhibitory"}

        if signed_flag and plasticity_rule is not None:
            raise ValueError(
                "Plasticity rules require unsigned projections. Set "
                "receptor='excitatory' or receptor='inhibitory'."
            )

        self._signed_projection = signed_flag
        self._receptor = receptor_mode

    def _generate_plasticity_rule(self) -> gabstract.PlasticityRule:
        recording = None
        projection_shapes = []
        projection_shapes.append(
            grenade.common.CuboidMultiIndexSequence(
                [len(self.get_connections())]
            )
        )
        population_shapes = []
        plasticity_rule = grenade.network.abstract.PlasticityRule(
            recording,
            grenade.network.abstract.PlasticityRule.ID(int(self.descriptor)),
            population_shapes,
            projection_shapes,
            grenade.common.TimeDomainOnTopology(),
        )
        return plasticity_rule

    @abstractmethod
    def get_connections(self) \
            -> Tuple[List[grenade.network.Connection], ...]:
        """ """

    @abstractmethod
    def source_population(self) -> BasePopulation:
        """ """

    @abstractmethod
    def target_population(self) -> BasePopulation:
        """ """

    def add_to_topology(
        self,
        experiment: grenade.abstract.frontend.Experiment.Snippet,
    ):
        # get pre and post populations
        pre = self.source_population()
        post = self.target_population()

        if pre.descriptor is None or post.descriptor is None:
            return False

        connections = self.get_connections()

        # Add excitatory connections
        connections_points = np.empty((len(connections), 2), dtype=int)
        for i, (idx_pre, idx_post, _) in enumerate(connections):
            connections_points[i, 0] = idx_pre
            connections_points[i, 1] = idx_post

        connections_segment = grenade.common.ListMultiIndexSequence([])
        connections_segment.from_numpy(
            connections_points,
            [grenade.common.CellOnPopulationDimensionUnit(),
             grenade.common.CellOnPopulationDimensionUnit()]
        )
        connections_segment_proj_0 = \
            connections_segment.distinct_projection({0})
        connections_segment_proj_1 = \
            connections_segment.distinct_projection({1})

        use_unsigned_projection = not self._signed_projection
        if use_unsigned_projection:
            vertex = grenade.common.Projection(
                gabstract.UncalibratedSynapse(),
                gabstract.UncalibratedSynapse.ParameterSpace(
                    [gabstract.UncalibratedSynapse.Weight(63)
                     for c in connections]),
                gcommon.SequenceConnector(
                    connections_segment_proj_0,
                    connections_segment_proj_1,
                    connections_segment),
                grenade.common.TimeDomainOnTopology(),
            )
        else:
            vertex = grenade.common.Projection(
                gabstract.UncalibratedSignedSynapse(
                    grenade.common.ReceptorOnCompartment(0),
                    grenade.common.ReceptorOnCompartment(1),
                ),
                gabstract.UncalibratedSignedSynapse.ParameterSpace(
                    [gabstract.UncalibratedSignedSynapse.Weight(63)
                     for c in connections]),
                gcommon.SequenceConnector(
                    connections_segment_proj_0,
                    connections_segment_proj_1,
                    connections_segment),
                grenade.common.TimeDomainOnTopology(),
            )

        if self.descriptor and experiment.topology.contains(self.descriptor):
            experiment.topology.clear_vertex(self.descriptor)
            experiment.topology.set(self.descriptor, vertex)
        else:
            self.descriptor = experiment.topology.add_vertex(vertex)

        # add in-edge
        in_edge = grenade.common.Edge(
            connections_segment_proj_0.cartesian_product(
                grenade.common.ListMultiIndexSequence([
                    grenade.common.MultiIndex(
                        # TODO Allow multi-compartment neurons
                        [int(CompartmentOnLogicalNeuron())])],
                    [grenade.common.CompartmentOnNeuronDimensionUnit()])),
            connections_segment_proj_0,
        )
        experiment.topology.add_edge(
            pre.descriptor, self.descriptor, in_edge,
        )

        # add out-edge
        if use_unsigned_projection:
            # TODO: Is this convention? Test it
            if self._receptor == "excitatory":
                receptor = 0
            elif self._receptor == "inhibitory":
                receptor = 1
            else:
                raise ValueError(
                    "Unsigned projections require receptor to be "
                    "'excitatory' or 'inhibitory'.")
            out_edge = grenade.common.Edge(
                connections_segment_proj_1,
                connections_segment_proj_1.cartesian_product(
                    grenade.common.ListMultiIndexSequence([
                        grenade.common.MultiIndex(
                            [int(CompartmentOnLogicalNeuron()),
                             int(
                                 grenade.common.ReceptorOnCompartment(receptor)
                            )]
                        )],
                        [grenade.common.CompartmentOnNeuronDimensionUnit(),
                         grenade.common.ReceptorOnCompartmentDimensionUnit()])
                )
            )
        else:
            out_edge = grenade.common.Edge(
                connections_segment_proj_1.cartesian_product(
                    grenade.common.ListMultiIndexSequence([
                        grenade.common.MultiIndex(
                            [int(grenade.common.ReceptorOnCompartment(0))]),
                        grenade.common.MultiIndex(
                            [int(grenade.common.ReceptorOnCompartment(1))])],
                        [grenade.common.ReceptorOnCompartmentDimensionUnit()])
                ),
                connections_segment_proj_1.cartesian_product(
                    grenade.common.ListMultiIndexSequence([
                        grenade.common.MultiIndex(
                            [int(CompartmentOnLogicalNeuron()),
                             int(grenade.common.ReceptorOnCompartment(0))]),
                        grenade.common.MultiIndex(
                            [int(CompartmentOnLogicalNeuron()),
                             int(grenade.common.ReceptorOnCompartment(1))])],
                        [grenade.common.CompartmentOnNeuronDimensionUnit(),
                         grenade.common.ReceptorOnCompartmentDimensionUnit()])
                )
            )
        experiment.topology.add_edge(
            self.descriptor, post.descriptor, out_edge,
        )

        self.log.TRACE("Added projection with descriptor: ", self.descriptor)
        self.changed_topology = True

        if self._plasticity_rule is None:
            return True

        # self._plasticity_rule.projections = self.descriptor

        if self.plasticity_rule_descriptor is not None and (
                experiment.topology.contains(self.plasticity_rule_descriptor)):
            experiment.topology.set(
                self.plasticity_rule_descriptor,
                self._generate_plasticity_rule(),
            )
        else:
            self.plasticity_rule_descriptor = experiment.topology.add_vertex(
                self._generate_plasticity_rule(),
            )

        edge = grenade.common.Edge(
            grenade.common.CuboidMultiIndexSequence([len(connections)]),
            grenade.common.CuboidMultiIndexSequence([len(connections)]),
            1 if use_unsigned_projection else 0,
            0,
        )

        experiment.topology.add_edge(
            self.descriptor,
            self.plasticity_rule_descriptor,
            edge,
        )

        return True

    def add_to_input_data(
        self,
        experiment: ExperimentSnippet,
        snippet_begin_time,
        snippet_end_time,
    ):
        experiment.input_data.ports.set(
            (self.descriptor, 1),
            self._generate_parameterization(),
        )
        self.changed_input_data = True

        if self._plasticity_rule is not None:
            input_ports = len(
                experiment.topology.get(
                    self.plasticity_rule_descriptor
                ).get_input_ports()
            )
            experiment.input_data.ports.set(
                (self.plasticity_rule_descriptor, input_ports - 1),
                self.generate_plasticity_rule_dynamics(
                    snippet_begin_time,
                    snippet_end_time,
                )
            )
            experiment.input_data.ports.set(
                (self.plasticity_rule_descriptor, input_ports - 2),
                self.generate_plasticity_rule_parameterization(),
            )

    def extract_output_data(
        self,
        experiment: List[grenade.Experiment.Snippet],
    ):
        self._recording_data = []
        if self._plasticity_rule is None:
            return
        for snippet in experiment:
            if snippet.output_data.ports.contains(
                    (self.plasticity_rule_descriptor, 0)):
                self._recording_data.append(
                    snippet.output_data.ports.get(
                        (self.plasticity_rule_descriptor, 0)
                    ).data
                )
            else:
                self._recording_data.append(None)

    def _generate_parameterization(self) \
            -> grenade.network.abstract.Projection.Parameterization:
        connections = self.get_connections()
        if not self._signed_projection:
            if self._receptor == "inhibitory":
                weights = [max(0, -int(c.weight)) for c in connections]
            elif self._receptor == "excitatory":
                weights = [max(0, int(c.weight)) for c in connections]
            else:
                raise ValueError(
                    "Unsigned projections require receptor to be "
                    "'excitatory' or 'inhibitory'."
                )
            params = gabstract.UncalibratedSynapse.ParameterSpace.\
                Parameterization(
                    [gabstract.UncalibratedSynapse.Weight(weight)
                     for weight in weights]
                )
        else:
            params = gabstract.UncalibratedSignedSynapse.ParameterSpace.\
                Parameterization(
                    [gabstract.UncalibratedSignedSynapse.Weight(int(c.weight))
                     for c in connections]
                )
        self.log.TRACE(
            f"Added {len(connections)} to projection {self.descriptor}"
        )
        return params

    def generate_plasticity_rule_dynamics(
        self,
        snippet_begin_time,
        snippet_end_time,
    ) -> gabstract.PlasticityRule.Dynamics:
        return gabstract.PlasticityRule.Dynamics(
            self._plasticity_rule.timer.to_grenade(
                snippet_begin_time,
                snippet_end_time,
            ),
            1,
        )

    def generate_plasticity_rule_parameterization(self) \
            -> gabstract.PlasticityRule.Parameterization:
        return gabstract.PlasticityRule.Parameterization(
            self._plasticity_rule.generate_kernel()
        )
