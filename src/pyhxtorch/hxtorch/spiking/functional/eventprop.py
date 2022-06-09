# pylint: disable=abstract-method, too-many-locals
from typing import NamedTuple, Tuple, Optional
import torch


class EventPropNeuron(torch.autograd.Function):
    """
    Define gradient using adjoint code (EventProp) from norse
    """

    # Allow redefining builtin for PyTorch consistency.
    # Allow names z (spikes), v (membrane), i (current) and T (time dimension
    #   length).
    # Allow different argument params to use dt, tau_mem etc.
    # pylint: disable=redefined-builtin, invalid-name, arguments-differ
    @staticmethod
    def forward(ctx, input: torch.Tensor, params: NamedTuple, dt: float,
                hw_data: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor]:
        r"""
        Forward function returning hardware data if given or otherwise
        integrating LIF neuron dynamics, therefore generating spikes at
        positions > 0.

        :param input: Weighted input spikes in shape (2, batch, time, neurons).
            input[0] holds graded spikes, input[1] holds zero-tensor with same
            shape as placeholder to allow backpropagation of two (batch, time,
            neurons)-shaped tensors.
        :param params: CUBALIFParams object holding neuron parameters.
        :param dt: Step width of integration.
        :param hw_data: Optionally available observables from hardware.

        TODO: Issue 4044. The current LIF implementation differs from
              `hxtorch.spiking.functional.cuba_lif_integration`.

        :returns: Returns the spike trains, membrane and current traces.
            All tensors are of shape (batch, time, neurons).
        """
        # If hardware observables are given, return them directly.
        dev = input.device
        if hw_data is not None:
            ctx.extra_kwargs = {"params": params, "dt": dt}
            hw_data = tuple(data.to(dev) if data is not None
                            else None for data in hw_data)
            ctx.save_for_backward(input, *hw_data)
            return hw_data

        # Otherwise integrate the neuron dynamics in software
        T, bs, ps = input[0].shape
        z, i, v = torch.zeros(bs, ps).to(dev), torch.zeros(bs, ps).to(dev), \
            torch.empty(bs, ps).fill_(params.leak).to(dev)
        spikes, current, membrane = [z], [i], [v]
        for ts in range(T - 1):
            # Current
            i = i * (1 - dt / params.tau_syn) + input[0][ts]
            current.append(i)

            # Membrane
            dv = dt / params.tau_mem * (params.leak - v + i)
            v = dv + v

            # Spikes
            spike = torch.gt(
                v - params.threshold, 0.0).to((v - params.threshold).dtype)
            z = spike

            # Reset
            v = (1 - z.detach()) * v + z.detach() * params.reset

            # Save data
            spikes.append(z)
            membrane.append(v)

        spikes = torch.stack(spikes)
        membrane = torch.stack(membrane)
        current = torch.stack(current)

        ctx.save_for_backward(input, spikes, membrane, current)
        ctx.extra_kwargs = {"params": params, "dt": dt}

        return spikes, membrane, current

    # pylint: disable=invalid-name, unused-argument
    @staticmethod
    def backward(ctx, grad_spikes: torch.Tensor, grad_membrane: torch.Tensor,
                 grad_current: torch.Tensor) \
            -> Tuple[Optional[torch.Tensor], ...]:
        r"""
        Implements 'EventProp' for backward.

        :param grad_spikes: Gradient with respect to output spikes.
        :param grad_membrane: Gradient with respect to membrane trace.
            (Not considered in EventProp algorithm, therefore not used further)
        :param grad_current: Gradient with respect to current.
            (Not considered in EventProp algorithm, therefore not used further)

        :returns: Gradient given by adjoint function lambda_i of current.
        """

        # input and layer data
        input_current = ctx.saved_tensors[0][0]
        T, _, _ = input_current.shape
        z = ctx.saved_tensors[1]
        params = ctx.extra_kwargs["params"]
        dt = ctx.extra_kwargs["dt"]

        # adjoints
        lambda_v = torch.zeros_like(input_current)
        lambda_i = torch.zeros_like(input_current)

        if ctx.saved_tensors[3] is not None:
            i = ctx.saved_tensors[3]
        else:
            i = torch.zeros_like(z)
            # compute current
            for ts in range(T - 1):
                i[ts + 1] = \
                    i[ts] * (1 - dt / params.tau_syn) + input_current[ts]

        for ts in range(T - 1, 0, -1):
            dv_m = params.leak - params.threshold + i[ts - 1]
            dv_p = params.leak - params.reset + i[ts - 1]

            lambda_i[ts - 1] = lambda_i[ts] + dt / \
                params.tau_syn * (lambda_v[ts] - lambda_i[ts])
            lambda_v[ts - 1] = lambda_v[ts] * (1 - dt / params.tau_mem)

            output_term = z[ts] / dv_m * grad_spikes[ts]
            output_term[torch.isnan(output_term)] = 0.0
            output_term[torch.isinf(output_term)] = 0.0

            jump_term = z[ts] * dv_p / dv_m
            jump_term[torch.isnan(jump_term)] = 0.0
            jump_term[torch.isinf(jump_term)] = 0.0

            lambda_v[ts - 1] = (
                (1 - z[ts]) * lambda_v[ts - 1]
                + jump_term * lambda_v[ts - 1]
                + output_term
            )

        return torch.stack((lambda_i * params.tau_syn,
                            lambda_v - lambda_i)), None, None, None


class EventPropSynapse(torch.autograd.Function):
    """
    Synapse function for proper gradient transport when using EventPropNeuron.
    """

    # pylint: disable=arguments-differ, redefined-builtin
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor,
                _: torch.Tensor = None) -> torch.Tensor:
        r"""
        This should be used in combination with EventPropNeuron. Apply linear
        to input using weight and use a stacked output in order to be able to
        return correct terms according to EventProp to previous layer and
        weights.

        :param input: Input spikes in shape (batch, time, in_neurons).
        :param weight: Weight in shape (out_neurons, in_neurons).
        :param _: Bias, which is unused here.

        :returns: Returns stacked tensor holding weighted spikes and
            tensor with zeros but same shape.
        """
        ctx.save_for_backward(input, weight)
        output = input.matmul(weight.t())
        return torch.stack((output, torch.zeros_like(output)))

    # pylint: disable=arguments-differ, redefined-builtin
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) \
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Split gradient_output coming from EventPropNeuron and return
        Input gradient and weight gradient (adjoint at spike times):
            W (\lambda_{v} - \lambda_{i})
            - \tau_{s} \lambda_{i} z

        :param grad_output: Backpropagated gradient with shape (2, batch, time,
            out_neurons). grad_output[0] holds gradients to be propagated to
            weight, grad_output[1] holds gradients to be propagated to previous
            neuron layer.

        :returns: Returns gradients with respect to input, weight and bias.
        """

        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output[1].matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = \
                grad_output[0].transpose(0, 1).transpose(1, 2).matmul(
                    input.transpose(0, 1))

        return grad_input, grad_weight, None


# pylint: disable=redefined-builtin
def eventprop_synapse(input: torch.Tensor, weight: torch.Tensor,
                      _: torch.Tensor = None) -> torch.Tensor:
    return EventPropSynapse.apply(input, weight)


# pylint: disable=redefined-builtin, invalid-name
def eventprop_neuron(input: torch.Tensor, params: NamedTuple, dt: float,
                     hw_data: Optional[torch.Tensor]) -> Tuple[torch.Tensor]:
    return EventPropNeuron.apply(input, params, dt, hw_data)
