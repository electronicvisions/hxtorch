"""
EventProp autograd function, 'copied' from norse, i.e.
https://github.com/norse/norse/blob/master/norse/torch/functional/adjoint/lif_adjoint.py
"""
from typing import NamedTuple, Tuple, Optional
import torch


class EventPropNeuron(torch.autograd.Function):
    # pylint: disable=line-too-long
    """
    Define gradient using adjoint code (EventProp) from norse
    """
    # Allow redefining builtin for PyTorch consistancy
    # Allow names z (spikes), v (membrane), i (current) and T (time dimension length)
    # Allow different argument params to use dt, tau_mem_inv etc.
    # pylint: disable=redefined-builtin, invalid-name, arguments-differ
    @staticmethod
    def forward(ctx, input: torch.Tensor,
                params: NamedTuple) -> Tuple[torch.Tensor]:
        """
        Forward function, generating spikes at positions > 0.

        :param input: Weighted input spikes in shape (2, batch, time, neurons).
            The 2 at dim 0 comes from stacked output in EventPropSynapse.
        :param params: LIFParams object holding neuron prameters.

        :returns: Returns the spike trains and membrane trace.
            Both tensors are of shape (batch, time, neurons)
        """
        input_current = input[0]
        z, i, v = (
            torch.zeros(input_current.shape[0], input_current.shape[2]),
            torch.zeros(input_current.shape[0], input_current.shape[2]),
            torch.empty(input_current.shape[0],
                        input_current.shape[2]).fill_(params.v_leak),
        )
        spikes, current, membrane = [z], [i], [v]
        T = input_current.shape[1]
        for ts in range(T - 1):
            # Current
            i = i * (1 - params.dt * params.tau_syn_inv) + input_current[:, ts]
            current.append(i)

            # Membrane
            dv = params.dt * params.tau_mem_inv * (params.v_leak - v + i)
            v = dv + v

            # Spikes
            spike = torch.gt(v - params.v_th, 0.0).to((v - params.v_th).dtype)
            z = spike

            # Reset
            v = (1 - z.detach()) * v + z.detach() * params.v_reset

            # Save data
            spikes.append(z)
            membrane.append(v)
        forward_result = (
            torch.stack(spikes).transpose(0, 1),
            torch.stack(membrane).transpose(0, 1)
        )
        ctx.current = torch.stack(current).transpose(0, 1)
        ctx.save_for_backward(input, *forward_result)
        ctx.extra_kwargs = {"params": params}

        return (*forward_result,)

    # TODO: On HW `TypeError: backward() takes 3 positional arguments but 4
    #       were given` occurs. Maybe because of overwriting autograd func in
    #       HXModule.exec_forward, which returns `hw_data`.
    # pylint: disable=invalid-name
    @staticmethod
    def backward(ctx, grad_spikes: torch.Tensor,
                 _: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Implements 'EventProp' for backward.

        :param grad_spikes: Backpropagted gradient wrt output spikes.
        :param _: backpropagated gradient wrt to membrane trace (not used)

        :returns: Gradient given by adjoint function lambda_i of current
        """
        # input and layer data
        input = ctx.saved_tensors[0]
        input_current = input[0]
        z = ctx.saved_tensors[1]
        params = ctx.extra_kwargs["params"]

        # adjoints
        lambda_v, lambda_i = torch.zeros_like(z), torch.zeros_like(z)

        try:
            i = ctx.current
        except AttributeError:
            i = torch.zeros_like(z)
            # compute current
            for ts in range(z.shape[1] - 1):
                i[:, ts + 1] = \
                    i[:, ts] * (1 - params.dt * params.tau_syn_inv) \
                    + input_current[:, ts]

        for ts in range(z.shape[1] - 1, 0, -1):
            dv_m = params.v_leak - params.v_th + i[:, ts - 1]
            dv_p = i[:, ts - 1]

            lambda_i[:, ts - 1] = lambda_i[:, ts] + params.dt * \
                params.tau_syn_inv * (lambda_v[:, ts] - lambda_i[:, ts])
            lambda_v[:, ts - 1] = lambda_v[:, ts] * \
                (1 - params.dt * params.tau_mem_inv)

            output_term = z[:, ts] / dv_m * grad_spikes[:, ts]
            output_term[torch.isnan(output_term)] = 0.0

            jump_term = z[:, ts] * dv_p / dv_m
            jump_term[torch.isnan(jump_term)] = 0.0

            lambda_v[:, ts - 1] = (
                (1 - z[:, ts]) * lambda_v[:, ts - 1]
                + jump_term * lambda_v[:, ts - 1]
                + output_term
            )
        return torch.stack((lambda_i,  # / params.tau_syn_inv,
                            - lambda_v + lambda_i)), None


class EventPropSynapse(torch.autograd.Function):
    """
    Synapse function for proper gradient transport when using EventPropNeuron
    """
    @staticmethod
    # pylint: disable=arguments-differ, redefined-builtin
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor,
                _: torch.Tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This should be used in combination with EventPropNeuron. Apply linear
        to input using weight and use a stacked output in order to be able to
        return correct terms according to EventProp to previous layer and
        weights.

        :param input: Input spikes in shape (batch, time, in_neurons)
        :param weight: Weight in shape (out_neurons, in_neurons)
        :param _: Bias, which is unused here

        :returns: Returns stacked tensor holding weighted spikes and
            tensor with zeros but same shape
        """
        ctx.save_for_backward(input, weight)
        output = input.matmul(weight.t())
        return torch.stack((output, torch.zeros_like(output)))

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    # pylint: disable=arguments-differ, redefined-builtin
    def backward(ctx, grad_output: torch.Tensor,
                 ) -> Tuple[Optional[torch.Tensor],
                            Optional[torch.Tensor]]:
        """
        Split gradient_output coming from EventPropNeuron and return
        weight * (lambda_v - lambda_i) as input gradient and
        - tau_s * lambda_i * input (i.e. lambda_i at spiketimes)
        as weight gradient.

        :param grad_output: Backpropagated gradient with shape (2, batch, time,
            out_neurons). The 2 is due to stacking in forward.

        :returns: Returns gradients w.r.t. input, weight and bias (None)
        """
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output[1].matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = \
                grad_output[0].transpose(1, 2).matmul(input)

        return grad_input, grad_weight, None


def eventprop_synapse(input: torch.Tensor, weight: torch.Tensor,
                      bias=None) -> torch.Tensor:
    # Allow redefinition of builtin in order to be consistent with PyTorch
    # pylint: disable=redefined-builtin
    """
    Function wrapper to apply EventPropSynapse.
    TODO:
        This is used to work around problems with "if autograd" stuff in
        snn/modules.py -> Figure out what exactly goes wrong when using
        EventPropSynapse directly as func in hxtoch.snn.Synapse...

    :param input: Input spikes in shape (batch, time, in_neurons)
    :param weight: Weight in shape (out_neurons, in_neurons)
    :param bias: Bias (which is unused in EventPropSynapse)

    :returns: Returns stacked tensor holding weighted spikes and
        tensor with zeros but same shape
    """
    return EventPropSynapse.apply(input, weight, bias)
