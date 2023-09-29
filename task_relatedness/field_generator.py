import typing
from typing import Any, Callable, List, Tuple, Union

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_output,
    _is_tuple,
)
from captum._utils.typing import (
    BaselineType,
    Literal,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.attribution import GradientAttribution
from captum.attr._utils.batching import _batch_attribution
from captum.attr._utils.common import _format_input_baseline, _validate_input
from captum.log import log_usage
from torch import Tensor


def _reshape_and_cumsum(tensor_input: Tensor, num_steps: int,
                        num_examples: int, layer_size: Tuple[int,
                                                             ...]) -> Tensor:
    return torch.cumsum(tensor_input.reshape((num_steps, num_examples) +
                                             layer_size),
                        dim=0).transpose(0, 1)


def _reshape_and_not_cumsum(tensor_input: Tensor, num_steps: int,
                            num_examples: int,
                            layer_size: Tuple[int, ...]) -> Tensor:

    return tensor_input.reshape((num_steps, num_examples) +
                                layer_size).transpose(0, 1)


class FieldGenerator(GradientAttribution):
    r"""
    Generate the field of the model
    """
    def __init__(
        self,
        forward_func: Callable,
        multiply_by_inputs=True,
        is_cumsum: bool = True,
    ) -> None:
        r"""
        Args:

            forward_func (callable):  The forward function of the model or any
                    modification of it
            multiply_by_inputs (bool, optional): Indicates whether to factor
                    model inputs' multiplier in the final attribution scores.
                    In the literature this is also known as local vs global
                    attribution. 
            is_cumsum: Whether to keep the cumulative attribution map
            
        """
        GradientAttribution.__init__(self, forward_func)
        self.is_cumsum = is_cumsum
        self._multiply_by_inputs = multiply_by_inputs

    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: Literal[False] = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        ...

    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        *,
        return_convergence_delta: Literal[True],
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, Tensor]:
        ...

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
    ) -> Union[TensorOrTupleOfTensorsGeneric, Tuple[
            TensorOrTupleOfTensorsGeneric, Tensor]]:
        is_inputs_tuple = _is_tuple(inputs)
        inputs, baselines = _format_input_baseline(inputs, baselines)
        _validate_input(inputs, baselines, n_steps, method)
        if internal_batch_size is not None:
            num_examples = inputs[0].shape[0]
            attributions = _batch_attribution(
                self,
                num_examples,
                internal_batch_size,
                n_steps,
                inputs=inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                method=method,
            )
        else:
            attributions = self._attribute(
                inputs=inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
                method=method,
            )
        if return_convergence_delta:
            start_point, end_point = baselines, inputs
            # computes approximation error based on the completeness axiom
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                additional_forward_args=additional_forward_args,
                target=target,
            )
            return _format_output(is_inputs_tuple, attributions), delta
        formated_output = _format_output(is_inputs_tuple, attributions)
        return formated_output

    def _attribute(self,
                   inputs: Tuple[Tensor, ...],
                   baselines: Tuple[Union[Tensor, int, float], ...],
                   target: TargetType = None,
                   additional_forward_args: Any = None,
                   n_steps: int = 50,
                   method: str = "gausslegendre",
                   step_sizes_and_alphas: Union[None,
                                                Tuple[List[float],
                                                      List[float]]] = None,
                   output_grad=False) -> Tuple[Tensor, ...]:
        if step_sizes_and_alphas is None:
            # retrieve step size and scaling factor for specified
            # approximation method
            step_sizes_func, alphas_func = approximation_parameters(method)
            step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
        else:
            step_sizes, alphas = step_sizes_and_alphas
        # print(alphas)
        # scale features and compute gradients. (batch size is abbreviated as bsz)
        # scaled_features' dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas],
                dim=0).requires_grad_()
            for input, baseline in zip(inputs, baselines))
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args)
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (_expand_additional_forward_args(
            additional_forward_args, n_steps) if additional_forward_args is not None else None)
        expanded_target = _expand_target(target, n_steps)
        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        grads = self.gradient_func(
            forward_fn=self.forward_func,
            inputs=scaled_features_tpl,
            target_ind=expanded_target,
            additional_forward_args=input_additional_args,
        )
        scaled_grads = [
            grad.contiguous().view(n_steps, -1) *
            torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
            for grad in grads
        ]
        # flattening grads so that we can multilpy it with step-size
        # calling contiguous to avoid `memory whole` problems
        if not self.is_cumsum:
            total_grads = tuple(
                _reshape_and_not_cumsum(scaled_grad, n_steps, grad.shape[0] //
                                        n_steps, grad.shape[1:])
                for (scaled_grad, grad) in zip(scaled_grads, grads))
        else:
            # aggregates across all steps for each tensor in the input tuple
            # total_grads has the same dimensionality as inputs
            # print(scaled_grads[0].shape, grads[0].shape) # torch.Size([50, 49152]) torch.Size([50, 3, 128, 128])
            total_grads = tuple(
                _reshape_and_cumsum(scaled_grad, n_steps, grad.shape[0] //
                                    n_steps, grad.shape[1:])
                for (scaled_grad, grad) in zip(scaled_grads, grads))
        # print(len(total_grads), total_grads[0].shape)
        # computes attribution for each tensor in input tuple
        # attributions has the same dimensionality as inputs
        if not self.multiplies_by_inputs:
            attributions = total_grads
        else:
            attributions = tuple(total_grad * (input - baseline).unsqueeze(1)
                                 for total_grad, input, baseline in zip(
                                     total_grads, inputs, baselines))
        return attributions

    def has_convergence_delta(self) -> bool:
        return True

    @property
    def multiplies_by_inputs(self):
        return self._multiply_by_inputs
