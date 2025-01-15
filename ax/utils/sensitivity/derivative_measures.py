# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Any

import torch
from ax.utils.sensitivity.derivative_gp import posterior_derivative
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize
from gpytorch.distributions import MultivariateNormal
from pyre_extensions import assert_is_instance, none_throws


def sample_discrete_parameters(
    input_mc_samples: torch.Tensor,
    discrete_features: None | list[int],
    bounds: torch.Tensor,
    num_mc_samples: int,
) -> torch.Tensor:
    r"""Samples the input parameters uniformly at random for the discrete features.

    Args:
        input_mc_samples: The input mc samples tensor to be modified.
        discrete_features: A list of integers (or None) of indices corresponding
            to discrete features.
        bounds: The parameter bounds.
        num_mc_samples: The number of Monte Carlo grid samples.

    Returns:
        A modified input mc samples tensor.
    """
    if discrete_features is None:
        return input_mc_samples
    all_low = bounds[0, discrete_features].to(dtype=torch.int).tolist()
    all_high = (bounds[1, discrete_features]).to(dtype=torch.int).tolist()
    for i, low, high in zip(discrete_features, all_low, all_high):
        randint = partial(torch.randint, low=low, high=high + 1)
        input_mc_samples[:, i] = randint(size=torch.Size([num_mc_samples]))
    return input_mc_samples


class GpDGSMGpMean:
    mean_gradients: torch.Tensor | None = None
    bootstrap_indices: torch.Tensor | None = None
    mean_gradients_btsp: list[torch.Tensor] | None = None

    def __init__(
        self,
        model: Model,
        bounds: torch.Tensor,
        derivative_gp: bool = False,
        kernel_type: str | None = None,
        Y_scale: float = 1.0,
        num_mc_samples: int = 10**4,
        input_qmc: bool = False,
        dtype: torch.dtype = torch.double,
        num_bootstrap_samples: int = 1,
        discrete_features: list[int] | None = None,
    ) -> None:
        r"""Computes three types of derivative based measures:
        the gradient, the gradient square and the gradient absolute measures.

        Args:
            model: A BoTorch model.
            bounds: Parameter bounds over which to evaluate model sensitivity.
            derivative_gp: If true, the derivative of the GP is used to compute
                the gradient instead of backward.
            kernel_type: Takes "rbf" or "matern", set only if `derivative_gp` is true.
            Y_scale: Scale the derivatives by this amount, to undo scaling
                done on the training data.
            num_mc_samples: The number of MonteCarlo grid samples
            input_qmc: If True, a qmc Sobol grid is use instead of uniformly random.
            dtype: Can be provided if the GP is fit to data of type `torch.float`.
            num_bootstrap_samples: If higher than 1, the method will compute the
                dgsm measure `num_bootstrap_samples` times by selecting subsamples
                from the `input_mc_samples` and return the variance and standard error
                across all computed measures.
            discrete_features: If specified, the inputs associated with the indices in
                this list are generated using an integer-valued uniform distribution,
                rather than the default (pseudo-)random continuous uniform distribution.
        """
        # pyre-fixme[4]: Attribute must be annotated.
        self.dim = assert_is_instance(model.train_inputs, tuple)[0].shape[-1]
        self.derivative_gp = derivative_gp
        self.kernel_type = kernel_type
        # pyre-fixme[4]: Attribute must be annotated.
        self.bootstrap = num_bootstrap_samples > 1
        # pyre-fixme[4]: Attribute must be annotated.
        self.num_bootstrap_samples = (
            num_bootstrap_samples - 1
        )  # deduct 1 because the first is meant to be the full grid
        if self.derivative_gp and (self.kernel_type is None):
            raise ValueError("Kernel type has to be specified to use derivative GP")
        self.num_mc_samples = num_mc_samples
        if input_qmc:
            # pyre-fixme[4]: Attribute must be annotated.
            self.input_mc_samples = (
                draw_sobol_samples(bounds=bounds, n=num_mc_samples, q=1, seed=1234)
                .squeeze(1)
                .to(dtype)
            )
        else:
            self.input_mc_samples = unnormalize(
                torch.rand(num_mc_samples, self.dim, dtype=dtype),
                bounds=bounds,
            )

        # uniform integral distribution for discrete features
        self.input_mc_samples = sample_discrete_parameters(
            input_mc_samples=self.input_mc_samples,
            discrete_features=discrete_features,
            bounds=bounds,
            num_mc_samples=num_mc_samples,
        )

        if self.derivative_gp:
            posterior = posterior_derivative(
                model, self.input_mc_samples, none_throws(self.kernel_type)
            )
        else:
            self.input_mc_samples.requires_grad = True
            posterior = assert_is_instance(
                model.posterior(self.input_mc_samples), GPyTorchPosterior
            )
        self._compute_gradient_quantities(posterior, Y_scale)

    def _compute_gradient_quantities(
        self, posterior: GPyTorchPosterior | MultivariateNormal, Y_scale: float
    ) -> None:
        if self.derivative_gp:
            self.mean_gradients = (
                assert_is_instance(posterior.mean, torch.Tensor) * Y_scale
            )
        else:
            predictive_mean = posterior.mean
            torch.sum(predictive_mean).backward()
            self.mean_gradients = (
                assert_is_instance(self.input_mc_samples.grad, torch.Tensor) * Y_scale
            )
        if self.bootstrap:
            subset_size = 2
            self.bootstrap_indices = torch.randint(
                0, self.num_mc_samples, (self.num_bootstrap_samples, subset_size)
            )
            self.mean_gradients_btsp = [
                torch.index_select(
                    assert_is_instance(self.mean_gradients, torch.Tensor), 0, indices
                )
                for indices in self.bootstrap_indices
            ]

    def aggregation(
        self, transform_fun: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        gradients_measure = torch.tensor(
            [
                torch.mean(transform_fun(none_throws(self.mean_gradients)[:, i]))
                for i in range(self.dim)
            ]
        )
        if not (self.bootstrap):
            return gradients_measure
        else:
            gradients_measures_btsp = [gradients_measure.unsqueeze(0)]
            for b in range(self.num_bootstrap_samples):
                gradients_measures_btsp.append(
                    torch.tensor(
                        [
                            torch.mean(
                                transform_fun(
                                    none_throws(self.mean_gradients_btsp)[b][:, i]
                                )
                            )
                            for i in range(self.dim)
                        ]
                    ).unsqueeze(0)
                )
            gradients_measures_btsp = torch.cat(gradients_measures_btsp, dim=0)
            return (
                torch.cat(
                    [
                        gradients_measures_btsp.mean(dim=0).unsqueeze(0),
                        gradients_measures_btsp.var(dim=0).unsqueeze(0),
                        torch.sqrt(
                            gradients_measures_btsp.var(dim=0)
                            / (self.num_bootstrap_samples + 1)
                        ).unsqueeze(0),
                    ],
                    dim=0,
                )
                .t()
                .detach()
            )

    def gradient_measure(self) -> torch.Tensor:
        r"""Computes the gradient measure:

        Returns:
            if `self.num_bootstrap_samples > 1`
                Tensor: (values, var_mc, stderr_mc) x dim
            else
                Tensor: (values) x dim
        """
        return self.aggregation(torch.as_tensor)

    def gradient_absolute_measure(self) -> torch.Tensor:
        r"""Computes the gradient absolute measure:

        Returns:
            if `self.num_bootstrap_samples > 1`
                Tensor: (values, var_mc, stderr_mc) x dim
            else
                Tensor: (values) x dim
        """
        return self.aggregation(torch.abs)

    def gradients_square_measure(self) -> torch.Tensor:
        r"""Computes the gradient square measure:

        Returns:
            if `num_bootstrap_samples > 1`
                Tensor: (values, var_mc, stderr_mc) x dim
            else
                Tensor: (values) x dim
        """
        return self.aggregation(torch.square)


class GpDGSMGpSampling(GpDGSMGpMean):
    samples_gradients: torch.Tensor | None = None
    samples_gradients_btsp: list[torch.Tensor] | None = None

    def __init__(
        self,
        model: Model,
        bounds: torch.Tensor,
        num_gp_samples: int,
        derivative_gp: bool = False,
        kernel_type: str | None = None,
        Y_scale: float = 1.0,
        num_mc_samples: int = 10**4,
        input_qmc: bool = False,
        gp_sample_qmc: bool = False,
        dtype: torch.dtype = torch.double,
        num_bootstrap_samples: int = 1,
    ) -> None:
        r"""Computes three types of derivative based measures:
        the gradient, the gradient square and the gradient absolute measures.

        Args:
            model: A BoTorch model.
            bounds: Parameter bounds over which to evaluate model sensitivity.
            num_gp_samples: If method is "GP samples", the number of GP samples has
                to be set.
            derivative_gp: If true, the derivative of the GP is used to compute the
                gradient instead of backward.
            kernel_type: Takes "rbf" or "matern", set only if `derivative_gp` is true.
            Y_scale: Scale the derivatives by this amount, to undo scaling done on
                the training data.
            num_mc_samples: The number of Monte Carlo grid samples.
            input_qmc: If True, a qmc Sobol grid is used instead of uniformly random.
            gp_sample_qmc: If True, the posterior sampling is done using
                `SobolQMCNormalSampler`.
            dtype: Can be provided if the GP is fit to data of type `torch.float`.
            num_bootstrap_samples: If higher than 1, the method will compute the
                dgsm measure `num_bootstrap_samples` times by selecting subsamples
                from the `input_mc_samples` and return the variance and standard error
                across all computed measures.

        Returns values of gradient_measure, gradient_absolute_measure and
        gradients_square_measure change to the following:
            if `num_bootstrap_samples > 1`:
                Tensor: (values, var_gp, stderr_gp, var_mc, stderr_mc) x dim
            else
                Tensor: (values, var_gp, stderr_gp) x dim
        """
        self.num_gp_samples = num_gp_samples
        self.gp_sample_qmc = gp_sample_qmc
        self.num_mc_samples = num_mc_samples
        super().__init__(
            model=model,
            bounds=bounds,
            derivative_gp=derivative_gp,
            kernel_type=kernel_type,
            Y_scale=Y_scale,
            num_mc_samples=num_mc_samples,
            input_qmc=input_qmc,
            dtype=dtype,
            num_bootstrap_samples=num_bootstrap_samples,
        )

    def _compute_gradient_quantities(
        self, posterior: Posterior | MultivariateNormal, Y_scale: float
    ) -> None:
        if self.gp_sample_qmc:
            sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([self.num_gp_samples]), seed=0
            )
            samples = sampler(posterior)
        else:
            samples = posterior.rsample(torch.Size([self.num_gp_samples]))
        if self.derivative_gp:
            self.samples_gradients = samples * Y_scale
        else:
            samples_gradients = []
            for j in range(self.num_gp_samples):
                torch.sum(samples[j]).backward(retain_graph=True)
                samples_gradients.append(
                    deepcopy(self.input_mc_samples.grad).unsqueeze(0)
                )
                self.input_mc_samples.grad.data.zero_()
            self.samples_gradients = torch.cat(samples_gradients, dim=0) * Y_scale
        if self.bootstrap:
            subset_size = 2
            self.bootstrap_indices = torch.randint(
                0, self.num_mc_samples, (self.num_bootstrap_samples, subset_size)
            )
            self.samples_gradients_btsp = []
            for j in range(self.num_gp_samples):
                none_throws(self.samples_gradients_btsp).append(
                    torch.cat(
                        [
                            torch.index_select(
                                none_throws(self.samples_gradients)[j], 0, indices
                            ).unsqueeze(0)
                            for indices in none_throws(self.bootstrap_indices)
                        ],
                        dim=0,
                    )
                )

    def aggregation(
        self, transform_fun: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        gradients_measure_list = []
        for j in range(self.num_gp_samples):
            gradients_measure_list.append(
                torch.tensor(
                    [
                        torch.mean(
                            transform_fun(none_throws(self.samples_gradients)[j][:, i])
                        )
                        for i in range(self.dim)
                    ]
                ).unsqueeze(0)
            )
        gradients_measure_list = torch.cat(gradients_measure_list, dim=0)
        if not (self.bootstrap):
            gradients_measure_mean_var = []
            for i in range(self.dim):
                gradients_measure_mean_var.append(
                    torch.tensor(
                        [
                            torch.mean(gradients_measure_list[:, i]),
                            torch.var(gradients_measure_list[:, i]),
                            torch.sqrt(
                                torch.var(gradients_measure_list[:, i])
                                / self.num_gp_samples
                            ),
                        ]
                    ).unsqueeze(0)
                )
            gradients_measure_mean_var = torch.cat(gradients_measure_mean_var, dim=0)
            return gradients_measure_mean_var
        else:
            gradients_measure_list_btsp = []
            for j in range(self.num_gp_samples):
                gradients_measure_btsp = [gradients_measure_list[j].unsqueeze(0)] + [
                    torch.tensor(
                        [
                            torch.mean(
                                transform_fun(
                                    none_throws(self.samples_gradients_btsp)[j][b][:, i]
                                )
                            )
                            for i in range(self.dim)
                        ]
                    ).unsqueeze(0)
                    for b in range(self.num_bootstrap_samples)
                ]
                gradients_measure_list_btsp.append(
                    torch.cat(gradients_measure_btsp, dim=0).unsqueeze(0)
                )
            gradients_measure_list_btsp = torch.cat(gradients_measure_list_btsp, dim=0)

            var_per_bootstrap = torch.var(gradients_measure_list_btsp, dim=0)
            gp_var = torch.mean(var_per_bootstrap, dim=0)
            gp_se = torch.sqrt(gp_var / self.num_gp_samples)
            var_per_gp_sample = torch.var(gradients_measure_list_btsp, dim=1)
            mc_var = torch.mean(var_per_gp_sample, dim=0)
            mc_se = torch.sqrt(mc_var / (self.num_bootstrap_samples + 1))
            total_mean = gradients_measure_list_btsp.reshape(-1, self.dim).mean(dim=0)
            gradients_measure_mean_vargp_segp_varmc_segp = torch.cat(
                [
                    torch.tensor(
                        [total_mean[i], gp_var[i], gp_se[i], mc_var[i], mc_se[i]]
                    ).unsqueeze(0)
                    for i in range(self.dim)
                ],
                dim=0,
            )
            return gradients_measure_mean_vargp_segp_varmc_segp


def compute_derivatives_from_model_list(
    model_list: list[Model],
    bounds: torch.Tensor,
    discrete_features: list[int] | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Computes average derivatives of a list of models on a bounded domain. Estimation
    is according to the GP posterior mean function.

    Args:
        model_list: A list of m botorch.models.model.Model types for which to compute
            the average derivative.
        bounds: A 2 x d Tensor of lower and upper bounds of the domain of the models.
        discrete_features: If specified, the inputs associated with the indices in
            this list are generated using an integer-valued uniform distribution,
            rather than the default (pseudo-)random continuous uniform distribution.
        kwargs: Passed along to GpDGSMGpMean.

    Returns:
        A (m x d) tensor of gradient measures.
    """
    indices = []
    for model in model_list:
        sens_class = GpDGSMGpMean(
            model=model, bounds=bounds, discrete_features=discrete_features, **kwargs
        )
        indices.append(sens_class.gradient_measure())
    return torch.stack(indices)
