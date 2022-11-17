# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Callable, List, Optional, Union

import torch
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.sensitivity.derivative_gp import posterior_derivative
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize
from gpytorch.distributions import MultivariateNormal


class GpDGSMGpMean(object):

    mean_gradients: Optional[torch.Tensor] = None
    bootstrap_indices: Optional[torch.Tensor] = None
    mean_gradients_btsp: Optional[List[torch.Tensor]] = None

    def __init__(
        self,
        model: Model,
        bounds: torch.Tensor,
        derivative_gp: bool = False,
        kernel_type: Optional[str] = None,
        Y_scale: float = 1.0,
        num_mc_samples: int = 10**4,
        input_qmc: bool = False,
        dtype: torch.dtype = torch.double,
        num_bootstrap_samples: int = 1,
    ) -> None:
        r"""Computes three types of derivative based measures:
        the gradient, the gradient square and the gradient absolute measures.

        Args:
            model: A BoTorch model.
            bounds: Parameter bounds over which to evaluate model sensitivity.
            derivative_gp: If true, the derivative of the GP is used to compute
                the gradient instead of backward. If `kernel_type` is matern_l1,
                only the mean function of derivative GP can be used, and the
                variance is not defined.
            kernel_type: Takes "rbf" or "matern_l1" or "matern_l2", set only
                if `derivative_gp` is true.
            Y_scale: Scale the derivatives by this amount, to undo scaling
                done on the training data.
            num_mc_samples: The number of MonteCarlo grid samples
            input_qmc: If True, a qmc Sobol grid is use instead of uniformly random.
            dtype: Can be provided if the GP is fit to data of type `torch.float`.
            num_bootstrap_samples: If higher than 1, the method will compute the
                dgsm measure `num_bootstrap_samples` times by selecting subsamples
                from the `input_mc_samples` and return the variance and standard error
                across all computed measures.
        """
        # pyre-fixme[4]: Attribute must be annotated.
        self.dim = checked_cast(tuple, model.train_inputs)[0].shape[-1]
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
                draw_sobol_samples(bounds=bounds, n=num_mc_samples, q=1)
                .squeeze(1)
                .to(dtype)
            )
        else:
            self.input_mc_samples = unnormalize(
                torch.rand(num_mc_samples, self.dim, dtype=dtype),
                bounds=bounds,
            )
        if self.derivative_gp:
            posterior = posterior_derivative(
                model, self.input_mc_samples, not_none(self.kernel_type)
            )
        else:
            self.input_mc_samples.requires_grad = True
            posterior = checked_cast(
                GPyTorchPosterior, model.posterior(self.input_mc_samples)
            )
        self._compute_gradient_quantities(posterior, Y_scale)

    def _compute_gradient_quantities(
        self, posterior: Union[GPyTorchPosterior, MultivariateNormal], Y_scale: float
    ) -> None:
        if self.derivative_gp:
            self.mean_gradients = checked_cast(torch.Tensor, posterior.mean) * Y_scale
        else:
            predictive_mean = posterior.mean
            torch.sum(predictive_mean).backward()
            self.mean_gradients = (
                checked_cast(torch.Tensor, self.input_mc_samples.grad) * Y_scale
            )
        if self.bootstrap:
            subset_size = 2
            self.bootstrap_indices = torch.randint(
                0, self.num_mc_samples, (self.num_bootstrap_samples, subset_size)
            )
            self.mean_gradients_btsp = [
                torch.index_select(
                    checked_cast(torch.Tensor, self.mean_gradients), 0, indices
                )
                for indices in self.bootstrap_indices
            ]

    def aggregation(
        self, transform_fun: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        gradients_measure = torch.tensor(
            [
                torch.mean(transform_fun(not_none(self.mean_gradients)[:, i]))
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
                                    not_none(self.mean_gradients_btsp)[b][:, i]
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
        return self.aggregation(torch.tensor)

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

    samples_gradients: Optional[torch.Tensor] = None
    samples_gradients_btsp: Optional[List[torch.Tensor]] = None

    def __init__(
        self,
        model: Model,
        bounds: torch.Tensor,
        num_gp_samples: int,
        derivative_gp: bool = False,
        kernel_type: Optional[str] = None,
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
                gradient instead of backward. If `kernel_type` is matern_l1,
                `derivative_gp` should be False because the variance is not defined.
            kernel_type: Takes "rbf" or "matern_l1" or "matern_l2", set only if
                `derivative_gp` is true.
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
        self, posterior: Union[Posterior, MultivariateNormal], Y_scale: float
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
                not_none(self.samples_gradients_btsp).append(
                    torch.cat(
                        [
                            torch.index_select(
                                not_none(self.samples_gradients)[j], 0, indices
                            ).unsqueeze(0)
                            for indices in not_none(self.bootstrap_indices)
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
                            transform_fun(not_none(self.samples_gradients)[j][:, i])
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
                                    not_none(self.samples_gradients_btsp)[j][b][:, i]
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
