# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy.typing as npt
import torch
from ax.modelbridge.torch import TorchModelBridge
from ax.models.torch.botorch import BotorchModel
from ax.models.torch.botorch_modular.model import BoTorchModel as ModularBoTorchModel
from ax.utils.sensitivity.derivative_measures import (
    compute_derivatives_from_model_list,
    sample_discrete_parameters,
)
from botorch.models.model import Model, ModelList
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import is_ensemble, unnormalize
from pyre_extensions import assert_is_instance
from torch import Tensor


class SobolSensitivity:
    def __init__(
        self,
        bounds: torch.Tensor,
        input_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        num_mc_samples: int = 10**4,
        input_qmc: bool = False,
        second_order: bool = False,
        first_order_idcs: torch.Tensor | None = None,
        num_bootstrap_samples: int = 1,
        bootstrap_array: bool = False,
        discrete_features: list[int] | None = None,
    ) -> None:
        r"""Computes three types of Sobol indices:
        first order indices, total indices and second order indices (if specified ).

        Args:
            bounds: Parameter bounds over which to evaluate model sensitivity.
            input_function: The objective function.
            num_mc_samples: The number of montecarlo grid samples
            input_qmc: If True, a qmc Sobol grid is use instead of uniformly random.
            second_order: If True, the second order indices are computed.
            bootstrap: If true, the MC error is returned.
            first_order_idcs: Tensor of previously computed first order indices, where
                first_order_idcs.shape = torch.Size([dim]).
            num_bootstrap_samples: If bootstrap is true, the number of bootstraps has
                to be specified.
            bootstrap_array: If true, all the num_bootstrap_samples extimated indices
                are returned instead of their mean and Var.
            discrete_features: If specified, the inputs associated with the indices in
                this list are generated using an integer-valued uniform distribution,
                rather than the default (pseudo-)random continuous uniform distribution.
        """
        self.input_function = input_function
        self.dim: int = bounds.shape[-1]
        self.num_mc_samples = num_mc_samples
        self.second_order = second_order
        self.bootstrap: bool = num_bootstrap_samples > 1
        self.num_bootstrap_samples: int = (
            num_bootstrap_samples - 1
        )  # deduct 1 because the first is meant to be the full grid
        self.bootstrap_array = bootstrap_array
        if input_qmc:
            sobol_kwargs = {"bounds": bounds, "n": num_mc_samples, "q": 1}
            seed_A, seed_B = 1234, 5678  # to make it reproducible
            # pyre-ignore
            self.A = draw_sobol_samples(**sobol_kwargs, seed=seed_A).squeeze(1)
            # pyre-ignore
            self.B = draw_sobol_samples(**sobol_kwargs, seed=seed_B).squeeze(1)
        else:
            self.A = unnormalize(torch.rand(num_mc_samples, self.dim), bounds=bounds)
            self.B = unnormalize(torch.rand(num_mc_samples, self.dim), bounds=bounds)

        # uniform integral distribution for discrete features
        self.A = sample_discrete_parameters(
            input_mc_samples=self.A,
            discrete_features=discrete_features,
            bounds=bounds,
            num_mc_samples=num_mc_samples,
        )
        self.B = sample_discrete_parameters(
            input_mc_samples=self.B,
            discrete_features=discrete_features,
            bounds=bounds,
            num_mc_samples=num_mc_samples,
        )

        # pyre-fixme[4]: Attribute must be annotated.
        self.A_B_ABi = self.generate_all_input_matrix().to(torch.double)

        if self.bootstrap:
            subset_size = 4
            # pyre-fixme[4]: Attribute must be annotated.
            self.bootstrap_indices = torch.randint(
                0, num_mc_samples, (self.num_bootstrap_samples, subset_size)
            )
        self.f_A: torch.Tensor | None = None
        self.f_B: torch.Tensor | None = None
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        self.f_ABis: list | None = None
        self.f_total_var: torch.Tensor | None = None
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        self.f_A_btsp: list | None = None
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        self.f_B_btsp: list | None = None
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        self.f_ABis_btsp: list | None = None
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        self.f_total_var_btsp: list | None = None
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        self.f_BAis: list | None = None
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        self.f_BAis_btsp: list | None = None
        self.first_order_idcs: torch.Tensor | None = first_order_idcs
        self.first_order_idcs_btsp: torch.Tensor | None = None

    def generate_all_input_matrix(self) -> torch.Tensor:
        # NOTE Sobol samples of A are ablated with samples of B, and vice versa
        # so baselies are A and B, but each one is ablated by the other for each
        # dimension. First all of A, then (optionally) all of B.
        A_B_ABi_list = [self.A, self.B]
        for i in range(self.dim):
            AB_i = deepcopy(self.A)
            AB_i[:, i] = self.B[:, i]
            A_B_ABi_list.append(AB_i)
        if self.second_order:
            for i in range(self.dim):
                BA_i = deepcopy(self.B)
                BA_i[:, i] = self.A[:, i]
                A_B_ABi_list.append(BA_i)
        A_B_ABi = torch.cat(A_B_ABi_list, dim=0)
        return A_B_ABi

    def evalute_function(self, f_A_B_ABi: torch.Tensor | None = None) -> None:
        r"""evaluates the objective function and devides the evaluation into
            torch.Tensors needed for the indices computation.
        Args:
            f_A_B_ABi: Function evaluations on the entire grid of size M(d+2).
        """
        if f_A_B_ABi is None:
            f_A_B_ABi = self.input_function(self.A_B_ABi)  # pyre-ignore
        # for multiple output models, average the outcomes
        if len(f_A_B_ABi.shape) == 3:
            f_A_B_ABi = f_A_B_ABi.mean(dim=0)

        self.f_A = f_A_B_ABi[: self.num_mc_samples]
        self.f_B = f_A_B_ABi[self.num_mc_samples : self.num_mc_samples * 2]

        # first, there is simply A and B, and then there are all the ablated variants
        # of each (which are retrieved here by slicing the input matrix)
        # but this only retrieves the ablations of A (and not B)
        self.f_ABis = [
            f_A_B_ABi[self.num_mc_samples * (i + 2) : self.num_mc_samples * (i + 3)]
            for i in range(self.dim)
        ]
        # Get the variances of A and B (so simply the variance of 2 num_mc samples)
        self.f_total_var = torch.var(f_A_B_ABi[: self.num_mc_samples * 2])
        if self.bootstrap:
            self.f_A_btsp = [
                torch.index_select(self.f_A, 0, indices)  # pyre-ignore
                for indices in self.bootstrap_indices
            ]
            self.f_B_btsp = [
                torch.index_select(self.f_B, 0, indices)  # pyre-ignore
                for indices in self.bootstrap_indices
            ]
            self.f_ABis_btsp = [
                [
                    torch.index_select(f_ABi, 0, indices)
                    for f_ABi in self.f_ABis  # pyre-ignore
                ]
                for indices in self.bootstrap_indices
            ]
            self.f_total_var_btsp = [
                torch.var(
                    torch.cat(
                        # pyre-fixme[16]: Optional type has no attribute `__getitem__`.
                        (self.f_A_btsp[i], self.f_B_btsp[i]),
                        dim=0,
                    )
                )
                for i in range(self.num_bootstrap_samples)
            ]
        if self.second_order:
            # If second order, we also need to retrieve the ablations of B
            # In total, we have f_ABis and f_BAis which are both of size M(d), and are
            # the ablations of each other
            self.f_BAis = [
                f_A_B_ABi[
                    self.num_mc_samples * (i + 2 + self.dim) : self.num_mc_samples
                    * (i + 3 + self.dim)
                ]
                for i in range(self.dim)
            ]
            if self.bootstrap:
                self.f_BAis_btsp = [
                    [torch.index_select(f_BAi, 0, indices) for f_BAi in self.f_BAis]
                    for indices in self.bootstrap_indices
                ]

    def first_order_indices(self) -> Tensor:
        r"""Computes the first order Sobol indices:

        Returns:
            if num_bootstrap_samples>1
                Tensor: (values,var_mc,stderr_mc)x dim
            else
                Tensor: (values)x dim
        """
        first_order_idcs = []
        for i in range(self.dim):
            # The only difference between self.f_ABis[i] and self.f_A is in dimension i
            # So we get the variance in the component that corresponds to dimension i
            vi = (
                torch.mean(self.f_B * (self.f_ABis[i] - self.f_A))  # pyre-ignore
                # pyre-fixme[58]: `/` is not supported for operand types `Tensor`
                #  and `Optional[Tensor]`.
                / self.f_total_var
            )
            first_order_idcs.append(vi.unsqueeze(0))
        self.first_order_idcs = torch.cat(first_order_idcs, dim=0).detach()
        if not self.bootstrap:
            return self.first_order_idcs
        else:
            first_order_idcs_btsp = [torch.cat(first_order_idcs, dim=0).unsqueeze(0)]
            for b in range(self.num_bootstrap_samples):
                first_order_idcs = []
                for i in range(self.dim):
                    vi = (
                        torch.mean(
                            self.f_B_btsp[b]
                            * (self.f_ABis_btsp[b][i] - self.f_A_btsp[b])
                        )
                        / self.f_total_var_btsp[b]
                    )
                    first_order_idcs.append(vi.unsqueeze(0))
                first_order_idcs_btsp.append(
                    torch.cat(first_order_idcs, dim=0).unsqueeze(0)
                )
            self.first_order_idcs_btsp = torch.cat(first_order_idcs_btsp, dim=0)
            if self.bootstrap_array:
                return self.first_order_idcs_btsp.detach()
            else:
                return (
                    torch.cat(
                        [
                            self.first_order_idcs_btsp.mean(dim=0).unsqueeze(0),
                            self.first_order_idcs_btsp.var(  # pyre-ignore
                                dim=0
                            ).unsqueeze(0),
                            torch.sqrt(
                                self.first_order_idcs_btsp.var(dim=0)
                                / (self.num_bootstrap_samples + 1)
                            ).unsqueeze(0),
                        ],
                        dim=0,
                    )
                    .t()
                    .detach()
                )

    def total_order_indices(self) -> Tensor:
        r"""Computes the total Sobol indices:

        Returns:
            if num_bootstrap_samples>1
                Tensor: (values,var_mc,stderr_mc)x dim
            else
                Tensor: (values)x dim
        """
        total_order_idcs = []
        for i in range(self.dim):
            vti = (
                0.5
                # pyre-fixme[58]: `-` is not supported for operand types
                #  `Optional[torch._tensor.Tensor]` and `Any`.
                # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
                * torch.mean(torch.pow(self.f_A - self.f_ABis[i], 2))
                # pyre-fixme[58]: `/` is not supported for operand types `Tensor`
                #  and `Optional[Tensor]`.
                / self.f_total_var
            )
            total_order_idcs.append(vti.unsqueeze(0))
        if not (self.bootstrap):
            total_order_idcs = torch.cat(total_order_idcs, dim=0).detach()
            return total_order_idcs
        else:
            total_order_idcs_btsp = [torch.cat(total_order_idcs, dim=0).unsqueeze(0)]
            for b in range(self.num_bootstrap_samples):
                total_order_idcs = []
                for i in range(self.dim):
                    vti = (
                        0.5
                        * torch.mean(
                            torch.pow(self.f_A_btsp[b] - self.f_ABis_btsp[b][i], 2)
                        )
                        / self.f_total_var_btsp[b]
                    )
                    total_order_idcs.append(vti.unsqueeze(0))
                total_order_idcs_btsp.append(
                    torch.cat(total_order_idcs, dim=0).unsqueeze(0)
                )
            total_order_idcs_btsp = torch.cat(total_order_idcs_btsp, dim=0)
            if self.bootstrap_array:
                return total_order_idcs_btsp.detach()
            else:
                return (
                    torch.cat(
                        [
                            total_order_idcs_btsp.mean(dim=0).unsqueeze(0),
                            total_order_idcs_btsp.var(dim=0).unsqueeze(0),
                            torch.sqrt(
                                total_order_idcs_btsp.var(dim=0)
                                / (self.num_bootstrap_samples + 1)
                            ).unsqueeze(0),
                        ],
                        dim=0,
                    )
                    .t()
                    .detach()
                )

    def second_order_indices(
        self,
        first_order_idcs: torch.Tensor | None = None,
        first_order_idcs_btsp: torch.Tensor | None = None,
    ) -> Tensor:
        r"""Computes the Second order Sobol indices:
        Args:
            first_order_idcs: Tensor of previously computed first order indices, where
                first_order_idcs.shape = torch.Size([dim]).
            first_order_idcs_btsp: Tensor of all first order indices given by bootstrap.
        Returns:
            if num_bootstrap_samples>1
                Tensor: (values,var_mc,stderr_mc)x dim
            else
                Tensor: (values)x dim
        """
        if not self.second_order:
            raise ValueError(
                "Second order indices has to be specified in the sensitivity definition"
            )

        # TODO Improve this part of the code by vectorization T204291129
        if first_order_idcs is None:
            if self.first_order_idcs is None:
                self.first_order_indices()
            first_order_idcs = self.first_order_idcs
        second_order_idcs = []
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                vij = torch.mean(
                    self.f_BAis[i] * self.f_ABis[j] - self.f_A * self.f_B  # pyre-ignore
                )
                vij = (
                    # pyre-fixme[58]: `/` is not supported for operand types
                    #  `Tensor` and `Optional[Tensor]`.
                    (vij / self.f_total_var)
                    - first_order_idcs[i]  # pyre-ignore
                    - first_order_idcs[j]
                )
                second_order_idcs.append(vij.unsqueeze(0))
        if not self.bootstrap:
            second_order_idcs = torch.cat(second_order_idcs, dim=0).detach()
            return second_order_idcs
        else:
            second_order_idcs_btsp = [torch.cat(second_order_idcs, dim=0).unsqueeze(0)]
            if first_order_idcs_btsp is None:
                first_order_idcs_btsp = self.first_order_idcs_btsp
            for b in range(self.num_bootstrap_samples):
                second_order_idcs = []
                for i in range(self.dim):
                    for j in range(i + 1, self.dim):
                        vij = torch.mean(
                            self.f_BAis_btsp[b][i] * self.f_ABis_btsp[b][j]
                            - self.f_A_btsp[b] * self.f_B_btsp[b]
                        )
                        vij = (
                            (vij / self.f_total_var_btsp[b])
                            - first_order_idcs_btsp[b][i]
                            - first_order_idcs_btsp[b][j]
                        )
                        second_order_idcs.append(vij.unsqueeze(0))
                second_order_idcs_btsp.append(
                    torch.cat(second_order_idcs, dim=0).unsqueeze(0)
                )
            second_order_idcs_btsp = torch.cat(second_order_idcs_btsp, dim=0)
            if self.bootstrap_array:
                return second_order_idcs_btsp.detach()
            else:
                return (
                    torch.cat(
                        [
                            second_order_idcs_btsp.mean(dim=0).unsqueeze(0),
                            second_order_idcs_btsp.var(dim=0).unsqueeze(0),
                            torch.sqrt(
                                second_order_idcs_btsp.var(dim=0)
                                / (self.num_bootstrap_samples + 1)
                            ).unsqueeze(0),
                        ],
                        dim=0,
                    )
                    .t()
                    .detach()
                )


def GaussianLinkMean(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    return mean


def ProbitLinkMean(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    a = mean / torch.sqrt(1 + var)
    return torch.distributions.Normal(0, 1).cdf(a)


class SobolSensitivityGPMean:
    def __init__(
        self,
        model: Model,  # TODO: narrow type down. E.g. ModelListGP does not work.
        bounds: torch.Tensor,
        num_mc_samples: int = 10**4,
        second_order: bool = False,
        input_qmc: bool = False,
        num_bootstrap_samples: int = 1,
        first_order_idcs: torch.Tensor | None = None,
        link_function: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = GaussianLinkMean,
        mini_batch_size: int = 128,
        discrete_features: list[int] | None = None,
    ) -> None:
        r"""Computes three types of Sobol indices:
        first order indices, total indices and second order indices (if specified ).

        Args:
            model: Botorch model
            bounds: `2 x d` parameter bounds over which to evaluate model sensitivity.
            method: if "predictive mean", the predictive mean is used for indices
                computation. If "GP samples", posterior sampling is used instead.
            num_mc_samples: The number of montecarlo grid samples
            second_order: If True, the second order indices are computed.
            input_qmc: If True, a qmc Sobol grid is use instead of uniformly random.
            num_bootstrap_samples: If bootstrap is true, the number of bootstraps has
                to be specified.
            first_order_idcs: Tensor of previously computed first order indices, where
                first_order_idcs.shape = torch.Size([dim]).
            link_function: The link function to be used when computing the indices.
                Indices can be computed for the mean or on samples of the posterior,
                predictive, but defaults to computing on the mean (GaussianLinkMean).
            mini_batch_size: The size of the mini-batches used while evaluating the
                model posterior. Increasing this will increase the memory usage.
            discrete_features: If specified, the inputs associated with the indices in
                this list are generated using an integer-valued uniform distribution,
                rather than the default (pseudo-)random continuous uniform distribution.
        """
        self.model = model
        self.second_order = second_order
        self.input_qmc = input_qmc
        # pyre-fixme[4]: Attribute must be annotated.
        self.bootstrap = num_bootstrap_samples > 1
        self.num_bootstrap_samples = num_bootstrap_samples
        self.num_mc_samples = num_mc_samples

        def input_function(x: Tensor) -> Tensor:
            with torch.no_grad():
                means, variances = [], []
                # Since we're only looking at mean & variance, we can freely
                # use mini-batches.
                for x_split in x.split(split_size=mini_batch_size):
                    p = assert_is_instance(
                        self.model.posterior(x_split),
                        GPyTorchPosterior,
                    )
                    means.append(p.mean)
                    variances.append(p.variance)

            cat_dim = 1 if is_ensemble(self.model) else 0
            return link_function(
                torch.cat(means, dim=cat_dim), torch.cat(variances, dim=cat_dim)
            )

        self.sensitivity = SobolSensitivity(
            bounds=bounds,
            num_mc_samples=self.num_mc_samples,
            input_function=input_function,
            second_order=self.second_order,
            input_qmc=self.input_qmc,
            num_bootstrap_samples=self.num_bootstrap_samples,
            discrete_features=discrete_features,
        )
        self.sensitivity.evalute_function()

    def first_order_indices(self) -> Tensor:
        r"""Computes the first order Sobol indices:

        Returns:
            if num_bootstrap_samples>1
                Tensor: (values,var_mc,stderr_mc)x dim
            else
                Tensor: (values)x dim
        """
        return self.sensitivity.first_order_indices()

    def total_order_indices(self) -> Tensor:
        r"""Computes the total Sobol indices:

        Returns:
            if num_bootstrap_samples>1
                Tensor: (values,var_mc,stderr_mc)x dim
            else
                Tensor: (values)x dim
        """
        return self.sensitivity.total_order_indices()

    def second_order_indices(self) -> Tensor:
        r"""Computes the Second order Sobol indices:

        Returns:
            if num_bootstrap_samples>1
                Tensor: (values,var_mc,stderr_mc)x dim(dim-1)/2
            else
                Tensor: (values)x dim(dim-1)/2
        """
        return self.sensitivity.second_order_indices()


class SobolSensitivityGPSampling:
    def __init__(
        self,
        model: Model,
        bounds: torch.Tensor,
        num_gp_samples: int = 10**3,
        num_mc_samples: int = 10**4,
        second_order: bool = False,
        input_qmc: bool = False,
        gp_sample_qmc: bool = False,
        num_bootstrap_samples: int = 1,
        discrete_features: list[int] | None = None,
    ) -> None:
        r"""Computes three types of Sobol indices:
        first order indices, total indices and second order indices (if specified ).

        Args:
            model: Botorch model.
            bounds: `2 x d` parameter bounds over which to evaluate model sensitivity.
            num_gp_samples: If method is "GP samples", the number of GP samples
                has to be set.
            num_mc_samples: The number of montecarlo grid samples
            second_order: If True, the second order indices are computed.
            input_qmc: If True, a qmc Sobol grid is use instead of uniformly random.
            gp_sample_qmc: If True, the posterior sampling is done using
                SobolQMCNormalSampler.
            num_bootstrap_samples: If bootstrap is true, the number of bootstraps has
                to be specified.
            discrete_features: If specified, the inputs associated with the indices in
                this list are generated using an integer-valued uniform distribution,
                rather than the default (pseudo-)random continuous uniform distribution.
        """
        self.model = model
        self.second_order = second_order
        self.input_qmc = input_qmc
        self.gp_sample_qmc = gp_sample_qmc
        # pyre-fixme[4]: Attribute must be annotated.
        self.bootstrap = num_bootstrap_samples > 1
        self.num_bootstrap_samples = num_bootstrap_samples
        self.num_mc_samples = num_mc_samples
        self.num_gp_samples = num_gp_samples
        self.sensitivity = SobolSensitivity(
            bounds=bounds,
            num_mc_samples=self.num_mc_samples,
            second_order=self.second_order,
            input_qmc=self.input_qmc,
            num_bootstrap_samples=self.num_bootstrap_samples,
            bootstrap_array=True,
            discrete_features=discrete_features,
        )
        # TODO: Ideally, we would reduce the memory consumption here as well
        # but this is a tricky since it uses joint posterior sampling.
        posterior = self.model.posterior(self.sensitivity.A_B_ABi)
        if self.gp_sample_qmc:
            sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([self.num_gp_samples]), seed=0
            )
            # pyre-fixme[4]: Attribute must be annotated.
            self.samples = sampler(posterior)
        else:
            with torch.no_grad():
                self.samples = posterior.rsample(torch.Size([self.num_gp_samples]))

    @property
    def dim(self) -> int:
        """Returns the input dimensionality of `self.model`."""
        return self.sensitivity.dim

    def first_order_indices(self) -> Tensor:
        r"""Computes the first order Sobol indices:

        Returns:
            if num_bootstrap_samples>1
                Tensor: (values, var_gp, stderr_gp, var_mc, stderr_mc) x dim
            else
                Tensor: (values, var, stderr) x dim
        """
        first_order_idcs_list = []
        for j in range(self.num_gp_samples):
            self.sensitivity.evalute_function(self.samples[j])
            first_order_idcs = self.sensitivity.first_order_indices()
            first_order_idcs_list.append(first_order_idcs.unsqueeze(0))
        # pyre-fixme[16]: `SobolSensitivityGPSampling` has no attribute
        #  `first_order_idcs_list`.
        self.first_order_idcs_list = torch.cat(first_order_idcs_list, dim=0)
        if not (self.bootstrap):
            first_order_idcs_mean_var_se = []
            for i in range(self.dim):
                first_order_idcs_mean_var_se.append(
                    torch.tensor(
                        [
                            torch.mean(self.first_order_idcs_list[:, i]),
                            torch.var(self.first_order_idcs_list[:, i]),
                            torch.sqrt(
                                torch.var(self.first_order_idcs_list[:, i])
                                / self.num_gp_samples
                            ),
                        ]
                    ).unsqueeze(0)
                )
            first_order_idcs_mean_var_se = torch.cat(
                first_order_idcs_mean_var_se, dim=0
            )
            return first_order_idcs_mean_var_se
        else:
            var_per_bootstrap = torch.var(self.first_order_idcs_list, dim=0)
            gp_var = torch.mean(var_per_bootstrap, dim=0)
            gp_se = torch.sqrt(gp_var / self.num_gp_samples)
            var_per_gp_sample = torch.var(self.first_order_idcs_list, dim=1)
            mc_var = torch.mean(var_per_gp_sample, dim=0)
            mc_se = torch.sqrt(mc_var / (self.num_bootstrap_samples + 1))
            total_mean = self.first_order_idcs_list.reshape(-1, self.dim).mean(dim=0)
            first_order_idcs_mean_vargp_segp_varmc_segp = torch.cat(
                [
                    torch.tensor(
                        [total_mean[i], gp_var[i], gp_se[i], mc_var[i], mc_se[i]]
                    ).unsqueeze(0)
                    for i in range(self.dim)
                ],
                dim=0,
            )
            return first_order_idcs_mean_vargp_segp_varmc_segp

    def total_order_indices(self) -> Tensor:
        r"""Computes the total Sobol indices:

        Returns:
            if num_bootstrap_samples>1
                Tensor: (values, var_gp, stderr_gp, var_mc, stderr_mc) x dim
            else
                Tensor: (values, var, stderr) x dim
        """
        total_order_idcs_list = []
        for j in range(self.num_gp_samples):
            self.sensitivity.evalute_function(self.samples[j])
            total_order_idcs = self.sensitivity.total_order_indices()
            total_order_idcs_list.append(total_order_idcs.unsqueeze(0))
        total_order_idcs_list = torch.cat(total_order_idcs_list, dim=0)
        if not (self.bootstrap):
            total_order_idcs_mean_var = []
            for i in range(self.dim):
                total_order_idcs_mean_var.append(
                    torch.tensor(
                        [
                            torch.mean(total_order_idcs_list[:, i]),
                            torch.var(total_order_idcs_list[:, i]),
                            torch.sqrt(
                                torch.var(total_order_idcs_list[:, i])
                                / self.num_gp_samples
                            ),
                        ]
                    ).unsqueeze(0)
                )
            total_order_idcs_mean_var = torch.cat(total_order_idcs_mean_var, dim=0)
            return total_order_idcs_mean_var
        else:
            var_per_bootstrap = torch.var(total_order_idcs_list, dim=0)
            gp_var = torch.mean(var_per_bootstrap, dim=0)
            gp_se = torch.sqrt(gp_var / self.num_gp_samples)
            var_per_gp_sample = torch.var(total_order_idcs_list, dim=1)
            mc_var = torch.mean(var_per_gp_sample, dim=0)
            mc_se = torch.sqrt(mc_var / (self.num_bootstrap_samples + 1))
            total_mean = total_order_idcs_list.reshape(-1, self.dim).mean(dim=0)
            total_order_idcs_mean_vargp_segp_varmc_segp = torch.cat(
                [
                    torch.tensor(
                        [total_mean[i], gp_var[i], gp_se[i], mc_var[i], mc_se[i]]
                    ).unsqueeze(0)
                    for i in range(self.dim)
                ],
                dim=0,
            )
            return total_order_idcs_mean_vargp_segp_varmc_segp

    def second_order_indices(self) -> Tensor:
        r"""Computes the Second order Sobol indices:

        Returns:
            if num_bootstrap_samples>1
                Tensor: (values, var_gp, stderr_gp, var_mc, stderr_mc) x dim(dim-1) / 2
            else
                Tensor: (values, var, stderr) x dim(dim-1) / 2
        """
        if not (self.bootstrap):
            second_order_idcs_list = []
            for j in range(self.num_gp_samples):
                self.sensitivity.evalute_function(self.samples[j])
                second_order_idcs = self.sensitivity.second_order_indices(
                    # pyre-fixme[16]: `SobolSensitivityGPSampling` has no attribute
                    #  `first_order_idcs_list`.
                    self.first_order_idcs_list[j]
                )
                second_order_idcs_list.append(second_order_idcs.unsqueeze(0))
            second_order_idcs_list = torch.cat(second_order_idcs_list, dim=0)
            second_order_idcs_mean_var = []
            # pyre-fixme[61]: `second_order_idcs` is undefined, or not always defined.
            for i in range(len(second_order_idcs)):
                second_order_idcs_mean_var.append(
                    torch.tensor(
                        [
                            torch.mean(second_order_idcs_list[:, i]),
                            torch.var(second_order_idcs_list[:, i]),
                            torch.sqrt(
                                torch.var(second_order_idcs_list[:, i])
                                / self.num_gp_samples
                            ),
                        ]
                    ).unsqueeze(0)
                )
            second_order_idcs_mean_var = torch.cat(second_order_idcs_mean_var, dim=0)
            return second_order_idcs_mean_var
        else:
            second_order_idcs_list = []
            for j in range(self.num_gp_samples):
                self.sensitivity.evalute_function(self.samples[j])
                second_order_idcs = self.sensitivity.second_order_indices(
                    self.first_order_idcs_list[j][0],
                    self.first_order_idcs_list[j][1:],
                )
                second_order_idcs_list.append(second_order_idcs.unsqueeze(0))
            second_order_idcs_list = torch.cat(second_order_idcs_list, dim=0)
            var_per_bootstrap = torch.var(second_order_idcs_list, dim=0)
            gp_var = torch.mean(var_per_bootstrap, dim=0)
            gp_se = torch.sqrt(gp_var / self.num_gp_samples)
            var_per_gp_sample = torch.var(second_order_idcs_list, dim=1)
            mc_var = torch.mean(var_per_gp_sample, dim=0)
            mc_se = torch.sqrt(mc_var / (self.num_bootstrap_samples + 1))
            num_second_order = second_order_idcs_list.shape[-1]
            total_mean = second_order_idcs_list.reshape(-1, num_second_order).mean(
                dim=0
            )
            second_order_idcs_mean_vargp_segp_varmc_segp = torch.cat(
                [
                    torch.tensor(
                        [total_mean[i], gp_var[i], gp_se[i], mc_var[i], mc_se[i]]
                    ).unsqueeze(0)
                    for i in range(num_second_order)
                ],
                dim=0,
            )
            return second_order_idcs_mean_vargp_segp_varmc_segp


def compute_sobol_indices_from_model_list(
    model_list: list[Model],
    bounds: Tensor,
    order: str = "first",
    discrete_features: list[int] | None = None,
    **sobol_kwargs: Any,
) -> Tensor:
    """
    Computes Sobol indices of a list of models on a bounded domain.

    Args:
        model_list: A list of botorch.models.model.Model types for which to compute
            the Sobol indices.
        bounds: A 2 x d Tensor of lower and upper bounds of the domain of the models.
        order: A string specifying the order of the Sobol indices to be computed.
            Supports "first", "second" and "total" and defaults to "first". "total"
            computes the importance of a variable considering its main effect and
            all of its higher-order interactions, whereas "first" and "second"
            the variance when altering the variable in isolation or with one other
            variable, respectively.
        discrete_features: If specified, the inputs associated with the indices in
            this list are generated using an integer-valued uniform distribution,
            rather than the default (pseudo-)random continuous uniform distribution.
        sobol_kwargs: keyword arguments passed on to SobolSensitivityGPMean.

    Returns:
        With m GPs, returns a (m x d) tensor of `order`-order Sobol indices.
    """
    if order not in ["first", "total", "second"]:
        raise NotImplementedError(
            f"Order {order} is not supported. Plese choose one of"
            " 'first', 'total' or 'second'."
        )
    indices = []
    method = getattr(SobolSensitivityGPMean, f"{order}_order_indices")
    second_order = order == "second"
    for model in model_list:
        sens_class = SobolSensitivityGPMean(
            model=model,
            bounds=bounds,
            discrete_features=discrete_features,
            second_order=second_order,
            **sobol_kwargs,
        )
        indices.append(method(sens_class))
    return torch.stack(indices)


def ax_parameter_sens(
    model_bridge: TorchModelBridge,
    metrics: list[str] | None = None,
    order: str = "first",
    signed: bool = True,
    **sobol_kwargs: Any,
) -> dict[str, dict[str, npt.NDArray]]:
    """
    Compute sensitivity for all metrics on an TorchModelBridge.

    Sobol measures are always positive regardless of the direction in which the
    parameter influences f. If `signed` is set to True, then the Sobol measure for each
    parameter will be given as its sign the sign of the average gradient with respect to
    that parameter across the search space. Thus, important parameters that, when
    increased, decrease f will have large and negative values; unimportant parameters
    will have values close to 0.

    Args:
        model_bridge: A ModelBridge object with models that were fit.
        metrics: The names of the metrics and outcomes for which to compute
            sensitivities. This should preferably be metrics with a good model fit.
            Defaults to model_bridge.outcomes.
        order: A string specifying the order of the Sobol indices to be computed.
            Supports "first" and "total" and defaults to "first".
        signed: A bool for whether the measure should be signed.
        sobol_kwargs: keyword arguments passed on to SobolSensitivityGPMean, and if
            signed, GpDGSMGpMean.

    Returns:
        Dictionary {'metric_name': {'parameter_name' or
        (parameter_name_1, 'parameter_name_2'): sensitivity_value}}, where the
        `sensitivity` value is cast to a Numpy array in order to be compatible with
        `plot_feature_importance_by_feature`.
    """
    if order not in ["first", "total", "second"]:
        raise NotImplementedError(
            f"Order {order} is not supported. Plese choose one of"
            " 'first', 'total' or 'second'."
        )
    if order == "second" and signed:
        raise NotImplementedError("Second order is not supported for signed indices.")
    if metrics is None:
        metrics = model_bridge.outcomes
    # can safely access _search_space_digest after type check
    torch_model = _get_torch_model(model_bridge)
    digest = torch_model.search_space_digest
    model_list = _get_model_per_metric(torch_model, metrics)
    bounds = torch.tensor(digest.bounds).T  # transposing to make it 2 x d

    # for second order indices, we need to compute first order indices first
    # which is what is done here. With the first order indices, we can then subtract
    # appropriately using the first-order indices to extract the second-order indices.
    ind = compute_sobol_indices_from_model_list(
        model_list=model_list,
        bounds=bounds,
        order="first" if order == "second" else order,
        discrete_features=digest.categorical_features + digest.ordinal_features,
        **sobol_kwargs,
    )
    if signed:
        ind_deriv = compute_derivatives_from_model_list(
            model_list=model_list,
            bounds=bounds,
            discrete_features=digest.categorical_features + digest.ordinal_features,
            **sobol_kwargs,
        )
        # categorical features don't have a direction, so we set the derivative to 1.0
        # in order not to zero our their sensitivity. We treat categorical features
        # separately in the sensitivity analysis plot as well, to make clear that they
        # are affecting the metric, but neither increasing nor decreasing. Note that the
        # orginal variables have a well defined direction, so we do not need to treat
        # them differently here.
        for i in digest.categorical_features:
            ind_deriv[:, i] = 1.0
        ind *= torch.sign(ind_deriv)

    feature_names = digest.feature_names
    indices = array_with_string_indices_to_dict(
        rows=metrics, cols=feature_names, A=ind.numpy()
    )
    if order == "second":
        second_order_values = compute_sobol_indices_from_model_list(
            model_list=model_list,
            bounds=bounds,
            order="second",
            discrete_features=digest.categorical_features + digest.ordinal_features,
            first_order_idcs=indices,
            **sobol_kwargs,
        )
        second_order_feature_names = [
            f"{f1} & {f2}" for f1, f2 in itertools.combinations(digest.feature_names, 2)
        ]

        second_order_dict = array_with_string_indices_to_dict(
            rows=metrics,
            cols=second_order_feature_names,
            A=second_order_values.numpy(),
        )
        for metric in metrics:
            indices[metric].update(second_order_dict[metric])

    return indices


def _get_torch_model(
    model_bridge: TorchModelBridge,
) -> BotorchModel | ModularBoTorchModel:
    """Returns the TorchModel of the model_bridge, if it is a type that stores
    SearchSpaceDigest during model fitting. At this point, this is BotorchModel, and
    ModularBoTorchModel.
    """
    if not isinstance(model_bridge, TorchModelBridge):
        raise NotImplementedError(
            f"{type(model_bridge)=}, but only TorchModelBridge is supported."
        )
    model = model_bridge.model  # should be of type TorchModel
    if not (isinstance(model, BotorchModel) or isinstance(model, ModularBoTorchModel)):
        raise NotImplementedError(
            f"{type(model_bridge.model)=}, but only "
            "Union[BotorchModel, ModularBoTorchModel] is supported."
        )
    return model


def _get_model_per_metric(
    model: BotorchModel | ModularBoTorchModel, metrics: list[str]
) -> list[Model]:
    """For a given TorchModel model, returns a list of botorch.models.model.Model
    objects corresponding to - and in the same order as - the given metrics.
    """
    if isinstance(model, BotorchModel):
        # guaranteed not to be None after accessing search_space_digest
        gp_model = model.model
        model_idx = [model.metric_names.index(m) for m in metrics]
        if not isinstance(gp_model, ModelList):
            if gp_model.num_outputs == 1:  # can accept single output models
                return [gp_model for _ in model_idx]
            raise NotImplementedError(
                f"type(model_bridge.model.model) = {type(gp_model)}, "
                "but only ModelList is supported."
            )
        return [gp_model.models[i] for i in model_idx]
    else:  # isinstance(model, ModularBoTorchModel):
        surrogate = model.surrogate
        outcomes = surrogate.outcomes
        model_list = []
        for m in metrics:  # for each metric, find a corresponding surrogate
            i = outcomes.index(m)
            metric_model = surrogate.model
            # since model is a ModularBoTorchModel, metric_model will be a
            # `botorch.models.model.Model` object, which have the `num_outputs`
            # property and `subset_outputs` method.
            if metric_model.num_outputs > 1:  # subset to relevant output
                metric_model = metric_model.subset_output([i])
            model_list.append(metric_model)
        return model_list


def array_with_string_indices_to_dict(
    rows: list[str],
    cols: list[str],
    A: npt.NDArray,
) -> dict[str, dict[str, npt.NDArray]]:
    """
    Args:
        rows: A list of strings with which to index rows of A.
        cols: A list of strings with which to index columns of A.
        A: A matrix, with `len(rows)` rows and `len(cols)` columns.

    Returns:
        A dictionary dict that satisfies dict[rows[i]][cols[j]] = A[i, j].
    """
    return {r: dict(zip(cols, a)) for r, a in zip(rows, A)}
