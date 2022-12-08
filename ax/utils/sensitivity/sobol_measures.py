# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Callable, List, Optional

import torch
from ax.utils.common.typeutils import checked_cast
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize
from torch._tensor import Tensor


class SobolSensitivity(object):
    def __init__(
        self,
        dim: int,
        bounds: torch.Tensor,
        input_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        num_mc_samples: int = 10**4,
        input_qmc: bool = False,
        second_order: bool = False,
        num_bootstrap_samples: int = 1,
        bootstrap_array: bool = False,
    ) -> None:
        r"""Computes three types of Sobol indices:
        first order indices, total indices and second order indices (if specified ).

        Args:
            dim: The dimension of the function.
            bounds: Parameter bounds over which to evaluate model sensitivity.
            input_function: The objective function.
            num_mc_samples: The number of montecarlo grid samples
            input_qmc: If True, a qmc Sobol grid is use instead of uniformly random.
            second_order: If True, the second order indices are computed.
            bootstrap: If true, the MC error is returned.
            num_bootstrap_samples: If bootstrap is true, the number of bootstraps has
                to be specified.
            bootstrap_array: If true, all the num_bootstrap_samples extimated indices
                are returned instead of their mean and Var.
        """
        self.input_function = input_function
        self.dim = dim
        self.num_mc_samples = num_mc_samples
        self.second_order = second_order
        # pyre-fixme[4]: Attribute must be annotated.
        self.bootstrap = num_bootstrap_samples > 1
        # pyre-fixme[4]: Attribute must be annotated.
        self.num_bootstrap_samples = (
            num_bootstrap_samples - 1
        )  # deduct 1 because the first is meant to be the full grid
        self.bootstrap_array = bootstrap_array
        if input_qmc:
            # pyre-fixme[4]: Attribute must be annotated.
            self.A = draw_sobol_samples(bounds=bounds, n=num_mc_samples, q=1).squeeze(1)
            # pyre-fixme[4]: Attribute must be annotated.
            self.B = draw_sobol_samples(bounds=bounds, n=num_mc_samples, q=1).squeeze(1)
        else:
            self.A = unnormalize(torch.rand(num_mc_samples, dim), bounds=bounds)
            self.B = unnormalize(torch.rand(num_mc_samples, dim), bounds=bounds)
        # pyre-fixme[4]: Attribute must be annotated.
        self.A_B_ABi = self.generate_all_input_matrix().to(torch.double)

        if self.bootstrap:
            subset_size = 4
            # pyre-fixme[4]: Attribute must be annotated.
            self.bootstrap_indices = torch.randint(
                0, num_mc_samples, (self.num_bootstrap_samples, subset_size)
            )
        self.f_A: Optional[torch.Tensor] = None
        self.f_B: Optional[torch.Tensor] = None
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        self.f_ABis: Optional[List] = None
        self.f_total_var: Optional[torch.Tensor] = None
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        self.f_A_btsp: Optional[List] = None
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        self.f_B_btsp: Optional[List] = None
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        self.f_ABis_btsp: Optional[List] = None
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        self.f_total_var_btsp: Optional[List] = None
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        self.f_BAis: Optional[List] = None
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        self.f_BAis_btsp: Optional[List] = None
        self.first_order_idxs: Optional[torch.Tensor] = None
        self.first_order_idxs_btsp: Optional[torch.Tensor] = None

    def generate_all_input_matrix(self) -> torch.Tensor:
        A_B_ABi = torch.cat((self.A, self.B), dim=0)
        for i in range(self.dim):
            AB_i = deepcopy(self.A)
            AB_i[:, i] = self.B[:, i]
            A_B_ABi = torch.cat((A_B_ABi, AB_i), dim=0)
        if self.second_order:
            for i in range(self.dim):
                BA_i = deepcopy(self.B)
                BA_i[:, i] = self.A[:, i]
                A_B_ABi = torch.cat((A_B_ABi, BA_i), dim=0)
        return A_B_ABi

    def evalute_function(self, f_A_B_ABi: Optional[torch.Tensor] = None) -> None:
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
        self.f_ABis = [
            f_A_B_ABi[self.num_mc_samples * (i + 2) : self.num_mc_samples * (i + 3)]
            for i in range(self.dim)
        ]
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
                        (self.f_A_btsp[i], self.f_B_btsp[i]), dim=0  # pyre-ignore
                    )
                )
                for i in range(self.num_bootstrap_samples)
            ]
        if self.second_order:
            self.f_BAis = [
                f_A_B_ABi[
                    self.num_mc_samples
                    * (i + 2 + self.dim) : self.num_mc_samples
                    * (i + 3 + self.dim)
                ]
                for i in range(self.dim)
            ]
            if self.bootstrap:
                self.f_BAis_btsp = [
                    [torch.index_select(f_BAi, 0, indices) for f_BAi in self.f_BAis]
                    for indices in self.bootstrap_indices
                ]

    def first_order_indices(self) -> torch.Tensor:
        r"""Computes the first order Sobol indices:

        Returns:
            if num_bootstrap_samples>1
                Tensor: (values,var_mc,stderr_mc)x dim
            else
                Tensor: (values)x dim
        """
        first_order_idxs = []
        for i in range(self.dim):
            vi = (
                torch.mean(self.f_B * (self.f_ABis[i] - self.f_A))  # pyre-ignore
                / self.f_total_var
            )
            first_order_idxs.append(vi.unsqueeze(0))
        self.first_order_idxs = torch.cat(first_order_idxs, dim=0).detach()
        if not (self.bootstrap):
            return self.first_order_idxs
        else:
            first_order_idxs_btsp = [torch.cat(first_order_idxs, dim=0).unsqueeze(0)]
            for b in range(self.num_bootstrap_samples):
                first_order_idxs = []
                for i in range(self.dim):
                    vi = (
                        torch.mean(
                            self.f_B_btsp[b]
                            * (self.f_ABis_btsp[b][i] - self.f_A_btsp[b])
                        )
                        / self.f_total_var_btsp[b]
                    )
                    first_order_idxs.append(vi.unsqueeze(0))
                first_order_idxs_btsp.append(
                    torch.cat(first_order_idxs, dim=0).unsqueeze(0)
                )
            self.first_order_idxs_btsp = torch.cat(first_order_idxs_btsp, dim=0)
            if self.bootstrap_array:
                return self.first_order_idxs_btsp.detach()
            else:
                return (
                    torch.cat(
                        [
                            self.first_order_idxs_btsp.mean(dim=0).unsqueeze(0),
                            self.first_order_idxs_btsp.var(  # pyre-ignore
                                dim=0
                            ).unsqueeze(0),
                            torch.sqrt(
                                self.first_order_idxs_btsp.var(dim=0)
                                / (self.num_bootstrap_samples + 1)
                            ).unsqueeze(0),
                        ],
                        dim=0,
                    )
                    .t()
                    .detach()
                )

    # pyre-fixme[3]: Return type must be annotated.
    def total_order_indices(self):
        r"""Computes the total Sobol indices:

        Returns:
            if num_bootstrap_samples>1
                Tensor: (values,var_mc,stderr_mc)x dim
            else
                Tensor: (values)x dim
        """
        total_order_idxs = []
        for i in range(self.dim):
            vti = (
                0.5
                # pyre-fixme[58]: `-` is not supported for operand types
                #  `Optional[torch._tensor.Tensor]` and `Any`.
                # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
                * torch.mean(torch.pow(self.f_A - self.f_ABis[i], 2))
                / self.f_total_var
            )
            total_order_idxs.append(vti.unsqueeze(0))
        if not (self.bootstrap):
            total_order_idxs = torch.cat(total_order_idxs, dim=0).detach()
            return total_order_idxs
        else:
            total_order_idxs_btsp = [torch.cat(total_order_idxs, dim=0).unsqueeze(0)]
            for b in range(self.num_bootstrap_samples):
                total_order_idxs = []
                for i in range(self.dim):
                    vti = (
                        0.5
                        * torch.mean(
                            torch.pow(self.f_A_btsp[b] - self.f_ABis_btsp[b][i], 2)
                        )
                        / self.f_total_var_btsp[b]
                    )
                    total_order_idxs.append(vti.unsqueeze(0))
                total_order_idxs_btsp.append(
                    torch.cat(total_order_idxs, dim=0).unsqueeze(0)
                )
            total_order_idxs_btsp = torch.cat(total_order_idxs_btsp, dim=0)
            if self.bootstrap_array:
                return total_order_idxs_btsp.detach()
            else:
                return (
                    torch.cat(
                        [
                            total_order_idxs_btsp.mean(dim=0).unsqueeze(0),
                            total_order_idxs_btsp.var(dim=0).unsqueeze(0),
                            torch.sqrt(
                                total_order_idxs_btsp.var(dim=0)
                                / (self.num_bootstrap_samples + 1)
                            ).unsqueeze(0),
                        ],
                        dim=0,
                    )
                    .t()
                    .detach()
                )

    # pyre-fixme[3]: Return type must be annotated.
    def second_order_indices(
        self,
        first_order_idxs: Optional[torch.Tensor] = None,
        first_order_idxs_btsp: Optional[torch.Tensor] = None,
    ):
        r"""Computes the Second order Sobol indices:
        Args:
            first_order_idxs: Tensor of first order indices.
            first_order_idxs_btsp: Tensor of all first order indices given by bootstrap.
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
        if first_order_idxs is None:
            first_order_idxs = self.first_order_idxs
        second_order_idxs = []
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                vij = torch.mean(
                    self.f_BAis[i] * self.f_ABis[j] - self.f_A * self.f_B  # pyre-ignore
                )
                vij = (
                    (vij / self.f_total_var)
                    - first_order_idxs[i]  # pyre-ignore
                    - first_order_idxs[j]
                )
                second_order_idxs.append(vij.unsqueeze(0))
        if not (self.bootstrap):
            second_order_idxs = torch.cat(second_order_idxs, dim=0).detach()
            return second_order_idxs
        else:
            second_order_idxs_btsp = [torch.cat(second_order_idxs, dim=0).unsqueeze(0)]
            if first_order_idxs_btsp is None:
                first_order_idxs_btsp = self.first_order_idxs_btsp
            for b in range(self.num_bootstrap_samples):
                second_order_idxs = []
                for i in range(self.dim):
                    for j in range(i + 1, self.dim):
                        vij = torch.mean(
                            self.f_BAis_btsp[b][i] * self.f_ABis_btsp[b][j]
                            - self.f_A_btsp[b] * self.f_B_btsp[b]
                        )
                        vij = (
                            (vij / self.f_total_var_btsp[b])
                            - first_order_idxs_btsp[b][i]
                            - first_order_idxs_btsp[b][j]
                        )
                        second_order_idxs.append(vij.unsqueeze(0))
                second_order_idxs_btsp.append(
                    torch.cat(second_order_idxs, dim=0).unsqueeze(0)
                )
            second_order_idxs_btsp = torch.cat(second_order_idxs_btsp, dim=0)
            if self.bootstrap_array:
                return second_order_idxs_btsp.detach()
            else:
                return (
                    torch.cat(
                        [
                            second_order_idxs_btsp.mean(dim=0).unsqueeze(0),
                            second_order_idxs_btsp.var(dim=0).unsqueeze(0),
                            torch.sqrt(
                                second_order_idxs_btsp.var(dim=0)
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


class SobolSensitivityGPMean(object):
    def __init__(
        self,
        model: Model,
        bounds: torch.Tensor,
        num_mc_samples: int = 10**4,
        second_order: bool = False,
        input_qmc: bool = False,
        num_bootstrap_samples: int = 1,
        link_function: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = GaussianLinkMean,
    ) -> None:
        r"""Computes three types of Sobol indices:
        first order indices, total indices and second order indices (if specified ).

        Args:
            model: Botorch model
            bounds: Parameter bounds over which to evaluate model sensitivity.
            method: if "predictive mean", the predictive mean is used for indices
                computation. If "GP samples", posterior sampling is used instead.
            num_mc_samples: The number of montecarlo grid samples
            second_order: If True, the second order indices are computed.
            input_qmc: If True, a qmc Sobol grid is use instead of uniformly random.
            num_bootstrap_samples: If bootstrap is true, the number of bootstraps has
                to be specified.
        """
        self.model = model
        self.dim = model.train_inputs[0].shape[-1]  # pyre-ignore
        self.second_order = second_order
        self.input_qmc = input_qmc
        # pyre-fixme[4]: Attribute must be annotated.
        self.bootstrap = num_bootstrap_samples > 1
        self.num_bootstrap_samples = num_bootstrap_samples
        self.num_mc_samples = num_mc_samples

        def input_function(x: Tensor) -> Tensor:
            with torch.no_grad():
                p = checked_cast(GPyTorchPosterior, self.model.posterior(x))
            return link_function(p.mean, p.variance)

        self.sensitivity = SobolSensitivity(
            dim=self.dim,
            num_mc_samples=self.num_mc_samples,
            input_function=input_function,
            bounds=bounds,
            second_order=self.second_order,
            input_qmc=self.input_qmc,
            num_bootstrap_samples=self.num_bootstrap_samples,
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

    # pyre-fixme[3]: Return type must be annotated.
    def total_order_indices(self):
        r"""Computes the total Sobol indices:

        Returns:
            if num_bootstrap_samples>1
                Tensor: (values,var_mc,stderr_mc)x dim
            else
                Tensor: (values)x dim
        """
        return self.sensitivity.total_order_indices()

    # pyre-fixme[3]: Return type must be annotated.
    def second_order_indices(self):
        r"""Computes the Second order Sobol indices:

        Returns:
            if num_bootstrap_samples>1
                Tensor: (values,var_mc,stderr_mc)x dim(dim-1)/2
            else
                Tensor: (values)x dim(dim-1)/2
        """
        return self.sensitivity.second_order_indices()


class SobolSensitivityGPSampling(object):
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
    ) -> None:
        r"""Computes three types of Sobol indices:
        first order indices, total indices and second order indices (if specified ).

        Args:
            model: Botorch model.
            bounds: Parameter bounds over which to evaluate model sensitivity.
            num_gp_samples: If method is "GP samples", the number of GP samples
                has to be set.
            num_mc_samples: The number of montecarlo grid samples
            second_order: If True, the second order indices are computed.
            input_qmc: If True, a qmc Sobol grid is use instead of uniformly random.
            gp_sample_qmc: If True, the posterior sampling is done using
                SobolQMCNormalSampler.
            num_bootstrap_samples: If bootstrap is true, the number of bootstraps has
                to be specified.
        """
        self.model = model
        self.dim = model.train_inputs[0].shape[-1]  # pyre-ignore
        self.second_order = second_order
        self.input_qmc = input_qmc
        self.gp_sample_qmc = gp_sample_qmc
        # pyre-fixme[4]: Attribute must be annotated.
        self.bootstrap = num_bootstrap_samples > 1
        self.num_bootstrap_samples = num_bootstrap_samples
        self.num_mc_samples = num_mc_samples
        self.num_gp_samples = num_gp_samples
        self.sensitivity = SobolSensitivity(
            dim=self.dim,
            num_mc_samples=self.num_mc_samples,
            bounds=bounds,
            second_order=self.second_order,
            input_qmc=self.input_qmc,
            num_bootstrap_samples=self.num_bootstrap_samples,
            bootstrap_array=True,
        )
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

    def first_order_indices(self) -> Tensor:
        r"""Computes the first order Sobol indices:

        Returns:
            if num_bootstrap_samples>1
                Tensor: (values,var_gp,stderr_gp,var_mc,stderr_mc)x dim
            else
                Tensor: (values,var,stderr)x dim
        """
        first_order_idxs_list = []
        for j in range(self.num_gp_samples):
            self.sensitivity.evalute_function(self.samples[j])
            first_order_idxs = self.sensitivity.first_order_indices()
            first_order_idxs_list.append(first_order_idxs.unsqueeze(0))
        # pyre-fixme[16]: `SobolSensitivityGPSampling` has no attribute
        #  `first_order_idxs_list`.
        self.first_order_idxs_list = torch.cat(first_order_idxs_list, dim=0)
        if not (self.bootstrap):
            first_order_idxs_mean_var_se = []
            for i in range(self.dim):
                first_order_idxs_mean_var_se.append(
                    torch.tensor(
                        [
                            torch.mean(self.first_order_idxs_list[:, i]),
                            torch.var(self.first_order_idxs_list[:, i]),
                            torch.sqrt(
                                torch.var(self.first_order_idxs_list[:, i])
                                / self.num_gp_samples
                            ),
                        ]
                    ).unsqueeze(0)
                )
            first_order_idxs_mean_var_se = torch.cat(
                first_order_idxs_mean_var_se, dim=0
            )
            return first_order_idxs_mean_var_se
        else:
            var_per_bootstrap = torch.var(self.first_order_idxs_list, dim=0)
            gp_var = torch.mean(var_per_bootstrap, dim=0)
            gp_se = torch.sqrt(gp_var / self.num_gp_samples)
            var_per_gp_sample = torch.var(self.first_order_idxs_list, dim=1)
            mc_var = torch.mean(var_per_gp_sample, dim=0)
            mc_se = torch.sqrt(mc_var / (self.num_bootstrap_samples + 1))
            total_mean = self.first_order_idxs_list.reshape(-1, self.dim).mean(dim=0)
            first_order_idxs_mean_vargp_segp_varmc_segp = torch.cat(
                [
                    torch.tensor(
                        [total_mean[i], gp_var[i], gp_se[i], mc_var[i], mc_se[i]]
                    ).unsqueeze(0)
                    for i in range(self.dim)
                ],
                dim=0,
            )
            return first_order_idxs_mean_vargp_segp_varmc_segp

    def total_order_indices(self) -> Tensor:
        r"""Computes the total Sobol indices:

        Returns:
            if num_bootstrap_samples>1
                Tensor: (values,var_gp,stderr_gp,var_mc,stderr_mc)x dim
            else
                Tensor: (values,var,stderr)x dim
        """
        total_order_idxs_list = []
        for j in range(self.num_gp_samples):
            self.sensitivity.evalute_function(self.samples[j])
            total_order_idxs = self.sensitivity.total_order_indices()
            total_order_idxs_list.append(total_order_idxs.unsqueeze(0))
        total_order_idxs_list = torch.cat(total_order_idxs_list, dim=0)
        if not (self.bootstrap):
            total_order_idxs_mean_var = []
            for i in range(self.dim):
                total_order_idxs_mean_var.append(
                    torch.tensor(
                        [
                            torch.mean(total_order_idxs_list[:, i]),
                            torch.var(total_order_idxs_list[:, i]),
                            torch.sqrt(
                                torch.var(total_order_idxs_list[:, i])
                                / self.num_gp_samples
                            ),
                        ]
                    ).unsqueeze(0)
                )
            total_order_idxs_mean_var = torch.cat(total_order_idxs_mean_var, dim=0)
            return total_order_idxs_mean_var
        else:
            var_per_bootstrap = torch.var(total_order_idxs_list, dim=0)
            gp_var = torch.mean(var_per_bootstrap, dim=0)
            gp_se = torch.sqrt(gp_var / self.num_gp_samples)
            var_per_gp_sample = torch.var(total_order_idxs_list, dim=1)
            mc_var = torch.mean(var_per_gp_sample, dim=0)
            mc_se = torch.sqrt(mc_var / (self.num_bootstrap_samples + 1))
            total_mean = total_order_idxs_list.reshape(-1, self.dim).mean(dim=0)
            total_order_idxs_mean_vargp_segp_varmc_segp = torch.cat(
                [
                    torch.tensor(
                        [total_mean[i], gp_var[i], gp_se[i], mc_var[i], mc_se[i]]
                    ).unsqueeze(0)
                    for i in range(self.dim)
                ],
                dim=0,
            )
            return total_order_idxs_mean_vargp_segp_varmc_segp

    def second_order_indices(self) -> Tensor:
        r"""Computes the Second order Sobol indices:

        Returns:
            if num_bootstrap_samples>1
                Tensor: (values,var_gp,stderr_gp,var_mc,stderr_mc)x dim(dim-1)/2
            else
                Tensor: (values,var,stderr)x dim(dim-1)/2
        """
        if not (self.bootstrap):
            second_order_idxs_list = []
            for j in range(self.num_gp_samples):
                self.sensitivity.evalute_function(self.samples[j])
                second_order_idxs = self.sensitivity.second_order_indices(
                    # pyre-fixme[16]: `SobolSensitivityGPSampling` has no attribute
                    #  `first_order_idxs_list`.
                    self.first_order_idxs_list[j]
                )
                second_order_idxs_list.append(second_order_idxs.unsqueeze(0))
            second_order_idxs_list = torch.cat(second_order_idxs_list, dim=0)
            second_order_idxs_mean_var = []
            for i in range(len(second_order_idxs)):
                second_order_idxs_mean_var.append(
                    torch.tensor(
                        [
                            torch.mean(second_order_idxs_list[:, i]),
                            torch.var(second_order_idxs_list[:, i]),
                            torch.sqrt(
                                torch.var(second_order_idxs_list[:, i])
                                / self.num_gp_samples
                            ),
                        ]
                    ).unsqueeze(0)
                )
            second_order_idxs_mean_var = torch.cat(second_order_idxs_mean_var, dim=0)
            return second_order_idxs_mean_var
        else:
            second_order_idxs_list = []
            for j in range(self.num_gp_samples):
                self.sensitivity.evalute_function(self.samples[j])
                second_order_idxs = self.sensitivity.second_order_indices(
                    self.first_order_idxs_list[j][0],
                    self.first_order_idxs_list[j][1:],
                )
                second_order_idxs_list.append(second_order_idxs.unsqueeze(0))
            second_order_idxs_list = torch.cat(second_order_idxs_list, dim=0)
            var_per_bootstrap = torch.var(second_order_idxs_list, dim=0)
            gp_var = torch.mean(var_per_bootstrap, dim=0)
            gp_se = torch.sqrt(gp_var / self.num_gp_samples)
            var_per_gp_sample = torch.var(second_order_idxs_list, dim=1)
            mc_var = torch.mean(var_per_gp_sample, dim=0)
            mc_se = torch.sqrt(mc_var / (self.num_bootstrap_samples + 1))
            num_second_order = second_order_idxs_list.shape[-1]
            total_mean = second_order_idxs_list.reshape(-1, num_second_order).mean(
                dim=0
            )
            second_order_idxs_mean_vargp_segp_varmc_segp = torch.cat(
                [
                    torch.tensor(
                        [total_mean[i], gp_var[i], gp_se[i], mc_var[i], mc_se[i]]
                    ).unsqueeze(0)
                    for i in range(num_second_order)
                ],
                dim=0,
            )
            return second_order_idxs_mean_vargp_segp_varmc_segp
