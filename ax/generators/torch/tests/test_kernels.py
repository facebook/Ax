#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from itertools import product
from math import sqrt

import torch
from ax.exceptions.core import AxError
from ax.generators.torch.botorch_modular.kernels import (
    DefaultMaternKernel,
    DefaultRBFKernel,
    ScaleMaternKernel,
    ScaleRBFLinearKernel,
    TemporalKernel,
)
from ax.utils.common.testutils import TestCase
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from gpytorch.constraints import Positive
from gpytorch.kernels import (
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
    ScaleKernel,
)
from gpytorch.priors import GammaPrior
from pyre_extensions import assert_is_instance


class KernelsTest(TestCase):
    def test_scalematern_kernel(self) -> None:
        covar = ScaleMaternKernel(
            ard_num_dims=10,
            lengthscale_prior=GammaPrior(6.0, 3.0),
            outputscale_prior=GammaPrior(2.0, 0.15),
            batch_shape=torch.Size([2]),
        )
        self.assertTrue(isinstance(covar.base_kernel, MaternKernel))
        self.assertTrue(isinstance(covar.base_kernel, MaternKernel))
        self.assertEqual(covar.base_kernel.ard_num_dims, 10)
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `rate`.
        self.assertEqual(covar.base_kernel.lengthscale_prior.rate, 3.0)
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
        #  `concentration`.
        self.assertEqual(covar.base_kernel.lengthscale_prior.concentration, 6.0)
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `rate`.
        self.assertEqual(covar.outputscale_prior.rate, 0.15)
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
        #  `concentration`.
        self.assertEqual(covar.outputscale_prior.concentration, 2.0)
        self.assertEqual(covar.base_kernel.batch_shape[0], 2)
        self.assertIsNone(covar.active_dims)

    def test_scalematern_kernel_active_dims(self) -> None:
        active_dims = [0, 2]
        covar = ScaleMaternKernel(
            ard_num_dims=len(active_dims), active_dims=active_dims
        )
        base_kernel = assert_is_instance(covar.base_kernel, MaternKernel)
        # active_dims lands on the inner MaternKernel, and the ScaleKernel
        # inherits it (so subsetting happens exactly once at the wrapper level).
        self.assertEqual(covar.active_dims.tolist(), active_dims)
        self.assertEqual(base_kernel.active_dims.tolist(), active_dims)
        self.assertEqual(base_kernel.ard_num_dims, len(active_dims))
        # The kernel only consumes the active columns: perturbing an inactive
        # column leaves the covariance unchanged.
        X = torch.randn(5, 3)
        X_perturbed = X.clone()
        X_perturbed[:, 1] = torch.randn(5)  # column 1 is inactive
        self.assertTrue(
            torch.allclose(covar(X).to_dense(), covar(X_perturbed).to_dense())
        )

    def test_scale_rbf_linear_kernel(self) -> None:
        covar = ScaleRBFLinearKernel(
            ard_num_dims=10,
            lengthscale_prior=GammaPrior(6.0, 3.0),
            outputscale_prior=GammaPrior(2.0, 0.15),
            variance_prior=GammaPrior(1.0, 1.0),
            batch_shape=torch.Size([2]),
        )
        # The kernel is a sum of a ScaleKernel(RBF) and a LinearKernel.
        scale_rbf = assert_is_instance(covar.kernels[0], ScaleKernel)
        linear = assert_is_instance(covar.kernels[1], LinearKernel)
        rbf = assert_is_instance(scale_rbf.base_kernel, RBFKernel)
        # RBF lengthscale prior and ard_num_dims.
        self.assertEqual(rbf.ard_num_dims, 10)
        lengthscale_prior = assert_is_instance(rbf.lengthscale_prior, GammaPrior)
        self.assertEqual(lengthscale_prior.rate, 3.0)
        self.assertEqual(lengthscale_prior.concentration, 6.0)
        # Outputscale prior on the ScaleKernel.
        outputscale_prior = assert_is_instance(scale_rbf.outputscale_prior, GammaPrior)
        self.assertEqual(outputscale_prior.rate, 0.15)
        self.assertEqual(outputscale_prior.concentration, 2.0)
        # Variance prior on the LinearKernel.
        variance_prior = assert_is_instance(linear.variance_prior, GammaPrior)
        self.assertEqual(variance_prior.rate, 1.0)
        self.assertEqual(variance_prior.concentration, 1.0)
        # Batch shape is shared by both components.
        self.assertEqual(rbf.batch_shape, torch.Size([2]))
        self.assertEqual(linear.batch_shape, torch.Size([2]))
        # The kernel evaluates and produces a PSD covariance.
        X = torch.randn(2, 5, 10)
        covar_matrix = covar(X).to_dense()
        self.assertEqual(covar_matrix.shape, torch.Size([2, 5, 5]))

    def test_scale_rbf_linear_kernel_active_dims(self) -> None:
        active_dims = [0, 2]
        covar = ScaleRBFLinearKernel(
            ard_num_dims=len(active_dims),
            active_dims=active_dims,
        )
        scale_rbf = assert_is_instance(covar.kernels[0], ScaleKernel)
        linear = assert_is_instance(covar.kernels[1], LinearKernel)
        rbf = assert_is_instance(scale_rbf.base_kernel, RBFKernel)
        # active_dims lands on both leaves, and the ScaleKernel inherits it
        # from the wrapped RBF kernel (so subsetting happens exactly once).
        self.assertEqual(scale_rbf.active_dims.tolist(), active_dims)
        self.assertEqual(rbf.active_dims.tolist(), active_dims)
        self.assertEqual(linear.active_dims.tolist(), active_dims)
        # ard_num_dims matches the number of active dims.
        self.assertEqual(rbf.ard_num_dims, len(active_dims))
        # The kernel only consumes the active columns: it produces the same
        # covariance regardless of the values in the inactive column.
        X = torch.randn(5, 3)
        X_perturbed = X.clone()
        X_perturbed[:, 1] = torch.randn(5)  # column 1 is inactive
        covar_matrix = covar(X).to_dense()
        covar_matrix_perturbed = covar(X_perturbed).to_dense()
        self.assertTrue(torch.allclose(covar_matrix, covar_matrix_perturbed))
        self.assertEqual(covar_matrix.shape, torch.Size([5, 5]))

    def test_temporal_kernel(self) -> None:
        ls_prior = GammaPrior(6.0, 3.0)
        os_prior = GammaPrior(2.0, 0.15)
        tls_prior = GammaPrior(3.0, 6.0)
        pl_prior = GammaPrior(2.0, 6.0)
        temporal_features = [8, 9]
        ls_constraint = Positive()
        os_constraint = Positive()
        tls_constraint = Positive()
        pl_constraint = Positive()
        for fixed_period_length, batch_shape, matern_ard_num_dims in product(
            (6.0, None), (torch.Size([]), torch.Size([2])), (10, None)
        ):
            covar = TemporalKernel(
                dim=12,
                matern_ard_num_dims=matern_ard_num_dims,
                temporal_features=temporal_features,
                batch_shape=batch_shape,
                lengthscale_prior=ls_prior,
                outputscale_prior=os_prior,
                temporal_lengthscale_prior=tls_prior,
                period_length_prior=pl_prior if fixed_period_length is None else None,
                fixed_period_length=fixed_period_length,
                lengthscale_constraint=ls_constraint,
                outputscale_constraint=os_constraint,
                period_length_constraint=(
                    pl_constraint if fixed_period_length is None else None
                ),
                temporal_lengthscale_constraint=tls_constraint,
            )
            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[An...
            self.assertTrue(isinstance(covar.base_kernel.kernels[0], MaternKernel))
            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[An...
            self.assertTrue(isinstance(covar.base_kernel.kernels[1], PeriodicKernel))
            # pyre-fixme[23]: Unable to unpack `Tensor | Module` into 2 values.
            matern, periodic = covar.base_kernel.kernels

            self.assertEqual(matern.ard_num_dims, matern_ard_num_dims)
            self.assertEqual(
                matern.active_dims.tolist(),
                list(set(range(12)) - set(temporal_features)),
            )
            self.assertIs(matern.lengthscale_prior, ls_prior)
            self.assertEqual(periodic.ard_num_dims, 2)
            self.assertEqual(periodic.active_dims.tolist(), temporal_features)
            self.assertIs(periodic.lengthscale_prior, tls_prior)
            if fixed_period_length is None:
                self.assertIs(periodic.period_length_prior, pl_prior)
                self.assertIs(periodic.raw_period_length_constraint, pl_constraint)
            self.assertIs(matern.raw_lengthscale_constraint, ls_constraint)
            self.assertIs(periodic.raw_lengthscale_constraint, tls_constraint)
            self.assertIs(covar.raw_outputscale_constraint, os_constraint)
            self.assertIs(covar.outputscale_prior, os_prior)
            self.assertEqual(matern.batch_shape, batch_shape)
            self.assertEqual(periodic.batch_shape, batch_shape)
            if fixed_period_length is not None:
                self.assertTrue(
                    torch.all(periodic.period_length == fixed_period_length)
                )
        # test no temporal features
        msg = "The temporal kernel should only be used if there are temporal features."
        with self.assertRaisesRegex(AxError, msg):
            TemporalKernel(
                dim=10,
                matern_ard_num_dims=10,
                temporal_features=[],
                batch_shape=torch.Size([]),
                lengthscale_prior=ls_prior,
                outputscale_prior=os_prior,
                temporal_lengthscale_prior=tls_prior,
                period_length_prior=pl_prior,
                lengthscale_constraint=ls_constraint,
                outputscale_constraint=os_constraint,
                period_length_constraint=pl_constraint,
                temporal_lengthscale_constraint=tls_constraint,
            )
        # test exception raise when fixed_period_length is not None
        # and period_length_prior or period_length_constraint are provided
        msg = (
            "If `fixed_period_length` is provided, then `period_length_prior` "
            "and `period_length_constraint` are not used."
        )
        with self.assertRaisesRegex(ValueError, msg):
            TemporalKernel(
                dim=10,
                matern_ard_num_dims=8,
                temporal_features=[8, 9],
                batch_shape=torch.Size([]),
                lengthscale_prior=ls_prior,
                outputscale_prior=os_prior,
                fixed_period_length=5.0,
                temporal_lengthscale_prior=tls_prior,
                period_length_prior=pl_prior,
                lengthscale_constraint=ls_constraint,
                outputscale_constraint=os_constraint,
                temporal_lengthscale_constraint=tls_constraint,
            )
        with self.assertRaisesRegex(ValueError, msg):
            TemporalKernel(
                dim=10,
                matern_ard_num_dims=8,
                temporal_features=[8, 9],
                batch_shape=torch.Size([]),
                lengthscale_prior=ls_prior,
                outputscale_prior=os_prior,
                fixed_period_length=5.0,
                temporal_lengthscale_prior=tls_prior,
                period_length_constraint=pl_constraint,
                lengthscale_constraint=ls_constraint,
                outputscale_constraint=os_constraint,
                temporal_lengthscale_constraint=tls_constraint,
            )

    def test_default_kernel(self) -> None:
        # Check that this is equivalent to the BoTorch defaults.
        for use_rbf_kernel, kwargs in product(
            (True, False),
            (
                {"ard_num_dims": 1},
                {
                    "ard_num_dims": 2,
                    "active_dims": [0, 1],
                    "batch_shape": torch.Size([2]),
                },
            ),
        ):
            kernel_class = DefaultRBFKernel if use_rbf_kernel else DefaultMaternKernel
            ax_kernel = kernel_class(**kwargs)  # pyre-ignore
            botorch_kernel = get_covar_module_with_dim_scaled_prior(
                use_rbf_kernel=use_rbf_kernel,
                **kwargs,  # pyre-ignore [6]
            )
            # Compare the state dicts for underlying compotents & their shapes.
            ax_dict = ax_kernel.state_dict()
            botorch_dict = botorch_kernel.state_dict()
            for k in ax_dict | botorch_dict:
                ax_value = ax_dict[k]
                botorch_value = botorch_dict[k]
                self.assertTrue(torch.equal(ax_value, botorch_value))
            # Active dims will not reflected in the state dict.
            if ax_kernel.active_dims is None:
                self.assertIsNone(botorch_kernel.active_dims)
            else:
                self.assertEqual(
                    ax_kernel.active_dims.tolist(), botorch_kernel.active_dims.tolist()
                )

    def test_default_mle(self) -> None:
        active_dims = [0, 1]
        for kernel_class in (DefaultRBFKernel, DefaultMaternKernel):
            kernel = kernel_class(
                ard_num_dims=2,
                active_dims=active_dims,
                batch_shape=torch.Size([3]),
                mle=True,
            )
            self.assertTrue((kernel.lengthscale == sqrt(2) / 10).all())
            self.assertEqual(kernel.lengthscale.shape, torch.Size([3, 1, 2]))
            self.assertFalse(hasattr(kernel, "lengthscale_prior"))
            self.assertEqual(kernel.active_dims.tolist(), active_dims)
