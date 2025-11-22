#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

import torch
from ax.adapter.registry import Generators
from ax.adapter.transforms.choice_encode import OrderedChoiceToIntegerRange
from ax.generators.torch.botorch_modular.optimizer_defaults import (
    BATCH_LIMIT,
    INIT_BATCH_LIMIT,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import mock_botorch_optimize_context_manager
from botorch.acquisition.analytic import ExpectedImprovement, PosteriorMean
from botorch.generation.gen import gen_candidates_scipy
from botorch.optim.optimize_mixed import generate_starting_points
from botorch.utils.testing import MockAcquisitionFunction, skip_if_import_error
from pyro.infer import MCMC


class TestMock(TestCase):
    def test_no_mocks_called(self) -> None:
        # Should raise by default if no mocks are called.
        with self.assertRaisesRegex(AssertionError, "No mocks were called"):
            with mock_botorch_optimize_context_manager():
                pass
        # Doesn't raise when force=True.
        with mock_botorch_optimize_context_manager(force=True):
            pass

    def test_botorch_mocks(self) -> None:
        # Should not raise when BoTorch mocks are called.
        with mock_botorch_optimize_context_manager():
            gen_candidates_scipy(
                initial_conditions=torch.tensor([[0.0]]),
                acquisition_function=MockAcquisitionFunction(),  # pyre-ignore [6]
            )

    def test_fully_bayesian_mocks(self) -> None:
        experiment = get_branin_experiment(with_completed_batch=True)
        with patch("botorch.fit.MCMC", wraps=MCMC) as mock_mcmc:
            with mock_botorch_optimize_context_manager():
                Generators.SAASBO(experiment=experiment, data=experiment.lookup_data())
        mock_mcmc.assert_called_once()
        kwargs = mock_mcmc.call_args.kwargs
        self.assertEqual(kwargs["num_samples"], 16)
        self.assertEqual(kwargs["warmup_steps"], 0)

    def test_mixed_optimizer_mocks(self) -> None:
        experiment = get_branin_experiment(
            with_completed_batch=True, with_choice_parameter=True
        )
        with patch(
            "botorch.optim.optimize_mixed.generate_starting_points",
            wraps=generate_starting_points,
        ) as mock_gen:
            with mock_botorch_optimize_context_manager():
                Generators.BOTORCH_MODULAR(
                    experiment=experiment,
                    data=experiment.lookup_data(),
                    transforms=[OrderedChoiceToIntegerRange],
                ).gen(n=1)
        mock_gen.assert_called_once()
        opt_inputs = mock_gen.call_args.kwargs["opt_inputs"]
        self.assertEqual(opt_inputs.raw_samples, 2)
        self.assertEqual(opt_inputs.num_restarts, 1)
        self.assertEqual(
            opt_inputs.options,
            {
                "init_batch_limit": INIT_BATCH_LIMIT,
                "batch_limit": BATCH_LIMIT,
                "maxiter_alternating": 1,
                "maxiter_continuous": 1,
                "maxiter_init": 1,
                "maxiter_discrete": 1,
            },
        )

    @skip_if_import_error
    def test_optimize_with_nsgaii_mocks(self) -> None:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize

        experiment = get_branin_experiment(with_completed_batch=True)
        with (
            patch(
                "botorch.utils.multi_objective.optimize.minimize",
                wraps=minimize,
            ) as mock_minimize,
            patch(
                "botorch.utils.multi_objective.optimize.NSGA2",
                wraps=NSGA2,
            ) as mock_nsgaii,
        ):
            with mock_botorch_optimize_context_manager():
                Generators.BOTORCH_MODULAR(
                    experiment=experiment,
                    data=experiment.lookup_data(),
                    botorch_acqf_classes_with_options=[
                        (PosteriorMean, {}),
                        (ExpectedImprovement, {}),
                    ],
                ).gen(n=1)
        mock_minimize.assert_called_once()
        mock_nsgaii.assert_called_once()
        self.assertEqual(mock_nsgaii.call_args.kwargs["pop_size"], 10)
        self.assertEqual(mock_minimize.call_args.kwargs["termination"].n_max_gen, 1)
