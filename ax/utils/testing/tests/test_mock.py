#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

import torch
from ax.modelbridge.registry import Models
from ax.modelbridge.transforms.choice_encode import OrderedChoiceToIntegerRange
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import mock_botorch_optimize_context_manager
from botorch.generation.gen import gen_candidates_scipy
from botorch.optim.optimize_mixed import generate_starting_points
from botorch.utils.testing import MockAcquisitionFunction
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
                Models.SAASBO(experiment=experiment, data=experiment.lookup_data())
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
                Models.BOTORCH_MODULAR(
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
                "init_batch_limit": 32,
                "batch_limit": 5,
                "maxiter_alternating": 1,
                "maxiter_continuous": 1,
                "maxiter_init": 1,
                "maxiter_discrete": 1,
            },
        )
