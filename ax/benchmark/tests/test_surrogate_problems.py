# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.problems.surrogate import (
    MOOSurrogateBenchmarkProblem,
    SOOSurrogateBenchmarkProblem,
)
from ax.core.runner import Runner
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_multi_objective_optimization_config,
    get_optimization_config,
    get_search_space,
)
from botorch.models.gp_regression import SingleTaskGP


class TestSurrogateProblems(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
        )
        self.datasets = []

    def test_lazy_instantiation(self) -> None:

        # test instantiation from init
        sbp = SOOSurrogateBenchmarkProblem(
            name="test",
            search_space=get_search_space(),
            optimization_config=get_optimization_config(),
            num_trials=10,
            infer_noise=False,
            metric_names=[],
            get_surrogate_and_datasets=lambda: (self.surrogate, self.datasets),
            optimal_value=0.0,
        )

        self.assertIsNone(sbp._runner)
        # sets runner
        self.assertIsInstance(sbp.runner, Runner)

        self.assertIsNotNone(sbp._runner)
        self.assertIsNotNone(sbp.runner)

        # repeat for MOO
        sbp = MOOSurrogateBenchmarkProblem(
            name="test",
            search_space=get_search_space(),
            optimization_config=get_multi_objective_optimization_config(),
            num_trials=10,
            infer_noise=False,
            metric_names=[],
            get_surrogate_and_datasets=lambda: (self.surrogate, self.datasets),
            maximum_hypervolume=1.0,
            reference_point=[],
        )
        self.assertIsNone(sbp._runner)
        # sets runner
        self.assertIsInstance(sbp.runner, Runner)

        self.assertIsNotNone(sbp._runner)
        self.assertIsNotNone(sbp.runner)
