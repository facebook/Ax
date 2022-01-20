# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_data, get_branin_experiment
from ax.utils.testing.mock import mock_mbo, ModelsMockingError
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models import SingleTaskGP


class MockTest(TestCase):
    def test_mock_mbo(self):
        n = 10

        experiment = get_branin_experiment(with_trial=True)
        data = get_branin_data(trials=[experiment.trials[0]])

        mbo_gpei = Models.BOTORCH_MODULAR(
            experiment=experiment,
            data=data,
            surrogate=Surrogate(SingleTaskGP),
            botorch_acqf_class=qNoisyExpectedImprovement,
        )
        with mock_mbo():
            generator_run = mbo_gpei.gen(n)
            self.assertIsNotNone(generator_run.arms)

        sobol = Models.SOBOL(search_space=experiment.search_space)
        with self.assertRaises(ModelsMockingError):
            with mock_mbo():
                generator_run = sobol.gen(n)
