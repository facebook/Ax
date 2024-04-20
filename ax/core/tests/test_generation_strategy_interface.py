# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional

from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.core.generator_run import GeneratorRun
from ax.exceptions.core import AxError, UnsupportedError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment, SpecialGenerationStrategy


class MyGSI(GenerationStrategyInterface):
    def gen_for_multiple_trials_with_multiple_models(
        self,
        experiment: Experiment,
        data: Optional[Data] = None,
        # TODO[drfreund, danielcohennyc, mgarrard]: Update the format of the arguments
        # below as we find the right one.
        num_generator_runs: int = 1,
        n: int = 1,
    ) -> List[List[GeneratorRun]]:
        raise NotImplementedError

    def clone_reset(self) -> "MyGSI":
        raise NotImplementedError


class TestGenerationStrategyInterface(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.exp = get_experiment()
        self.gsi = MyGSI(name="my_GSI")
        self.special_gsi = SpecialGenerationStrategy()

    def test_constructor(self) -> None:
        with self.assertRaisesRegex(TypeError, ".* abstract"):
            GenerationStrategyInterface(name="my_GSI")  # pyre-ignore[45]
        self.assertEqual(self.gsi.name, "my_GSI")

    def test_abstract(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.gsi.gen_for_multiple_trials_with_multiple_models(experiment=self.exp)

        with self.assertRaises(NotImplementedError):
            self.gsi.clone_reset()

    def test_experiment(self) -> None:
        with self.assertRaisesRegex(AxError, "No experiment"):
            self.gsi.experiment
        self.gsi.experiment = self.exp
        exp_2 = get_experiment()
        exp_2.name = "exp_2"
        with self.assertRaisesRegex(UnsupportedError, "has been used for"):
            self.gsi.experiment = exp_2
