#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict

from ax.core import Experiment
from ax.core.experiment_design import EXPERIMENT_DESIGN_KEY, ExperimentDesign
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_search_space


class ExperimentDesignTest(TestCase):
    """Tests covering ExperimentDesign class and its usage in ax Experiment"""

    def test_experiment_design_property(self) -> None:
        """Test that Experiment.design property returns ExperimentDesign instance."""
        experiment = Experiment(
            name="test",
            search_space=get_branin_search_space(),
        )
        self.assertIsInstance(experiment.design, ExperimentDesign)
        self.assertIsNone(experiment.design.concurrency_limit)

        properties: Dict[str, Any] = {EXPERIMENT_DESIGN_KEY: {"concurrency_limit": 42}}
        experiment = Experiment(
            name="test", search_space=get_branin_search_space(), properties=properties
        )
        self.assertEqual(experiment.design.concurrency_limit, 42)
