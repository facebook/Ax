# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from ax.adapter.transfer_learning.adapter import TL_EXP, TransferLearningAdapter
from ax.adapter.transforms.metadata_to_task import MetadataToTask
from ax.core.arm import Arm
from ax.core.auxiliary_source import AuxiliarySource
from ax.core.experiment import Experiment
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from pyre_extensions import assert_is_instance


def _make_ss(params: dict[str, tuple[float, float]]) -> SearchSpace:
    return SearchSpace(
        parameters=[
            RangeParameter(
                name=n,
                lower=lo,
                upper=hi,
                parameter_type=ParameterType.FLOAT,
            )
            for n, (lo, hi) in params.items()
        ]
    )


def _gen_experiment(
    experiment_name: str,
    num_trials: int,
    search_space: SearchSpace | None = None,
) -> Experiment:
    exp = get_experiment_with_observations(
        observations=torch.rand(num_trials, 1).tolist(),
        search_space=search_space,
    )
    exp.name = experiment_name
    return exp


class SetSearchSpaceTest(TestCase):
    """_set_search_space adds source-only params to _model_space while
    preserving target bounds for shared params."""

    def test_model_space_has_source_only_params(self) -> None:
        target_ss = _make_ss({"x": (0, 1), "y": (0, 1)})
        source_ss = _make_ss({"x": (0, 5), "y": (0, 5), "z": (0, 5)})
        target_exp = _gen_experiment("target", num_trials=3, search_space=target_ss)
        source_exp = _gen_experiment("source", num_trials=3, search_space=source_ss)
        source_exp.status_quo = Arm(parameters={"x": 1.0, "y": 1.0, "z": 2.5})
        target_exp.auxiliary_experiments_by_purpose[TL_EXP] = [
            AuxiliarySource(experiment=source_exp)
        ]
        adapter = TransferLearningAdapter(
            experiment=target_exp,
            search_space=target_ss,
            data=target_exp.lookup_data(),
            generator=BoTorchGenerator(),
            transforms=[MetadataToTask],
            fit_on_init=False,
        )
        with self.subTest("model_space_has_z"):
            self.assertIn("z", adapter._model_space.parameters)
        with self.subTest("search_space_unchanged"):
            self.assertNotIn("z", adapter._search_space.parameters)
        with self.subTest("backfilled_not_source_only"):
            self.assertNotIn("z", adapter._source_only_params)
        with self.subTest("shared_params_keep_target_bounds"):
            x_param = assert_is_instance(
                adapter._model_space.parameters["x"], RangeParameter
            )
            self.assertEqual(x_param.lower, 0.0)
            self.assertEqual(x_param.upper, 1.0)
        with self.subTest("source_only_without_backfill"):
            source_ss2 = _make_ss({"x": (0, 5), "w": (0, 10)})
            source_exp2 = _gen_experiment(
                "source2", num_trials=3, search_space=source_ss2
            )
            target_exp.auxiliary_experiments_by_purpose[TL_EXP] = [
                AuxiliarySource(experiment=source_exp2)
            ]
            adapter2 = TransferLearningAdapter(
                experiment=target_exp,
                search_space=target_ss,
                data=target_exp.lookup_data(),
                generator=BoTorchGenerator(),
                transforms=[MetadataToTask],
                fit_on_init=False,
            )
            self.assertIn("w", adapter2._model_space.parameters)
            self.assertIsInstance(adapter2._model_space.parameters["w"], RangeParameter)


class GetTargetDataParametersTest(TestCase):
    """_get_target_data_parameters filters joint params to target-only + task."""

    def test_filters_source_only_params(self) -> None:
        adapter = MagicMock(spec=TransferLearningAdapter)
        adapter._source_only_params = {"z"}
        joint_params = ["x", "y", "z", Keys.TASK_FEATURE_NAME.value]
        result = TransferLearningAdapter._get_target_data_parameters(
            adapter, joint_params
        )
        self.assertEqual(result, ["x", "y", Keys.TASK_FEATURE_NAME.value])

    def test_no_source_only_params_returns_all(self) -> None:
        adapter = MagicMock(spec=TransferLearningAdapter)
        adapter._source_only_params = set()
        params = ["x", "y", Keys.TASK_FEATURE_NAME.value]
        result = TransferLearningAdapter._get_target_data_parameters(adapter, params)
        self.assertEqual(result, params)
