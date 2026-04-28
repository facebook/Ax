# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import MagicMock, PropertyMock

from ax.adapter.transfer_learning.adapter import TransferLearningAdapter
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace, SearchSpaceDigest
from ax.exceptions.core import UnsupportedError
from ax.utils.common.testutils import TestCase


class ExpandSsdToJointSpaceTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.adapter = MagicMock(spec=TransferLearningAdapter)

    def _make_joint_ss(self, params: dict[str, tuple[float, float]]) -> SearchSpace:
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

    def test_no_extra_params_returns_unchanged(self) -> None:
        type(self.adapter).joint_search_space = PropertyMock(
            return_value=self._make_joint_ss({"x1": (0, 1), "x2": (0, 1)})
        )
        ssd = SearchSpaceDigest(
            feature_names=["x1", "x2", "task"],
            bounds=[(0, 1), (0, 1), (0, 2)],
            task_features=[2],
            target_values={2: 0},
        )
        result = TransferLearningAdapter._expand_ssd_to_joint_space(self.adapter, ssd)
        self.assertIs(result, ssd)

    def test_single_task_feature_inserts_before_task(self) -> None:
        type(self.adapter).joint_search_space = PropertyMock(
            return_value=self._make_joint_ss(
                {"x1": (0, 1), "x2": (0, 1), "x3": (-2, 5)}
            )
        )
        ssd = SearchSpaceDigest(
            feature_names=["x1", "x2", "task"],
            bounds=[(0, 1), (0, 1), (0, 2)],
            task_features=[2],
            target_values={2: 0},
        )
        result = TransferLearningAdapter._expand_ssd_to_joint_space(self.adapter, ssd)
        self.assertEqual(result.feature_names, ["x1", "x2", "x3", "task"])
        self.assertEqual(result.bounds, [(0, 1), (0, 1), (-2, 5), (0, 2)])
        self.assertEqual(result.task_features, [3])
        self.assertEqual(result.target_values, {3: 0})

    def test_zero_task_features_appends(self) -> None:
        type(self.adapter).joint_search_space = PropertyMock(
            return_value=self._make_joint_ss({"x1": (0, 1), "x2": (-1, 3)})
        )
        ssd = SearchSpaceDigest(
            feature_names=["x1"],
            bounds=[(0, 1)],
        )
        result = TransferLearningAdapter._expand_ssd_to_joint_space(self.adapter, ssd)
        self.assertEqual(result.feature_names, ["x1", "x2"])
        self.assertEqual(result.bounds, [(0, 1), (-1, 3)])

    def test_discrete_choices_on_task_feature_shifted(self) -> None:
        type(self.adapter).joint_search_space = PropertyMock(
            return_value=self._make_joint_ss({"x1": (0, 1), "x2": (0, 1), "x3": (0, 1)})
        )
        ssd = SearchSpaceDigest(
            feature_names=["x1", "x2", "task"],
            bounds=[(0, 1), (0, 1), (0, 2)],
            task_features=[2],
            target_values={2: 0},
            discrete_choices={2: [0, 1, 2]},
        )
        result = TransferLearningAdapter._expand_ssd_to_joint_space(self.adapter, ssd)
        self.assertEqual(result.discrete_choices, {3: [0, 1, 2]})
        self.assertEqual(result.task_features, [3])

    def test_hierarchical_dependencies_at_task_idx_raises(self) -> None:
        type(self.adapter).joint_search_space = PropertyMock(
            return_value=self._make_joint_ss({"x1": (0, 1), "x2": (0, 1), "x3": (0, 1)})
        )
        ssd = SearchSpaceDigest(
            feature_names=["x1", "x2", "task"],
            bounds=[(0, 1), (0, 1), (0, 2)],
            task_features=[2],
            target_values={2: 0},
            hierarchical_dependencies={2: {0: [1]}},
        )
        with self.assertRaisesRegex(UnsupportedError, "hierarchical_dependencies"):
            TransferLearningAdapter._expand_ssd_to_joint_space(self.adapter, ssd)

    def test_multiple_task_features_raises(self) -> None:
        type(self.adapter).joint_search_space = PropertyMock(
            return_value=self._make_joint_ss({"x1": (0, 1), "x2": (0, 1), "x3": (0, 1)})
        )
        ssd = SearchSpaceDigest(
            feature_names=["x1", "task1", "task2"],
            bounds=[(0, 1), (0, 1), (0, 1)],
            task_features=[1, 2],
        )
        with self.assertRaisesRegex(UnsupportedError, "Multiple task features"):
            TransferLearningAdapter._expand_ssd_to_joint_space(self.adapter, ssd)
