# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.adapter.transfer_learning.utils import get_joint_search_space
from ax.adapter.transfer_learning.utils_torch import get_mapped_parameter_names
from ax.adapter.transforms.one_hot import OneHot
from ax.adapter.transforms.remove_fixed import RemoveFixed
from ax.core.auxiliary_source import AuxiliarySource
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment, get_branin_search_space


class TestUtilsTorch(TestCase):
    def setUp(self) -> None:
        super().setUp()
        base_params = list(get_branin_search_space().parameters.values())
        fp1 = FixedParameter(name="fp1", parameter_type=ParameterType.STRING, value="a")
        fp2 = FixedParameter(name="fp2", parameter_type=ParameterType.STRING, value="b")
        x3 = RangeParameter(
            name="x3", parameter_type=ParameterType.FLOAT, lower=0, upper=1
        )
        rp1 = RangeParameter(
            name="rp1", parameter_type=ParameterType.FLOAT, lower=-10, upper=20
        )
        cp1 = ChoiceParameter(
            name="cp1",
            parameter_type=ParameterType.STRING,
            values=["a", "b", "c"],
            is_ordered=False,
        )
        self.target_ss = SearchSpace(base_params + [x3, fp1])
        source_ss = SearchSpace(base_params + [rp1, fp2])
        source_ss2 = SearchSpace(base_params + [x3, fp1, rp1, cp1])
        transfer_param_config = {"x3": "rp1"}

        source_exp1 = get_branin_experiment(
            with_completed_trial=True, search_space=self.target_ss.clone()
        )
        source_exp2 = get_branin_experiment(
            with_completed_trial=True, search_space=source_ss
        )
        source_exp3 = get_branin_experiment(
            with_completed_trial=True, search_space=source_ss2
        )

        self.auxsrc1 = AuxiliarySource(
            experiment=source_exp1, update_fixed_params=False
        )
        self.auxsrc2 = AuxiliarySource(
            experiment=source_exp2, transfer_param_config=transfer_param_config
        )
        self.auxsrc3 = AuxiliarySource(
            experiment=source_exp2,
            transfer_param_config=transfer_param_config,
            update_fixed_params=False,
        )
        self.auxsrc4 = AuxiliarySource(experiment=source_exp3)

    def test_mapped_parameter_names(self) -> None:
        # Auxsrc1 has 4 params that should all get returned.
        # The search space is same as the target.
        mapped_names = get_mapped_parameter_names(
            self.auxsrc1, target_search_space=self.target_ss
        )
        self.assertEqual(mapped_names, ["x1", "x2", "x3", "fp1"])
        # Auxsrc2 has 4 params. The search space is different from the target.
        # The fixed param fp2 will be replaced with fp1. rp1 will be mapped to x3.
        mapped_names = get_mapped_parameter_names(
            self.auxsrc2, target_search_space=self.target_ss
        )
        self.assertEqual(mapped_names, ["x1", "x2", "x3", "fp1"])
        # This is same search space as auxsrc2 but fixed param should not change.
        # rp1 will be mapped to x3.
        mapped_names = get_mapped_parameter_names(
            self.auxsrc3, target_search_space=self.target_ss
        )
        self.assertEqual(mapped_names, ["x1", "x2", "fp2", "x3"])
        # Auxsrc4 has 6 params. No change expected.
        mapped_names = get_mapped_parameter_names(
            self.auxsrc4, target_search_space=self.target_ss
        )
        self.assertEqual(mapped_names, ["x1", "x2", "x3", "rp1", "cp1", "fp1"])
        # With OneHot, cp1 will convert to 3 parameters. RemoveFixed will remove fp1.
        joint_ss = get_joint_search_space(
            search_space=self.target_ss,
            auxiliary_sources=[self.auxsrc4],
        )
        mapped_names = get_mapped_parameter_names(
            self.auxsrc4,
            target_search_space=self.target_ss,
            transforms={  # pyre-ignore[6]
                "OneHot": OneHot(search_space=joint_ss),
                "RemoveFixed": RemoveFixed(search_space=joint_ss),
            },
        )
        self.assertEqual(
            mapped_names,
            [
                "x1",
                "x2",
                "x3",
                "rp1",
                "cp1_OH_PARAM_0",
                "cp1_OH_PARAM_1",
                "cp1_OH_PARAM_2",
            ],
        )
