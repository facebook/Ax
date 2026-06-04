# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.adapter.transfer_learning.utils import (
    get_joint_search_space,
    merge_dependents,
    merge_parameters,
)
from ax.core.auxiliary_source import AuxiliarySource
from ax.core.experiment import Experiment
from ax.core.parameter import (
    ChoiceParameter,
    DerivedParameter,
    FixedParameter,
    Parameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.utils.common.testutils import TestCase
from pyre_extensions import assert_is_instance, none_throws


class AxFbCoreUtilsTest(TestCase):
    def test_get_joint_search_space(self) -> None:
        parameters: list[Parameter] = [
            RangeParameter(f"x{i}", parameter_type=ParameterType.INT, lower=0, upper=5)
            for i in range(3)
        ]
        exp1 = Experiment(
            search_space=SearchSpace(parameters=parameters[:2]), name="test1"
        )
        exp2 = Experiment(
            search_space=SearchSpace(parameters=parameters[:2]), name="test2"
        )
        exp3 = Experiment(
            search_space=SearchSpace(parameters=parameters[1:]), name="test3"
        )
        aux_2 = AuxiliarySource(experiment=exp2)
        aux_3 = AuxiliarySource(experiment=exp3)
        aux_4 = AuxiliarySource(experiment=exp3, transfer_param_config={"x0": "x2"})
        for exp, aux_srcs, expected_params in (
            (exp1, [aux_2], {"x0", "x1"}),
            (exp1, [aux_2, aux_3], {"x0", "x1", "x2"}),
            (exp1, [aux_2, aux_4], {"x0", "x1"}),
        ):
            self.assertEqual(
                set(
                    get_joint_search_space(
                        search_space=exp.search_space, auxiliary_sources=aux_srcs
                    ).parameters.keys()
                ),
                expected_params,
            )

    def test_get_joint_search_space_update_fixed_params(self) -> None:
        # test update fixed params
        range_param = RangeParameter(
            "x", parameter_type=ParameterType.INT, lower=0, upper=5
        )
        fixed_param1 = FixedParameter("y", parameter_type=ParameterType.INT, value=1)
        fixed_param2 = FixedParameter("y", parameter_type=ParameterType.INT, value=2)
        exp = Experiment(
            search_space=SearchSpace(parameters=[range_param, fixed_param1]),
            name="test1",
        )
        exp2 = Experiment(
            search_space=SearchSpace(parameters=[range_param, fixed_param2]),
            name="test2",
        )
        for update_fixed_params in [True, False]:
            aux2 = AuxiliarySource(
                experiment=exp2, update_fixed_params=update_fixed_params
            )
            ss_params = get_joint_search_space(
                search_space=exp.search_space, auxiliary_sources=[aux2]
            ).parameters
            self.assertEqual(
                assert_is_instance(ss_params["y"], FixedParameter).value, 1
            )
            self.assertIn("x", ss_params)

    def test_get_joint_search_space_with_hss_and_choice(self) -> None:
        ss1 = SearchSpace(
            parameters=[
                FixedParameter(
                    "root",
                    parameter_type=ParameterType.INT,
                    value=1,
                    dependents={1: ["learning_rate", "optimizer", "method"]},
                ),
                ChoiceParameter(
                    "learning_rate",
                    parameter_type=ParameterType.FLOAT,
                    values=[0.01, 0.05],
                ),
                ChoiceParameter(
                    "optimizer",
                    parameter_type=ParameterType.STRING,
                    values=["Adam", "SGD", "AdaGrad"],
                ),
                ChoiceParameter(
                    "method",
                    parameter_type=ParameterType.STRING,
                    values=["train", "eval"],
                ),
            ]
        )
        ss2 = SearchSpace(
            parameters=[
                FixedParameter(
                    "root2",
                    parameter_type=ParameterType.INT,
                    value=1,
                    dependents={1: ["lr", "optimizer"]},
                ),
                ChoiceParameter(
                    "lr", parameter_type=ParameterType.FLOAT, values=[0.01, 0.1]
                ),
                ChoiceParameter(
                    "optimizer",
                    parameter_type=ParameterType.STRING,
                    values=["Adam", "SGD"],
                ),
            ]
        )
        aux_src = AuxiliarySource(
            experiment=Experiment(search_space=ss2, name="test"),
            transfer_param_config={"learning_rate": "lr", "root": "root2"},
            update_fixed_params=False,
        )
        joint_ss = get_joint_search_space(search_space=ss1, auxiliary_sources=[aux_src])
        self.assertEqual(
            set(joint_ss.parameters.keys()),
            {"root", "learning_rate", "optimizer", "method"},
        )
        self.assertEqual(
            set(joint_ss["root"].dependents[1]),
            {"learning_rate", "optimizer", "method"},
        )
        self.assertEqual(
            assert_is_instance(
                joint_ss.parameters["learning_rate"], ChoiceParameter
            ).values,
            [0.01, 0.05, 0.1],
        )
        self.assertEqual(
            set(
                assert_is_instance(
                    joint_ss.parameters["optimizer"], ChoiceParameter
                ).values
            ),
            {"Adam", "SGD", "AdaGrad"},
        )

    def test_merge_dependents(self) -> None:
        p_no_dependents = FixedParameter(
            "p", parameter_type=ParameterType.BOOL, value=True
        )
        # No dependents returns None.
        self.assertIsNone(
            merge_dependents(
                p1=p_no_dependents, p2=p_no_dependents, reverse_param_config={}
            )
        )
        p_dependents_1 = FixedParameter(
            "p1", parameter_type=ParameterType.INT, value=1, dependents={1: ["q"]}
        )
        p_dependents_2 = FixedParameter(
            "p2", parameter_type=ParameterType.INT, value=1, dependents={1: ["z"]}
        )
        # p1 dependents do not get renamed.
        self.assertEqual(
            merge_dependents(
                p1=p_dependents_1, p2=p_no_dependents, reverse_param_config={"q": "w"}
            ),
            {1: ["q"]},
        )
        # p2 dependents get renamed.
        self.assertEqual(
            merge_dependents(
                p1=p_no_dependents, p2=p_dependents_1, reverse_param_config={"q": "w"}
            ),
            {1: ["w"]},
        )
        # Merge p1 & p2 dependents with renaming for p2 only.
        self.assertEqual(
            set(
                none_throws(
                    merge_dependents(
                        p1=p_dependents_1,
                        p2=p_dependents_2,
                        reverse_param_config={"q": "w", "z": "v"},
                    )
                )[1]
            ),
            {"q", "v"},
        )

    def test_merge_parameters(self) -> None:
        p_fixed = FixedParameter(
            name="fixed", parameter_type=ParameterType.BOOL, value=True
        )
        p_fixed_2 = FixedParameter(name="f2", parameter_type=ParameterType.INT, value=1)
        p_fixed_3 = FixedParameter(name="f3", parameter_type=ParameterType.INT, value=2)
        p_fixed_4 = FixedParameter(
            name="f4", parameter_type=ParameterType.INT, value=1, dependents={1: ["a"]}
        )
        with self.assertRaisesRegex(ValueError, "different names"):
            merge_parameters(p1=p_fixed, p2=p_fixed_2, reverse_param_config={})
        with self.assertRaisesRegex(ValueError, "different types"):
            merge_parameters(
                p1=p_fixed, p2=p_fixed_2, reverse_param_config={"f2": "fixed"}
            )
        # Check that it works with both values of update_fixed_params.
        for update_fixed_params in [True, False]:
            self.assertEqual(
                merge_parameters(
                    p1=p_fixed_2,
                    p2=p_fixed_3,
                    reverse_param_config={"f3": "f2"},
                    update_fixed_params=update_fixed_params,
                ),
                FixedParameter(
                    name="f2",
                    parameter_type=ParameterType.INT,
                    value=1,
                ),
            )
        self.assertEqual(
            merge_parameters(
                p1=p_fixed_2, p2=p_fixed_4, reverse_param_config={"f4": "f2"}
            ),
            FixedParameter(
                name="f2",
                parameter_type=ParameterType.INT,
                value=1,
                dependents={1: ["a"]},
            ),
        )
        p_range_1 = RangeParameter(
            name="p", parameter_type=ParameterType.INT, lower=1, upper=3
        )
        p_range_2 = RangeParameter(
            name="p", parameter_type=ParameterType.INT, lower=0, upper=2
        )
        self.assertEqual(
            merge_parameters(p1=p_range_1, p2=p_range_2, reverse_param_config={}),
            RangeParameter(
                name="p", parameter_type=ParameterType.INT, lower=0, upper=3
            ),
        )
        p_choice_1 = ChoiceParameter(
            name="p",
            parameter_type=ParameterType.STRING,
            values=["a", "b", "c"],
            dependents={"a": ["p1"], "c": ["p2"]},
        )
        p_choice_2 = ChoiceParameter(
            name="p", parameter_type=ParameterType.STRING, values=["a", "b", "d"]
        )
        self.assertEqual(
            merge_parameters(p1=p_choice_1, p2=p_choice_2, reverse_param_config={}),
            ChoiceParameter(
                name="p",
                parameter_type=ParameterType.STRING,
                values=["a", "b", "c", "d"],
                dependents={"a": ["p1"], "c": ["p2"]},
            ),
        )

        # FixedParameter + ChoiceParameter: fixed value already in choices.
        p_fixed_str = FixedParameter(
            name="p", parameter_type=ParameterType.STRING, value="a"
        )
        merged_fc = merge_parameters(
            p1=p_fixed_str, p2=p_choice_1, reverse_param_config={}
        )
        self.assertIsInstance(merged_fc, ChoiceParameter)
        merged_fc_choice = assert_is_instance(merged_fc, ChoiceParameter)
        self.assertEqual(set(merged_fc_choice.values), {"a", "b", "c"})
        # Dependents from the choice parameter are preserved.
        self.assertEqual(merged_fc_choice.dependents, {"a": ["p1"], "c": ["p2"]})

        # FixedParameter + ChoiceParameter: fixed value NOT in choices.
        p_fixed_str_new = FixedParameter(
            name="p", parameter_type=ParameterType.STRING, value="z"
        )
        merged_fc2 = merge_parameters(
            p1=p_fixed_str_new, p2=p_choice_1, reverse_param_config={}
        )
        self.assertEqual(
            set(assert_is_instance(merged_fc2, ChoiceParameter).values),
            {"a", "b", "c", "z"},
        )

        # Reversed order: ChoiceParameter as p1, FixedParameter as p2.
        merged_cf = merge_parameters(
            p1=p_choice_1, p2=p_fixed_str_new, reverse_param_config={}
        )
        self.assertEqual(
            set(assert_is_instance(merged_cf, ChoiceParameter).values),
            {"a", "b", "c", "z"},
        )

        # DerivedParameter: same expression succeeds.
        p_derived_1 = DerivedParameter(
            name="d",
            parameter_type=ParameterType.FLOAT,
            expression_str="0.5 * x + 0.3 * y",
        )
        p_derived_2 = DerivedParameter(
            name="d",
            parameter_type=ParameterType.FLOAT,
            expression_str="0.5 * x + 0.3 * y",
        )
        merged = merge_parameters(
            p1=p_derived_1, p2=p_derived_2, reverse_param_config={}
        )
        self.assertIsInstance(merged, DerivedParameter)
        self.assertEqual(
            assert_is_instance(merged, DerivedParameter).expression_str,
            "0.5 * x + 0.3 * y",
        )
        self.assertEqual(merged.name, "d")

        # DerivedParameter: different expressions raises ValueError.
        p_derived_3 = DerivedParameter(
            name="d",
            parameter_type=ParameterType.FLOAT,
            expression_str="0.7 * x + 0.1 * y",
        )
        with self.assertRaisesRegex(ValueError, "different expressions"):
            merge_parameters(p1=p_derived_1, p2=p_derived_3, reverse_param_config={})

        # DerivedParameter vs FixedParameter raises ValueError (type mismatch).
        p_fixed_float = FixedParameter(
            name="d", parameter_type=ParameterType.FLOAT, value=1.0
        )
        with self.assertRaisesRegex(ValueError, "different types"):
            merge_parameters(
                p1=p_derived_1,
                p2=p_fixed_float,
                reverse_param_config={},
            )
