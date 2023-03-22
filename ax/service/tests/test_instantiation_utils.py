#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from ax.core.metric import Metric
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.parameter import FixedParameter, ParameterType, RangeParameter
from ax.core.search_space import HierarchicalSearchSpace
from ax.exceptions.core import UnsupportedError
from ax.service.utils.instantiation import InstantiationBase
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast


class TestInstantiationtUtils(TestCase):
    """Testing the instantiation utilities functionality that is not tested in
    main `AxClient` testing suite (`TestServiceAPI`)."""

    def test_parameter_type_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "No AE parameter type"):
            # pyre-fixme[6]: For 1st param expected `Union[Type[bool], Type[float],
            #  Type[int], Type[str]]` but got `Type[list]`.
            InstantiationBase._get_parameter_type(list)

    def test_constraint_from_str(self) -> None:
        with self.assertRaisesRegex(ValueError, "Bound for the constraint"):
            InstantiationBase.constraint_from_str(
                "x1 + x2 <= not_numerical_bound",
                # pyre-fixme[6]: For 2nd param expected `Dict[str, Parameter]` but
                #  got `Dict[str, None]`.
                {"x1": None, "x2": None},
            )
        with self.assertRaisesRegex(ValueError, "Outcome constraint bound"):
            InstantiationBase.outcome_constraint_from_str("m1 <= not_numerical_bound")
        three_val_constaint = InstantiationBase.constraint_from_str(
            "x1 + x2 + x3 <= 3",
            {
                "x1": RangeParameter(
                    name="x1", parameter_type=ParameterType.FLOAT, lower=0.1, upper=2.0
                ),
                "x2": RangeParameter(
                    name="x2", parameter_type=ParameterType.FLOAT, lower=0.1, upper=2.0
                ),
                "x3": RangeParameter(
                    name="x3", parameter_type=ParameterType.FLOAT, lower=0.1, upper=2.0
                ),
            },
        )

        self.assertEqual(three_val_constaint.bound, 3.0)
        with self.assertRaisesRegex(ValueError, "Parameter constraint should"):
            InstantiationBase.constraint_from_str(
                "x1 + x2 + <= 3",
                # pyre-fixme[6]: For 2nd param expected `Dict[str, Parameter]` but
                #  got `Dict[str, None]`.
                {"x1": None, "x2": None, "x3": None},
            )
        with self.assertRaisesRegex(ValueError, "Parameter constraint should"):
            InstantiationBase.constraint_from_str(
                "x1 + x2 + x3 = 3",
                # pyre-fixme[6]: For 2nd param expected `Dict[str, Parameter]` but
                #  got `Dict[str, None]`.
                {"x1": None, "x2": None, "x3": None},
            )
        one_val_constraint = InstantiationBase.constraint_from_str(
            "x1 <= 0",
            # pyre-fixme[6]: For 2nd param expected `Dict[str, Parameter]` but
            #  got `Dict[str, None]`.
            {"x1": None, "x2": None},
        )
        self.assertEqual(one_val_constraint.bound, 0.0)
        self.assertEqual(one_val_constraint.constraint_dict, {"x1": 1.0})
        one_val_constraint = InstantiationBase.constraint_from_str(
            "-0.5*x1 >= -0.1",
            # pyre-fixme[6]: For 2nd param expected `Dict[str, Parameter]` but
            #  got `Dict[str, None]`.
            {"x1": None, "x2": None},
        )
        self.assertEqual(one_val_constraint.bound, 0.1)
        self.assertEqual(one_val_constraint.constraint_dict, {"x1": 0.5})
        three_val_constaint2 = InstantiationBase.constraint_from_str(
            "-x1 + 2.1*x2 - 4*x3 <= 3",
            {
                "x1": RangeParameter(
                    name="x1", parameter_type=ParameterType.FLOAT, lower=0.1, upper=4.0
                ),
                "x2": RangeParameter(
                    name="x2", parameter_type=ParameterType.FLOAT, lower=0.1, upper=4.0
                ),
                "x3": RangeParameter(
                    name="x3", parameter_type=ParameterType.FLOAT, lower=0.1, upper=4.0
                ),
            },
        )

        self.assertEqual(three_val_constaint2.bound, 3.0)
        self.assertEqual(
            three_val_constaint2.constraint_dict, {"x1": -1.0, "x2": 2.1, "x3": -4.0}
        )
        with self.assertRaisesRegex(ValueError, "Multiplier should be float"):
            InstantiationBase.constraint_from_str(
                "x1 - e*x2 + x3 <= 3",
                # pyre-fixme[6]: For 2nd param expected `Dict[str, Parameter]` but
                #  got `Dict[str, None]`.
                {"x1": None, "x2": None, "x3": None},
            )
        with self.assertRaisesRegex(ValueError, "A linear constraint should be"):
            InstantiationBase.constraint_from_str(
                "x1 - 2 *x2 + 3 *x3 <= 3",
                # pyre-fixme[6]: For 2nd param expected `Dict[str, Parameter]` but
                #  got `Dict[str, None]`.
                {"x1": None, "x2": None, "x3": None},
            )
        with self.assertRaisesRegex(ValueError, "A linear constraint should be"):
            InstantiationBase.constraint_from_str(
                "x1 - 2* x2 + 3* x3 <= 3",
                # pyre-fixme[6]: For 2nd param expected `Dict[str, Parameter]` but
                #  got `Dict[str, None]`.
                {"x1": None, "x2": None, "x3": None},
            )
        with self.assertRaisesRegex(ValueError, "A linear constraint should be"):
            InstantiationBase.constraint_from_str(
                "x1 - 2 * x2 + 3*x3 <= 3",
                # pyre-fixme[6]: For 2nd param expected `Dict[str, Parameter]` but
                #  got `Dict[str, None]`.
                {"x1": None, "x2": None, "x3": None},
            )

    def test_objective_validation(self) -> None:
        with self.assertRaisesRegex(UnsupportedError, "Ambiguous objective definition"):
            InstantiationBase.make_experiment(
                # pyre-fixme[6]: For 1st param expected `List[Dict[str, Union[None,
                #  Dict[str, List[str]], List[Union[None, bool, float, int, str]],
                #  bool, float, int, str]]]` but got `Dict[str, Union[List[int],
                #  str]]`.
                parameters={"name": "x", "type": "range", "bounds": [0, 1]},
                objective_name="branin",
                objectives={"branin": "minimize", "currin": "maximize"},
            )

    def test_add_tracking_metrics(self) -> None:
        experiment = InstantiationBase.make_experiment(
            parameters=[{"name": "x", "type": "range", "bounds": [0, 1]}],
            tracking_metric_names=None,
        )
        self.assertDictEqual(experiment._tracking_metrics, {})

        metrics_names = ["metric_1", "metric_2"]
        experiment = InstantiationBase.make_experiment(
            parameters=[{"name": "x", "type": "range", "bounds": [0, 1]}],
            tracking_metric_names=metrics_names,
        )
        self.assertDictEqual(
            experiment._tracking_metrics,
            {metric_name: Metric(name=metric_name) for metric_name in metrics_names},
        )

    def test_make_objectives(self) -> None:
        with self.assertRaisesRegex(ValueError, "specify 'minimize' or 'maximize'"):
            InstantiationBase.make_objectives({"branin": "unknown"})
        objectives = InstantiationBase.make_objectives(
            {"branin": "minimize", "currin": "maximize"}
        )
        branin_metric = [o.minimize for o in objectives if o.metric.name == "branin"]
        self.assertTrue(branin_metric[0])
        currin_metric = [o.minimize for o in objectives if o.metric.name == "currin"]
        self.assertFalse(currin_metric[0])

    def test_make_optimization_config(self) -> None:
        objectives = {"branin": "minimize", "currin": "maximize"}
        objective_thresholds = ["branin <= 0", "currin >= 0"]
        with self.subTest("Single-objective optimizations with objective thresholds"):
            with self.assertRaisesRegex(ValueError, "not specify objective thresholds"):
                InstantiationBase.make_optimization_config(
                    {"branin": "minimize"},
                    objective_thresholds,
                    outcome_constraints=[],
                    status_quo_defined=False,
                )

        with self.subTest("MOO with partial objective thresholds"):
            multi_optimization_config = InstantiationBase.make_optimization_config(
                objectives,
                objective_thresholds=objective_thresholds[:1],
                outcome_constraints=[],
                status_quo_defined=False,
            )
            self.assertEqual(len(multi_optimization_config.objective.metrics), 2)
            self.assertEqual(
                len(
                    checked_cast(
                        MultiObjectiveOptimizationConfig, multi_optimization_config
                    ).objective_thresholds
                ),
                1,
            )

        with self.subTest("MOO with all objective threshold"):
            multi_optimization_config = InstantiationBase.make_optimization_config(
                objectives,
                objective_thresholds,
                outcome_constraints=[],
                status_quo_defined=False,
            )
            self.assertEqual(len(multi_optimization_config.objective.metrics), 2)
            self.assertEqual(
                len(
                    checked_cast(
                        MultiObjectiveOptimizationConfig, multi_optimization_config
                    ).objective_thresholds
                ),
                2,
            )

        with self.subTest(
            "Single-objective optimizations without objective thresholds"
        ):
            single_optimization_config = InstantiationBase.make_optimization_config(
                {"branin": "minimize"},
                objective_thresholds=[],
                outcome_constraints=[],
                status_quo_defined=False,
            )
            self.assertEqual(single_optimization_config.objective.metric.name, "branin")

    def test_single_valued_choice_to_fixed_param_conversion(self) -> None:
        for use_dependents in [True, False]:
            representation: Dict[str, Any] = {
                "name": "test",
                "type": "choice",
                "values": [1.0],
            }
            if use_dependents:
                representation["dependents"] = {1.0: ["foo_or_bar", "bazz"]}
            output = checked_cast(
                FixedParameter, InstantiationBase.parameter_from_json(representation)
            )
            self.assertIsInstance(output, FixedParameter)
            self.assertEqual(output.value, 1.0)
            if use_dependents:
                self.assertEqual(output.dependents, {1.0: ["foo_or_bar", "bazz"]})

    def test_hss(self) -> None:
        parameter_dicts = [
            {
                "name": "root",
                "type": "fixed",
                "value": "HierarchicalSearchSpace",
                "dependents": {"HierarchicalSearchSpace": ["foo_or_bar", "bazz"]},
            },
            {
                "name": "foo_or_bar",
                "type": "choice",
                "values": ["Foo", "Bar"],
                "dependents": {"Foo": ["an_int"], "Bar": ["a_float"]},
            },
            {
                "name": "an_int",
                "type": "choice",
                "values": [1, 2, 3],
                "dependents": None,
            },
            {"name": "a_float", "type": "range", "bounds": [1.0, 1000.0]},
            {
                "name": "bazz",
                "type": "fixed",
                "value": "Bazz",
                "dependents": {"Bazz": ["another_int"]},
            },
            {"name": "another_int", "type": "fixed", "value": "2"},
        ]
        search_space = InstantiationBase.make_search_space(
            # pyre-fixme[6]: For 1st param expected `List[Dict[str, Union[None, Dict[...
            parameters=parameter_dicts,
            parameter_constraints=[],
        )
        self.assertIsInstance(search_space, HierarchicalSearchSpace)
        # pyre-fixme[16]: `SearchSpace` has no attribute `_root`.
        self.assertEqual(search_space._root.name, "root")
