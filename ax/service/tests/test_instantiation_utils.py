#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.metric import Metric
from ax.core.parameter import ParameterType, RangeParameter
from ax.exceptions.core import UnsupportedError
from ax.service.utils.instantiation import (
    _get_parameter_type,
    constraint_from_str,
    outcome_constraint_from_str,
    make_experiment,
    make_objectives,
    make_optimization_config,
    raw_data_to_evaluation,
)
from ax.utils.common.testutils import TestCase


class TestInstantiationtUtils(TestCase):
    """Testing the instantiation utilities functionality that is not tested in
    main `AxClient` testing suite (`TestServiceAPI`)."""

    def test_parameter_type_validation(self):
        with self.assertRaisesRegex(ValueError, "No AE parameter type"):
            _get_parameter_type(list)

    def test_constraint_from_str(self):
        with self.assertRaisesRegex(ValueError, "Bound for the constraint"):
            constraint_from_str(
                "x1 + x2 <= not_numerical_bound", {"x1": None, "x2": None}
            )
        with self.assertRaisesRegex(ValueError, "Outcome constraint bound"):
            outcome_constraint_from_str("m1 <= not_numerical_bound")
        three_val_constaint = constraint_from_str(
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
            constraint_from_str("x1 + x2 + <= 3", {"x1": None, "x2": None, "x3": None})
        with self.assertRaisesRegex(ValueError, "Parameter constraint should"):
            constraint_from_str(
                "x1 + x2 + x3 = 3", {"x1": None, "x2": None, "x3": None}
            )
        three_val_constaint2 = constraint_from_str(
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
            constraint_from_str(
                "x1 - e*x2 + x3 <= 3", {"x1": None, "x2": None, "x3": None}
            )
        with self.assertRaisesRegex(ValueError, "A linear constraint should be"):
            constraint_from_str(
                "x1 - 2 *x2 + 3 *x3 <= 3", {"x1": None, "x2": None, "x3": None}
            )
        with self.assertRaisesRegex(ValueError, "A linear constraint should be"):
            constraint_from_str(
                "x1 - 2* x2 + 3* x3 <= 3", {"x1": None, "x2": None, "x3": None}
            )
        with self.assertRaisesRegex(ValueError, "A linear constraint should be"):
            constraint_from_str(
                "x1 - 2 * x2 + 3*x3 <= 3", {"x1": None, "x2": None, "x3": None}
            )

    def test_objective_validation(self):
        with self.assertRaisesRegex(UnsupportedError, "Ambiguous objective definition"):
            make_experiment(
                parameters={"name": "x", "type": "range", "bounds": [0, 1]},
                objective_name="branin",
                objectives={"branin": "minimize", "currin": "maximize"},
            )

    def test_add_tracking_metrics(self):
        experiment = make_experiment(
            parameters=[{"name": "x", "type": "range", "bounds": [0, 1]}],
            tracking_metric_names=None,
        )
        self.assertDictEqual(experiment._tracking_metrics, {})

        metrics_names = ["metric_1", "metric_2"]
        experiment = make_experiment(
            parameters=[{"name": "x", "type": "range", "bounds": [0, 1]}],
            tracking_metric_names=metrics_names,
        )
        self.assertDictEqual(
            experiment._tracking_metrics,
            {metric_name: Metric(name=metric_name) for metric_name in metrics_names},
        )

    def test_make_objectives(self):
        with self.assertRaisesRegex(ValueError, "specify 'minimize' or 'maximize'"):
            make_objectives({"branin": "unknown"})
        objectives = make_objectives({"branin": "minimize", "currin": "maximize"})
        branin_metric = [o.minimize for o in objectives if o.metric.name == "branin"]
        self.assertTrue(branin_metric[0])
        currin_metric = [o.minimize for o in objectives if o.metric.name == "currin"]
        self.assertFalse(currin_metric[0])

    def test_make_optimization_config(self):
        objectives = {"branin": "minimize", "currin": "maximize"}
        objective_thresholds = ["branin <= 0", "currin >= 0"]
        with self.subTest("Single-objective optimizations with objective thresholds"):
            with self.assertRaisesRegex(ValueError, "not specify objective thresholds"):
                make_optimization_config(
                    {"branin": "minimize"},
                    objective_thresholds,
                    outcome_constraints=[],
                    status_quo_defined=False,
                )

        with self.subTest("MOO missing objective thresholds"):
            with self.assertLogs(
                "ax.service.utils.instantiation", level="INFO"
            ) as logs:
                multi_optimization_config = make_optimization_config(
                    objectives,
                    objective_thresholds=objective_thresholds[:1],
                    outcome_constraints=[],
                    status_quo_defined=False,
                )
                self.assertTrue(
                    any(
                        "Due to non-specification" in output and "currin" in output
                        for output in logs.output
                    ),
                    logs.output,
                )
                self.assertEqual(len(multi_optimization_config.objective.metrics), 2)
                self.assertEqual(len(multi_optimization_config.objective_thresholds), 1)

        with self.subTest("MOO with all objective threshold"):
            multi_optimization_config = make_optimization_config(
                objectives,
                objective_thresholds,
                outcome_constraints=[],
                status_quo_defined=False,
            )
            self.assertEqual(len(multi_optimization_config.objective.metrics), 2)
            self.assertEqual(len(multi_optimization_config.objective_thresholds), 2)

        with self.subTest(
            "Single-objective optimizations without objective thresholds"
        ):
            single_optimization_config = make_optimization_config(
                {"branin": "minimize"},
                objective_thresholds=[],
                outcome_constraints=[],
                status_quo_defined=False,
            )
            self.assertEqual(single_optimization_config.objective.metric.name, "branin")


class TestRawDataToEvaluation(TestCase):
    def test_raw_data_is_not_dict_of_dicts(self):
        with self.assertRaises(ValueError):
            raw_data_to_evaluation(
                raw_data={"arm_0": {"objective_a": 6}},
                metric_names=["objective_a"],
            )

    def test_it_converts_to_floats_in_dict_and_leaves_tuples(self):
        result = raw_data_to_evaluation(
            raw_data={
                "objective_a": 6,
                "objective_b": 1.0,
                "objective_c": ("some", "tuple"),
            },
            metric_names=["objective_a", "objective_b"],
        )
        self.assertEqual(result["objective_a"], (6.0, None))
        self.assertEqual(result["objective_b"], (1.0, None))
        self.assertEqual(result["objective_c"], ("some", "tuple"))

    def test_dict_entries_must_be_int_float_or_tuple(self):
        with self.assertRaises(ValueError):
            raw_data_to_evaluation(
                raw_data={"objective_a": [6.0, None]},
                metric_names=["objective_a"],
            )

    def test_it_requires_a_dict_for_multi_objectives(self):
        with self.assertRaises(ValueError):
            raw_data_to_evaluation(
                raw_data=(6.0, None),
                metric_names=["objective_a", "objective_b"],
            )

    def test_it_accepts_a_list_for_single_objectives(self):
        raw_data = [({"arm__0": {}}, {"objective_a": (1.4, None)})]
        result = raw_data_to_evaluation(
            raw_data=raw_data,
            metric_names=["objective_a"],
        )
        self.assertEqual(raw_data, result)

    def test_it_turns_a_tuple_into_a_dict(self):
        raw_data = (1.4, None)
        result = raw_data_to_evaluation(
            raw_data=raw_data,
            metric_names=["objective_a"],
        )
        self.assertEqual(result["objective_a"], raw_data)

    def test_it_turns_an_int_into_a_dict_of_tuple(self):
        result = raw_data_to_evaluation(
            raw_data=1,
            metric_names=["objective_a"],
        )
        self.assertEqual(result["objective_a"], (1.0, None))

    def test_it_turns_a_float_into_a_dict_of_tuple(self):
        result = raw_data_to_evaluation(
            raw_data=1.6,
            metric_names=["objective_a"],
        )
        self.assertEqual(result["objective_a"], (1.6, None))

    def test_it_raises_for_unexpected_types(self):
        with self.assertRaises(ValueError):
            raw_data_to_evaluation(
                raw_data="1.6",
                metric_names=["objective_a"],
            )
