#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from contextlib import ExitStack
from copy import deepcopy
from unittest import mock
from unittest.mock import Mock, patch

import numpy as np
from ax.adapter.base import Adapter
from ax.adapter.torch import TorchAdapter
from ax.adapter.transforms.derelativize import Derelativize, logger
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import ObservationData
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.exceptions.core import DataRequiredError
from ax.generators.base import Generator
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data,
    get_branin_experiment,
    get_experiment_with_observations,
)
from ax.utils.testing.mock import mock_botorch_optimize
from pyre_extensions import none_throws


class DerelativizeTransformTest(TestCase):
    def test_DerelativizeTransform(self) -> None:
        for negative_metrics in [False, True]:
            sq_sign = -1.0 if negative_metrics else 1.0
            observed_means = sq_sign * np.array([[1.0, 2.0], [1.0, 6.0]])
            sq_b_observed = np.mean(observed_means[1, 1])
            experiment = get_experiment_with_observations(
                observations=observed_means.tolist(),
                sems=[[1.0, 2.0], [1.0, 2.0]],
                parameterizations=[{"x": 2.0, "y": 10.0}, {"x": None, "y": None}],
            )

            predicted_means = sq_sign * np.array([3.0, 5.0])
            sq_b_predicted = predicted_means[-1]
            predict_return_value = [
                ObservationData(
                    means=predicted_means,
                    covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
                    metric_signatures=["m1", "m2"],
                )
            ]
            with ExitStack() as es:
                mock_predict = es.enter_context(
                    mock.patch(
                        "ax.adapter.base.Adapter._predict",
                        autospec=True,
                        return_value=predict_return_value,
                    )
                )
                self._test_DerelativizeTransform(
                    experiment=experiment,
                    mock_predict=mock_predict,
                    sq_b_observed=sq_b_observed,
                    sq_b_predicted=sq_b_predicted,
                )

    def _test_DerelativizeTransform(
        self,
        experiment: Experiment,
        mock_predict: Mock,
        sq_b_observed: float,
        sq_b_predicted: float,
    ) -> None:
        t = Derelativize(search_space=None)
        # Adapter with in-design status quo
        search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x", parameter_type=ParameterType.FLOAT, lower=0, upper=20
                ),
                RangeParameter(
                    name="y", parameter_type=ParameterType.FLOAT, lower=0, upper=20
                ),
            ]
        )
        experiment.search_space = search_space
        # clone experiment since we can't overwrite sq in subsequent tests
        experiment_1 = experiment.clone_with(
            name="experiment_1",
            status_quo=Arm(parameters={"x": 2.0, "y": 10.0}, name="0_0"),
        )
        g = Adapter(experiment=experiment_1, generator=Generator())

        # Test with no relative constraints
        objective = Objective(Metric(name="c"), minimize=True)
        oc = OptimizationConfig(
            objective=objective,
            outcome_constraints=[
                OutcomeConstraint(
                    Metric(name="m1"), ComparisonOp.LEQ, bound=2, relative=False
                ),
                ScalarizedOutcomeConstraint(
                    metrics=[Metric(name="m1"), Metric(name="m2")],
                    op=ComparisonOp.LEQ,
                    bound=2,
                    weights=[0.5, 0.5],
                    relative=False,
                ),
            ],
        )
        oc2 = t.transform_optimization_config(oc, g, None)
        self.assertTrue(oc == oc2)

        # Test with relative constraint, in-design status quo
        relative_bound = -10
        oc = OptimizationConfig(
            objective=objective,
            outcome_constraints=[
                OutcomeConstraint(
                    Metric(name="m1"), ComparisonOp.LEQ, bound=2, relative=False
                ),
                OutcomeConstraint(
                    Metric(name="m2"),
                    ComparisonOp.LEQ,
                    bound=relative_bound,
                    relative=True,
                ),
                ScalarizedOutcomeConstraint(
                    metrics=[Metric(name="m1"), Metric(name="m2")],
                    weights=[0.0, 1.0],
                    op=ComparisonOp.LEQ,
                    bound=relative_bound,
                    relative=True,
                ),
            ],
        )
        oc2 = t.transform_optimization_config(oc, g, None)
        sq_val = sq_b_predicted
        absolute_bound = (1 + np.sign(sq_val) * relative_bound / 100.0) * sq_val
        self.assertTrue(
            oc2.outcome_constraints
            == [
                OutcomeConstraint(
                    Metric(name="m1"), ComparisonOp.LEQ, bound=2, relative=False
                ),
                OutcomeConstraint(
                    Metric(name="m2"),
                    ComparisonOp.LEQ,
                    bound=absolute_bound,
                    relative=False,
                ),
                ScalarizedOutcomeConstraint(
                    metrics=[Metric(name="m1"), Metric(name="m2")],
                    weights=[0.0, 1.0],
                    op=ComparisonOp.LEQ,
                    bound=absolute_bound,
                    relative=False,
                ),
            ]
        )
        obsf = mock_predict.call_args.kwargs["observation_features"][0]
        self.assertEqual(obsf.parameters, {"x": 2.0, "y": 10.0})
        self.assertEqual(mock_predict.call_count, 1)

        # The model should not be used when `use_raw_status_quo` is True
        t2 = deepcopy(t)
        t2.config["use_raw_status_quo"] = True
        t2.transform_optimization_config(deepcopy(oc), g, None)
        self.assertEqual(mock_predict.call_count, 1)

        # Test with relative constraint, out-of-design status quo
        mock_predict.side_effect = RuntimeError()
        experiment_2 = experiment.clone_with(
            name="experiment_2",
            status_quo=Arm(parameters={"x": None, "y": None}, name="1_0"),
        )
        g = Adapter(experiment=experiment_2, generator=Generator())
        oc = OptimizationConfig(
            objective=objective,
            outcome_constraints=[
                OutcomeConstraint(
                    Metric(name="m1"), ComparisonOp.LEQ, bound=2, relative=False
                ),
                OutcomeConstraint(
                    Metric(name="m2"),
                    ComparisonOp.LEQ,
                    bound=relative_bound,
                    relative=True,
                ),
                ScalarizedOutcomeConstraint(
                    metrics=[Metric(name="m1"), Metric(name="m2")],
                    weights=[0.0, 1.0],
                    op=ComparisonOp.LEQ,
                    bound=relative_bound,
                    relative=True,
                ),
            ],
        )
        oc2 = t.transform_optimization_config(oc, g, None)
        sq_val = sq_b_observed
        absolute_bound = (1 + np.sign(sq_val) * relative_bound / 100.0) * sq_val
        self.assertEqual(
            oc2.outcome_constraints,
            [
                OutcomeConstraint(
                    Metric(name="m1"), ComparisonOp.LEQ, bound=2, relative=False
                ),
                OutcomeConstraint(
                    Metric(name="m2"),
                    ComparisonOp.LEQ,
                    bound=absolute_bound,
                    relative=False,
                ),
                ScalarizedOutcomeConstraint(
                    metrics=[Metric(name="m1"), Metric(name="m2")],
                    weights=[0.0, 1.0],
                    op=ComparisonOp.LEQ,
                    bound=absolute_bound,
                    relative=False,
                ),
            ],
        )
        self.assertEqual(mock_predict.call_count, 1)

        # Raises error if predict fails with in-design status quo
        experiment_3 = experiment.clone_with(name="experiment_3")
        experiment_3 = experiment.clone_with(
            name="experiment_3",
            status_quo=Arm(parameters={"x": 2.0, "y": 10.0}, name="0_0"),
        )
        g = Adapter(experiment=experiment_3, generator=Generator())
        oc = OptimizationConfig(
            objective=objective,
            outcome_constraints=[
                OutcomeConstraint(
                    Metric(name="m1"), ComparisonOp.LEQ, bound=2, relative=False
                ),
                OutcomeConstraint(
                    Metric(name="m2"), ComparisonOp.LEQ, bound=-10, relative=True
                ),
            ],
        )
        with self.assertRaises(RuntimeError):
            t.transform_optimization_config(oc, g, None)

        # Bypasses error if use_raw_sq
        t2 = Derelativize(search_space=None, config={"use_raw_status_quo": True})
        t2.transform_optimization_config(deepcopy(oc), g, None)

        # But not if sq arm is not available.
        with patch(
            f"{Derelativize.__module__}.unwrap_observation_data", return_value=({}, {})
        ), self.assertRaisesRegex(
            DataRequiredError, "Status-quo metric value not yet available for metric "
        ):
            t2.transform_optimization_config(deepcopy(oc), g, None)

        # Same for scalarized constraint only.
        oc_scalarized_only = OptimizationConfig(
            objective=objective,
            outcome_constraints=[
                ScalarizedOutcomeConstraint(
                    metrics=[Metric(name="m1"), Metric(name="m2")],
                    weights=[0.0, 1.0],
                    op=ComparisonOp.LEQ,
                    bound=-10,
                    relative=True,
                ),
            ],
        )
        with patch(
            f"{Derelativize.__module__}.unwrap_observation_data", return_value=({}, {})
        ), self.assertRaisesRegex(
            DataRequiredError,
            "Status-quo metric value not yet available for metric\\(s\\) ",
        ):
            t2.transform_optimization_config(deepcopy(oc_scalarized_only), g, None)

        # Raises error with relative constraint, no status quo.
        g = Adapter(
            experiment=Experiment(search_space=search_space), generator=Generator()
        )
        with self.assertRaises(DataRequiredError):
            t.transform_optimization_config(deepcopy(oc), g, None)

        # Raises error with relative constraint, no adapter.
        with self.assertRaises(ValueError):
            t.transform_optimization_config(deepcopy(oc), None, None)

    def test_errors(self) -> None:
        t = Derelativize(search_space=None)
        oc = OptimizationConfig(
            objective=Objective(Metric(name="c"), minimize=False),
            outcome_constraints=[
                OutcomeConstraint(
                    Metric(name="m1"), ComparisonOp.LEQ, bound=2, relative=True
                )
            ],
        )
        search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x", parameter_type=ParameterType.FLOAT, lower=0, upper=20
                )
            ]
        )
        adapter = Adapter(
            experiment=Experiment(search_space=search_space), generator=Generator()
        )
        with self.assertRaises(ValueError):
            t.transform_optimization_config(
                optimization_config=oc, adapter=None, fixed_features=None
            )
        with self.assertRaises(DataRequiredError):
            t.transform_optimization_config(
                optimization_config=oc, adapter=adapter, fixed_features=None
            )

    @mock_botorch_optimize
    def test_with_out_of_design_status_quo(self) -> None:
        # This test checks that model predictions are not used when the status quo
        # is out of design, regardless of whether it has data attached.

        # Branin experiment with out of design status quo.
        # Status quo is 0.0, so we adjust the search space to have it out of design.
        exp = get_branin_experiment(
            with_status_quo=True,
            with_completed_trial=True,
            with_relative_constraint=True,
            search_space=SearchSpace(
                parameters=[
                    RangeParameter(
                        name=name,
                        parameter_type=ParameterType.FLOAT,
                        lower=5.0,
                        upper=10.0,
                    )
                    for name in ("x1", "x2")
                ]
            ),
        )
        for attach_data_for_sq in [False, True]:
            if attach_data_for_sq:
                # Attach data for status quo, which will get filtered out.
                trial = exp.new_trial().add_arm(exp.status_quo).run().mark_completed()
                exp.attach_data(get_branin_data(trials=[trial], metrics=exp.metrics))

            for with_raw_sq in [False, True]:
                adapter = TorchAdapter(
                    experiment=exp,
                    generator=BoTorchGenerator(),
                    transforms=[Derelativize],
                    transform_configs={
                        "Derelativize": {"use_raw_status_quo": with_raw_sq}
                    },
                    expand_model_space=False,  # To keep SQ out of design.
                )
                if attach_data_for_sq:
                    # Since SQ is out of design, we shouldn't be using
                    # model predictions.
                    with mock.patch.object(adapter, "predict") as mock_predict:
                        adapter.gen(n=1)
                    # Check that it wasn't called with SQ features.
                    # It will be called after gen with generated candidate.
                    self.assertFalse(
                        any(
                            args.args[0] == [none_throws(adapter.status_quo).features]
                            for args in mock_predict.call_args_list
                        )
                    )
                else:
                    # If there is no data, SQ is not set and we get an error.
                    with self.assertRaisesRegex(
                        DataRequiredError, "was not fit with status quo"
                    ):
                        adapter.gen(n=1)

    @mock_botorch_optimize
    def test_warning_if_raw_sq_is_out_of_CI(self) -> None:
        # This test checks that a warning is raised if the raw status quo values
        # are outside of the CI for the predictions.

        exp = get_branin_experiment(
            with_status_quo=True,
            with_completed_trial=True,
            with_relative_constraint=True,
        )
        # Add data for SQ.
        trial = exp.new_trial().add_arm(exp.status_quo).run().mark_completed()
        exp.attach_data(get_branin_data(trials=[trial], metrics=exp.metrics))

        adapter = TorchAdapter(
            experiment=exp, generator=BoTorchGenerator(), transforms=[Derelativize]
        )
        # Shouldn't log in regular usage here.
        with self.assertNoLogs(logger=logger):
            adapter.gen(n=1)
        # With if we mock the predictions, we should get a log.
        mock_predictions = [
            {m: [10.0] for m in exp.metrics},
            {m: {m: [0.0]} for m in exp.metrics},
        ]
        with mock.patch.object(
            adapter, "predict", return_value=mock_predictions
        ), self.assertLogs(logger=logger) as mock_logs:
            adapter.gen(n=1)
        self.assertIn("deviate more than", mock_logs.records[0].getMessage())
