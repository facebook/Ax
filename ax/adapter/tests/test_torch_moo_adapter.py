#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import torch
from ax.adapter.adapter_utils import (
    get_pareto_frontier_and_configs,
    observed_hypervolume,
    observed_pareto_frontier,
    pareto_frontier,
    predicted_hypervolume,
    predicted_pareto_frontier,
)
from ax.adapter.factory import get_sobol
from ax.adapter.registry import Cont_X_trans, ST_MTGP_trans, Y_trans
from ax.adapter.torch import TorchAdapter
from ax.core.metric import Metric
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import (
    ComparisonOp,
    ObjectiveThreshold,
    OutcomeConstraint,
)
from ax.core.parameter_constraint import ParameterConstraint
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.generators.torch.botorch_moo import MultiObjectiveLegacyBoTorchGenerator
from ax.generators.torch.botorch_moo_defaults import (
    infer_objective_thresholds,
    pareto_frontier_evaluator,
)
from ax.utils.common.random import set_rng_seed
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data_multi_objective,
    get_branin_experiment_with_multi_objective,
    get_hierarchical_search_space,
    get_hss_trials_with_fixed_parameter,
    TEST_SOBOL_SEED,
)
from ax.utils.testing.mock import mock_botorch_optimize, skip_fit_gpytorch_mll
from ax.utils.testing.modeling_stubs import transform_1, transform_2
from botorch.utils.multi_objective.pareto import is_non_dominated
from pyre_extensions import assert_is_instance, none_throws

PARETO_FRONTIER_EVALUATOR_PATH = (
    f"{get_pareto_frontier_and_configs.__module__}.pareto_frontier_evaluator"
)
STUBS_PATH: str = get_branin_experiment_with_multi_objective.__module__


class MultiObjectiveTorchAdapterTest(TestCase):
    @patch(
        # Mocking `BraninMetric` as not available while running, so it will
        # be grabbed from cache during `fetch_data`.
        f"{STUBS_PATH}.BraninMetric.is_available_while_running",
        return_value=False,
    )
    @skip_fit_gpytorch_mll
    def helper_test_pareto_frontier(
        self, _, outcome_constraints: list[OutcomeConstraint] | None
    ) -> None:
        """
        Make sure Pareto-related functions run.

        Data is generated manually; this does not check that the points are actually
        Pareto-efficient or that everything works end-to-end.
        """
        exp = get_branin_experiment_with_multi_objective(
            has_optimization_config=True, with_batch=True
        )
        for trial in exp.trials.values():
            trial.mark_running(no_runner_required=True).mark_completed()
        metrics_dict = none_throws(exp.optimization_config).metrics
        objective_bound = 5.0
        objective_thresholds = [
            ObjectiveThreshold(
                metric=metrics_dict[f"branin_{letter}"],
                bound=objective_bound,
                relative=False,
                op=ComparisonOp.LEQ,
            )
            for letter in "ab"
        ]
        exp.optimization_config = assert_is_instance(
            exp.optimization_config, MultiObjectiveOptimizationConfig
        ).clone_with_args(
            objective_thresholds=objective_thresholds,
            outcome_constraints=outcome_constraints,
        )

        n_outcomes = len(exp.metrics)
        exp.attach_data(
            get_branin_data_multi_objective(
                trial_indices=exp.trials.keys(), outcomes=list(exp.metrics)
            ),
        )
        adapter = TorchAdapter(
            search_space=exp.search_space,
            generator=MultiObjectiveLegacyBoTorchGenerator(),
            optimization_config=exp.optimization_config,
            transforms=[transform_1, transform_2],
            experiment=exp,
            data=exp.fetch_data(),
        )
        with patch(
            PARETO_FRONTIER_EVALUATOR_PATH, wraps=pareto_frontier_evaluator
        ) as wrapped_frontier_evaluator:
            observed_frontier = observed_pareto_frontier(
                adapter=adapter, objective_thresholds=objective_thresholds
            )

        wrapped_frontier_evaluator.assert_called_once()
        self.assertIsNone(wrapped_frontier_evaluator.call_args[1]["X"])
        self.assertEqual(1, len(observed_frontier))
        self.assertEqual(observed_frontier[0].arm_name, "0_0")

        with self.assertRaises(ValueError):
            predicted_pareto_frontier(
                adapter=adapter,
                objective_thresholds=objective_thresholds,
                observation_features=[],
            )

        predicted_frontier = predicted_pareto_frontier(
            adapter=adapter,
            objective_thresholds=objective_thresholds,
            observation_features=None,
        )
        self.assertEqual(predicted_frontier[0].arm_name, "0_0")

        observation_features = [
            ObservationFeatures(parameters={"x1": 0.0, "x2": 1.0}),
            # magic number to ensure the model makes one non-negative
            # prediction, so there is a point within the thresholds
            ObservationFeatures(parameters={"x1": -2.0, "x2": 6.3}),
        ]
        extra_outcome = ["branin_c"] if outcome_constraints is not None else []
        observation_data = [
            ObservationData(
                metric_names=["branin_b", "branin_a"] + extra_outcome,
                means=np.arange(1, 1 + n_outcomes),
                covariance=np.eye(n_outcomes),
            ),
            ObservationData(
                metric_names=["branin_a", "branin_b"] + extra_outcome,
                means=np.arange(3, 3 + n_outcomes),
                covariance=np.eye(n_outcomes),
            ),
        ]
        # appease Pyre (this condition is True)
        if isinstance(exp.optimization_config, MultiObjectiveOptimizationConfig):
            predicted_frontier = predicted_pareto_frontier(
                adapter=adapter,
                objective_thresholds=objective_thresholds,
                observation_features=observation_features,
                optimization_config=exp.optimization_config,
            )
        self.assertTrue(len(predicted_frontier) <= 2)
        self.assertIsNone(predicted_frontier[0].arm_name, None)

        with patch(
            PARETO_FRONTIER_EVALUATOR_PATH, wraps=pareto_frontier_evaluator
        ) as wrapped_frontier_evaluator:
            (
                observed_frontier,
                f,
                obj_w,
                obj_t,
            ) = get_pareto_frontier_and_configs(
                adapter=adapter,
                objective_thresholds=objective_thresholds,
                observation_features=observation_features,
                observation_data=observation_data,
                use_model_predictions=False,
            )

        wrapped_frontier_evaluator.assert_called_once()
        self.assertEqual(f.shape, (1, n_outcomes))
        self.assertTrue(torch.equal(obj_w[:2], -torch.ones(2, dtype=torch.double)))
        self.assertTrue(obj_t is not None)
        self.assertTrue(
            torch.equal(
                none_throws(obj_t)[:2], torch.full((2,), 5.0, dtype=torch.double)
            )
        )
        observed_frontier2 = pareto_frontier(
            adapter=adapter,
            objective_thresholds=objective_thresholds,
            observation_features=observation_features,
            observation_data=observation_data,
            use_model_predictions=False,
        )
        self.assertEqual(observed_frontier, observed_frontier2)

        # Remove the thresholds for testing None handling.
        assert_is_instance(
            adapter._optimization_config, MultiObjectiveOptimizationConfig
        )._objective_thresholds = []
        with patch(
            PARETO_FRONTIER_EVALUATOR_PATH, wraps=pareto_frontier_evaluator
        ) as wrapped_frontier_evaluator:
            (
                observed_frontier,
                f,
                obj_w,
                obj_t,
            ) = get_pareto_frontier_and_configs(
                adapter=adapter,
                objective_thresholds=None,
                observation_features=observation_features,
                observation_data=observation_data,
                use_model_predictions=False,
            )
        wrapped_frontier_evaluator.assert_called_once()
        self.assertIsNone(wrapped_frontier_evaluator.call_args[1]["X"])
        true_Y = torch.tensor([[2.0, 1.0], [3.0, 4.0]], dtype=torch.double)
        self.assertTrue(
            torch.equal(
                wrapped_frontier_evaluator.call_args[1]["Y"][:, :2],
                true_Y,
            )
        )
        self.assertTrue(torch.equal(objective_bound - f[:, :2], true_Y[1:, :]))
        self.assertIsNone(obj_t)

    def test_pareto_frontier(self) -> None:
        """
        Run helper_test_pareto_frontier with and without outcome constraints.

        The constraint won't come close to binding, so it shouldn't affect results.
        """
        constraint = OutcomeConstraint(
            Metric(name="branin_c"), ComparisonOp.LEQ, bound=100.0, relative=False
        )
        for outcome_constraints in [None, [constraint]]:
            with self.subTest(outcome_constraints=outcome_constraints):
                self.helper_test_pareto_frontier(
                    outcome_constraints=outcome_constraints
                )

    @mock_botorch_optimize
    def test_get_pareto_frontier_and_configs_input_validation(self) -> None:
        exp = get_branin_experiment_with_multi_objective(
            has_optimization_config=True, with_batch=True
        )
        for trial in exp.trials.values():
            trial.mark_running(no_runner_required=True).mark_completed()
        metrics_dict = none_throws(exp.optimization_config).metrics
        objective_thresholds = [
            ObjectiveThreshold(
                metric=metrics_dict[f"branin_{letter}"],
                bound=5.0,
                relative=False,
                op=ComparisonOp.LEQ,
            )
            for letter in "ab"
        ]
        exp.optimization_config = assert_is_instance(
            exp.optimization_config, MultiObjectiveOptimizationConfig
        ).clone_with_args(
            objective_thresholds=objective_thresholds,
        )

        exp.attach_data(
            get_branin_data_multi_objective(trial_indices=exp.trials.keys()),
        )
        adapter = TorchAdapter(
            experiment=exp,
            search_space=exp.search_space,
            data=exp.fetch_data(),
            generator=MultiObjectiveLegacyBoTorchGenerator(),
            transforms=[],
        )
        observation_features = [
            ObservationFeatures(parameters={"x1": 0.0, "x2": 1.0}),
            ObservationFeatures(parameters={"x1": 1.0, "x2": 0.0}),
        ]

        with self.assertWarns(
            Warning,
            msg="You provided `observation_data` when `use_model_predictions` is True; "
            "`observation_data` will not be used.",
        ):
            res = get_pareto_frontier_and_configs(
                adapter,
                observation_features=observation_features,
                use_model_predictions=True,
                observation_data=[],
            )
            self.assertEqual(len(res), 4)

        with self.assertRaises(ValueError):
            get_pareto_frontier_and_configs(
                adapter, observation_features=[], use_model_predictions=False
            )

    @patch(
        # Mocking `BraninMetric` as not available while running, so it will
        # be grabbed from cache during `fetch_data`.
        f"{STUBS_PATH}.BraninMetric.is_available_while_running",
        return_value=False,
    )
    def test_hypervolume(self, _, cuda: bool = False) -> None:
        for num_objectives in (2, 3):
            exp = get_branin_experiment_with_multi_objective(
                has_optimization_config=True,
                with_batch=True,
                num_objectives=num_objectives,
            )
            for trial in exp.trials.values():
                trial.mark_running(no_runner_required=True).mark_completed()
            # pyre-fixme[16]: Optional type has no attribute `metrics`.
            metrics_dict = exp.optimization_config.metrics
            # Objective thresholds and synthetic observations chosen to have closed-form
            # hypervolumes to test.
            objective_thresholds = [
                ObjectiveThreshold(
                    metric=metrics_dict["branin_a"],
                    bound=10.0,
                    relative=False,
                    op=ComparisonOp.LEQ,
                ),
                ObjectiveThreshold(
                    metric=metrics_dict["branin_b"],
                    bound=9.0,
                    relative=False,
                    op=ComparisonOp.LEQ,
                ),
            ]
            if num_objectives == 3:
                objective_thresholds.append(
                    ObjectiveThreshold(
                        metric=metrics_dict["branin_c"],
                        bound=8.0,
                        relative=False,
                        op=ComparisonOp.LEQ,
                    )
                )
            optimization_config = assert_is_instance(
                exp.optimization_config, MultiObjectiveOptimizationConfig
            ).clone_with_args(objective_thresholds=objective_thresholds)
            exp.attach_data(
                get_branin_data_multi_objective(
                    trial_indices=exp.trials.keys(),
                    outcomes=list(optimization_config.metrics),
                )
            )
            adapter = TorchAdapter(
                search_space=exp.search_space,
                generator=MultiObjectiveLegacyBoTorchGenerator(),
                optimization_config=optimization_config,
                transforms=[],
                experiment=exp,
                data=exp.fetch_data(),
                torch_device=torch.device("cuda" if cuda else "cpu"),
            )
            with patch(
                PARETO_FRONTIER_EVALUATOR_PATH, wraps=pareto_frontier_evaluator
            ) as wrapped_frontier_evaluator:
                hv = observed_hypervolume(
                    adapter=adapter, objective_thresholds=objective_thresholds
                )
            expected_hv = 20 if num_objectives == 2 else 60  # 5 * 4 (* 3)
            wrapped_frontier_evaluator.assert_called_once()
            self.assertEqual(expected_hv, hv)
            if num_objectives == 3:
                # Test selected_metrics
                hv = observed_hypervolume(
                    adapter=adapter,
                    objective_thresholds=objective_thresholds,
                    selected_metrics=["branin_a", "branin_c"],
                )
                expected_hv = 15  # (5 - 0) * (5 - 2)
                self.assertEqual(expected_hv, hv)
                # test that non-objective outcome raises value error
                with self.assertRaises(ValueError):
                    hv = observed_hypervolume(
                        adapter=adapter,
                        objective_thresholds=objective_thresholds,
                        selected_metrics=["tracking"],
                    )

            with self.assertRaises(ValueError):
                predicted_hypervolume(
                    adapter=adapter,
                    objective_thresholds=objective_thresholds,
                    observation_features=[],
                )

            observation_features = [
                ObservationFeatures(parameters={"x1": 1.0, "x2": 2.0}),
                ObservationFeatures(parameters={"x1": 2.0, "x2": 1.0}),
            ]
            predicted_hv = predicted_hypervolume(
                adapter=adapter,
                objective_thresholds=objective_thresholds,
                observation_features=observation_features,
            )
            self.assertTrue(predicted_hv >= 0)
            if num_objectives == 3:
                # Test selected_metrics
                predicted_hv = predicted_hypervolume(
                    adapter=adapter,
                    objective_thresholds=objective_thresholds,
                    observation_features=observation_features,
                    selected_metrics=["branin_a", "branin_c"],
                )
                self.assertTrue(predicted_hv >= 0)

    def test_hypervolume_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_hypervolume(cuda=True)

    @patch(
        # Mocking `BraninMetric` as not available while running, so it will
        # be grabbed from cache during `fetch_data`.
        f"{STUBS_PATH}.BraninMetric.is_available_while_running",
        return_value=False,
    )
    @mock_botorch_optimize
    def test_infer_objective_thresholds(self, _, cuda: bool = False) -> None:
        # lightweight test
        exp = get_branin_experiment_with_multi_objective(
            has_optimization_config=True,
            with_batch=True,
            with_status_quo=True,
        )
        for trial in exp.trials.values():
            trial.mark_running(no_runner_required=True).mark_completed()
        exp.attach_data(
            get_branin_data_multi_objective(trial_indices=exp.trials.keys())
        )
        data = exp.fetch_data()
        adapter = TorchAdapter(
            search_space=exp.search_space,
            generator=MultiObjectiveLegacyBoTorchGenerator(),
            optimization_config=exp.optimization_config,
            transforms=Cont_X_trans + Y_trans,
            torch_device=torch.device("cuda" if cuda else "cpu"),
            experiment=exp,
            data=data,
        )
        fixed_features = ObservationFeatures(parameters={"x1": 0.0})
        search_space = exp.search_space.clone()
        param_constraints = [
            ParameterConstraint(constraint_dict={"x1": 1.0}, bound=10.0)
        ]
        search_space.add_parameter_constraints(param_constraints)
        oc = none_throws(exp.optimization_config).clone()
        oc.objective._objectives[0].minimize = True

        for use_partial_thresholds in (False, True):
            if use_partial_thresholds:
                assert_is_instance(
                    oc, MultiObjectiveOptimizationConfig
                )._objective_thresholds = [
                    ObjectiveThreshold(
                        metric=oc.objective.metrics[0],
                        bound=2.0,
                        relative=False,
                        op=ComparisonOp.LEQ,
                    )
                ]
            expected_base_gen_args = adapter._get_transformed_gen_args(
                search_space=search_space.clone(),
                optimization_config=oc,
                fixed_features=fixed_features,
            )
            with ExitStack() as es:
                mock_model_infer_obj_t = es.enter_context(
                    patch(
                        "ax.adapter.torch.infer_objective_thresholds",
                        wraps=infer_objective_thresholds,
                    )
                )
                mock_get_transformed_gen_args = es.enter_context(
                    patch.object(
                        adapter,
                        "_get_transformed_gen_args",
                        wraps=adapter._get_transformed_gen_args,
                    )
                )
                mock_get_transformed_model_gen_args = es.enter_context(
                    patch.object(
                        adapter,
                        "_get_transformed_model_gen_args",
                        wraps=adapter._get_transformed_model_gen_args,
                    )
                )
                mock_untransform_objective_thresholds = es.enter_context(
                    patch.object(
                        adapter,
                        "_untransform_objective_thresholds",
                        wraps=adapter._untransform_objective_thresholds,
                    )
                )
                obj_thresholds = adapter.infer_objective_thresholds(
                    search_space=search_space,
                    optimization_config=oc,
                    fixed_features=fixed_features,
                )
                expected_obj_weights = torch.tensor([-1.0, -1.0], dtype=torch.double)
                ckwargs = mock_model_infer_obj_t.call_args[1]
                self.assertTrue(
                    torch.equal(ckwargs["objective_weights"], expected_obj_weights)
                )
                # check that transforms have been applied (at least UnitX)
                self.assertEqual(ckwargs["bounds"], [(0.0, 1.0), (0.0, 1.0)])
                lc = ckwargs["linear_constraints"]
                self.assertTrue(
                    torch.equal(lc[0], torch.tensor([[15.0, 0.0]], dtype=torch.double))
                )
                self.assertTrue(
                    torch.equal(lc[1], torch.tensor([[15.0]], dtype=torch.double))
                )
                self.assertEqual(ckwargs["fixed_features"], {0: 1.0 / 3.0})
                mock_get_transformed_gen_args.assert_called_once()
                mock_get_transformed_model_gen_args.assert_called_once_with(
                    search_space=expected_base_gen_args.search_space,
                    fixed_features=expected_base_gen_args.fixed_features,
                    pending_observations=expected_base_gen_args.pending_observations,
                    optimization_config=expected_base_gen_args.optimization_config,
                )
                mock_untransform_objective_thresholds.assert_called_once()
                ckwargs = mock_untransform_objective_thresholds.call_args[1]

                self.assertTrue(
                    torch.equal(ckwargs["objective_weights"], expected_obj_weights)
                )
            self.assertEqual(obj_thresholds[0].metric.name, "branin_a")
            self.assertEqual(obj_thresholds[1].metric.name, "branin_b")
            self.assertEqual(obj_thresholds[0].op, ComparisonOp.LEQ)
            self.assertEqual(obj_thresholds[1].op, ComparisonOp.LEQ)
            self.assertFalse(obj_thresholds[0].relative)
            self.assertFalse(obj_thresholds[1].relative)
            df = exp.to_df()
            Y = np.stack([df.branin_a.values, df.branin_b.values]).T
            Y = torch.from_numpy(Y)
            Y[:, 0] *= -1
            pareto_Y = Y[is_non_dominated(Y)]
            nadir = pareto_Y.min(dim=0).values
            self.assertTrue(
                np.all(
                    -np.array([obj_thresholds[0].bound, obj_thresholds[1].bound])
                    < nadir.numpy()
                )
            )

        # test using MTGP
        sobol_generator = get_sobol(
            search_space=exp.search_space,
            seed=TEST_SOBOL_SEED,
            # set initial position equal to the number of sobol arms generated
            # so far. This means that new sobol arms will complement the previous
            # arms in a space-filling fashion
            init_position=len(exp.arms_by_name) - 1,
        )
        sobol_run = sobol_generator.gen(n=2)
        trial = exp.new_batch_trial(add_status_quo_arm=True)
        trial.add_generator_run(sobol_run)
        trial.mark_running(no_runner_required=True).mark_completed()
        data = exp.fetch_data()
        set_rng_seed(0)  # make model fitting deterministic
        adapter = TorchAdapter(
            search_space=exp.search_space,
            generator=MultiObjectiveLegacyBoTorchGenerator(),
            optimization_config=exp.optimization_config,
            transforms=ST_MTGP_trans,
            experiment=exp,
            data=data,
        )
        fixed_features = ObservationFeatures(parameters={}, trial_index=1)
        expected_base_gen_args = adapter._get_transformed_gen_args(
            search_space=search_space.clone(),
            optimization_config=exp.optimization_config,
            fixed_features=fixed_features,
        )
        with ExitStack() as es:
            mock_model_infer_obj_t = es.enter_context(
                patch(
                    "ax.adapter.torch.infer_objective_thresholds",
                    wraps=infer_objective_thresholds,
                )
            )
            mock_untransform_objective_thresholds = es.enter_context(
                patch.object(
                    adapter,
                    "_untransform_objective_thresholds",
                    wraps=adapter._untransform_objective_thresholds,
                )
            )
            obj_thresholds = adapter.infer_objective_thresholds(
                search_space=search_space,
                optimization_config=exp.optimization_config,
                fixed_features=fixed_features,
            )
            ckwargs = mock_model_infer_obj_t.call_args[1]
            self.assertEqual(ckwargs["fixed_features"], {2: 1.0})
            mock_untransform_objective_thresholds.assert_called_once()
        self.assertEqual(obj_thresholds[0].metric.name, "branin_a")
        self.assertEqual(obj_thresholds[1].metric.name, "branin_b")
        self.assertEqual(obj_thresholds[0].op, ComparisonOp.LEQ)
        self.assertEqual(obj_thresholds[1].op, ComparisonOp.LEQ)
        self.assertFalse(obj_thresholds[0].relative)
        self.assertFalse(obj_thresholds[1].relative)
        df = exp.to_df()
        trial_mask = df.trial_index == 1
        Y = np.stack([df.branin_a.values[trial_mask], df.branin_b.values[trial_mask]]).T
        Y = torch.from_numpy(Y)
        pareto_Y = Y[is_non_dominated(Y)]
        nadir = pareto_Y.min(dim=0).values
        self.assertTrue(
            np.all(
                np.array([obj_thresholds[0].bound, obj_thresholds[1].bound])
                < nadir.numpy()
            )
        )

        # test with HSS
        hss = get_hierarchical_search_space(with_fixed_parameter=True)
        exp = get_branin_experiment_with_multi_objective(has_optimization_config=True)
        data = get_branin_data_multi_objective(trial_indices=[0, 1])
        # Update trials to match the search space.
        exp._search_space = hss
        exp._trials = get_hss_trials_with_fixed_parameter(exp=exp)
        adapter = TorchAdapter(
            search_space=hss,
            generator=MultiObjectiveLegacyBoTorchGenerator(),
            optimization_config=exp.optimization_config,
            transforms=Cont_X_trans + Y_trans,
            torch_device=torch.device("cuda" if cuda else "cpu"),
            experiment=exp,
            data=data,
            # [T143911996] The trials get ignored without fit_out_of_design.
            fit_out_of_design=True,
        )
        self.assertIn("Cast", adapter.transforms)
        with patch.object(
            adapter,
            "_untransform_objective_thresholds",
            wraps=adapter._untransform_objective_thresholds,
        ) as mock_untransform, patch.object(
            adapter.transforms["Cast"],
            "untransform_observation_features",
            wraps=adapter.transforms["Cast"].untransform_observation_features,
        ) as wrapped_cast:
            obj_thresholds = adapter.infer_objective_thresholds(
                search_space=hss,
                optimization_config=exp.optimization_config,
                fixed_features=None,
            )
        mock_untransform.assert_called_once()
        self.assertEqual(wrapped_cast.call_count, 0)
        self.assertEqual(obj_thresholds[0].metric.name, "branin_a")
        self.assertEqual(obj_thresholds[1].metric.name, "branin_b")
        self.assertEqual(obj_thresholds[0].op, ComparisonOp.LEQ)
        self.assertEqual(obj_thresholds[1].op, ComparisonOp.LEQ)
        self.assertFalse(obj_thresholds[0].relative)
        self.assertFalse(obj_thresholds[1].relative)

        # test with MBM
        exp = get_branin_experiment_with_multi_objective(
            has_optimization_config=True,
            with_batch=True,
            with_status_quo=True,
        )
        for trial in exp.trials.values():
            trial.mark_running(no_runner_required=True).mark_completed()
        exp.attach_data(
            get_branin_data_multi_objective(trial_indices=exp.trials.keys())
        )
        data = exp.fetch_data()
        adapter = TorchAdapter(
            search_space=exp.search_space,
            generator=BoTorchGenerator(),
            optimization_config=exp.optimization_config,
            transforms=Cont_X_trans + Y_trans,
            torch_device=torch.device("cuda" if cuda else "cpu"),
            experiment=exp,
            data=data,
            # [T143911996] The trials get ignored without fit_out_of_design.
            fit_out_of_design=True,
        )
        obj_thresholds = adapter.infer_objective_thresholds(
            search_space=exp.search_space,
            optimization_config=exp.optimization_config,
            fixed_features=None,
        )
        self.assertEqual(obj_thresholds[0].metric.name, "branin_a")
        self.assertEqual(obj_thresholds[1].metric.name, "branin_b")
        self.assertEqual(obj_thresholds[0].op, ComparisonOp.LEQ)
        self.assertEqual(obj_thresholds[1].op, ComparisonOp.LEQ)
        self.assertFalse(obj_thresholds[0].relative)
        self.assertFalse(obj_thresholds[1].relative)

    def test_best_point(self) -> None:
        exp = get_branin_experiment_with_multi_objective(
            has_optimization_config=True, with_batch=True
        )
        for trial in exp.trials.values():
            trial.mark_running(no_runner_required=True).mark_completed()
        exp.attach_data(
            get_branin_data_multi_objective(trial_indices=exp.trials.keys())
        )
        adapter = TorchAdapter(
            search_space=exp.search_space,
            generator=MultiObjectiveLegacyBoTorchGenerator(),
            optimization_config=exp.optimization_config,
            transforms=[],
            experiment=exp,
            data=exp.fetch_data(),
        )
        self.assertIsNone(adapter.model_best_point())
