#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

from ax.adapter.base import Adapter
from ax.adapter.data_utils import (
    DataLoaderConfig,
    ExperimentData,
    extract_experiment_data,
)
from ax.adapter.transforms.objective_as_constraint import ObjectiveAsConstraint
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import (
    ObjectiveThreshold,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
)
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.generators.base import Generator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from pyre_extensions import none_throws


class ObjectiveAsConstraintTest(TestCase):
    def _make_experiment_adapter_and_data(
        self,
        observations: list[list[float]],
        constraint_bound: float = 1.0,
        constraint_op: ComparisonOp = ComparisonOp.GEQ,
        relative_constraint: bool = False,
        minimize: bool = False,
    ) -> tuple[Experiment, Adapter, ExperimentData]:
        """Helper to create an experiment, adapter, and experiment data.

        Creates a single-objective experiment with one outcome constraint on m2.
        The objective is on m1.

        Args:
            observations: List of [m1_value, m2_value] observations. The first
                observation is the status quo.
            constraint_bound: Bound for the outcome constraint on m2.
            constraint_op: Comparison op for the constraint.
            relative_constraint: Whether the constraint is relative.
            minimize: Whether the objective should be minimized.

        Returns:
            Tuple of (experiment, adapter, experiment_data).
        """
        optimization_config = OptimizationConfig(
            objective=Objective(
                metric=Metric("m1", lower_is_better=minimize), minimize=minimize
            ),
            outcome_constraints=[
                OutcomeConstraint(
                    metric=Metric("m2", lower_is_better=True),
                    op=constraint_op,
                    bound=constraint_bound,
                    relative=relative_constraint,
                ),
            ],
        )
        search_space = SearchSpace(
            parameters=[
                RangeParameter("x", ParameterType.FLOAT, 0.0, 10.0),
                RangeParameter("y", ParameterType.FLOAT, 0.0, 10.0),
            ]
        )
        sq_params = {"x": 0.0, "y": 0.0}
        parameterizations = [sq_params] + [
            {"x": float(j + 1), "y": float(j + 1)} for j in range(len(observations) - 1)
        ]

        experiment = get_experiment_with_observations(
            observations=observations,
            optimization_config=optimization_config,
            parameterizations=parameterizations,
            search_space=search_space,
            status_quo=Arm(parameters=sq_params, name="0_0"),
        )

        adapter = Adapter(experiment=experiment, generator=Generator())

        experiment_data = extract_experiment_data(
            experiment=experiment,
            data_loader_config=DataLoaderConfig(),
        )

        return experiment, adapter, experiment_data

    def test_no_op_when_feasible_points_exist(self) -> None:
        """Test that the transform is a no-op when some points are feasible."""
        # m2 >= 1.0 is the constraint. The second observation has m2 = 5.0
        # which satisfies m2 >= 1.0, so there are feasible points.
        _, adapter, experiment_data = self._make_experiment_adapter_and_data(
            observations=[[1.0, 0.5], [2.0, 5.0]],
            constraint_bound=1.0,
            constraint_op=ComparisonOp.GEQ,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        self.assertFalse(t._should_add_constraint)

        # transform_optimization_config should not modify the config
        opt_config = none_throws(deepcopy(adapter._experiment.optimization_config))
        transformed = t.transform_optimization_config(opt_config, adapter)
        self.assertEqual(len(transformed.outcome_constraints), 1)

    def test_adds_constraint_when_no_feasible_points(self) -> None:
        """Test that an absolute constraint at the SQ value is added when no
        points are feasible."""
        # m2 >= 10.0 is the constraint. Both observations have m2 < 10.
        # SQ has m1 = 1.0, so we expect constraint m1 >= 1.0.
        _, adapter, experiment_data = self._make_experiment_adapter_and_data(
            observations=[[1.0, 0.5], [2.0, 5.0]],
            constraint_bound=10.0,
            constraint_op=ComparisonOp.GEQ,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        self.assertTrue(t._should_add_constraint)

        opt_config = deepcopy(adapter._experiment.optimization_config)
        assert opt_config is not None
        transformed = t.transform_optimization_config(opt_config, adapter)
        self.assertEqual(len(transformed.outcome_constraints), 2)

        new_constraint = transformed.outcome_constraints[1]
        self.assertEqual(new_constraint.metric.name, "m1")
        self.assertEqual(new_constraint.op, ComparisonOp.GEQ)
        self.assertEqual(new_constraint.bound, 1.0)  # SQ value for m1
        self.assertFalse(new_constraint.relative)

    def test_adds_leq_constraint_when_minimizing(self) -> None:
        """Test that LEQ constraint at SQ value is added when minimizing."""
        # SQ has m1 = 1.0, so we expect constraint m1 <= 1.0.
        _, adapter, experiment_data = self._make_experiment_adapter_and_data(
            observations=[[1.0, 0.5], [2.0, 5.0]],
            constraint_bound=10.0,
            constraint_op=ComparisonOp.GEQ,
            minimize=True,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        self.assertTrue(t._should_add_constraint)

        opt_config = deepcopy(adapter._experiment.optimization_config)
        assert opt_config is not None
        transformed = t.transform_optimization_config(opt_config, adapter)

        new_constraint = transformed.outcome_constraints[1]
        self.assertEqual(new_constraint.metric.name, "m1")
        self.assertEqual(new_constraint.op, ComparisonOp.LEQ)
        self.assertEqual(new_constraint.bound, 1.0)  # SQ value for m1
        self.assertFalse(new_constraint.relative)

    def test_no_op_without_status_quo(self) -> None:
        """Test that the transform is a no-op without a status quo."""
        optimization_config = OptimizationConfig(
            objective=Objective(Metric("m1", lower_is_better=False), minimize=False),
            outcome_constraints=[
                OutcomeConstraint(
                    Metric("m2", lower_is_better=True),
                    ComparisonOp.GEQ,
                    bound=10.0,
                ),
            ],
        )
        experiment = get_experiment_with_observations(
            observations=[[1.0, 0.5], [2.0, 5.0]],
            optimization_config=optimization_config,
        )

        adapter = Adapter(experiment=experiment, generator=Generator())
        experiment_data = extract_experiment_data(
            experiment=experiment,
            data_loader_config=DataLoaderConfig(),
        )

        t = ObjectiveAsConstraint(
            search_space=experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        self.assertFalse(t._should_add_constraint)

    def test_no_op_without_constraints(self) -> None:
        """Test that the transform is a no-op when there are no constraints."""
        optimization_config = OptimizationConfig(
            objective=Objective(Metric("m1", lower_is_better=False), minimize=False),
        )
        search_space = SearchSpace(
            parameters=[
                RangeParameter("x", ParameterType.FLOAT, 0.0, 10.0),
                RangeParameter("y", ParameterType.FLOAT, 0.0, 10.0),
            ]
        )
        sq_params = {"x": 0.0, "y": 0.0}
        experiment = get_experiment_with_observations(
            observations=[[1.0], [2.0]],
            optimization_config=optimization_config,
            parameterizations=[sq_params, {"x": 1.0, "y": 2.0}],
            search_space=search_space,
            status_quo=Arm(parameters=sq_params, name="0_0"),
        )

        adapter = Adapter(experiment=experiment, generator=Generator())
        experiment_data = extract_experiment_data(
            experiment=experiment,
            data_loader_config=DataLoaderConfig(),
        )

        t = ObjectiveAsConstraint(
            search_space=experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        self.assertFalse(t._should_add_constraint)

    def test_untransform_removes_added_constraint(self) -> None:
        """Test that untransform removes the added objective constraint."""
        _, adapter, experiment_data = self._make_experiment_adapter_and_data(
            observations=[[1.0, 0.5], [2.0, 5.0]],
            constraint_bound=10.0,
            constraint_op=ComparisonOp.GEQ,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        opt_config = deepcopy(adapter._experiment.optimization_config)
        assert opt_config is not None
        transformed = t.transform_optimization_config(opt_config, adapter)
        self.assertEqual(len(transformed.outcome_constraints), 2)

        # Untransform should remove the added constraint
        untransformed = t.untransform_outcome_constraints(
            outcome_constraints=transformed.outcome_constraints,
        )
        self.assertEqual(len(untransformed), 1)
        self.assertEqual(untransformed[0].metric.name, "m2")

    def test_raises_on_relative_constraints(self) -> None:
        """Test that a ValueError is raised in transform_optimization_config
        if any constraint is relative (Derelativize has not been applied yet).
        """
        # Set up with infeasible points and relative constraint.
        _, adapter, experiment_data = self._make_experiment_adapter_and_data(
            observations=[[1.0, 1.0], [2.0, 1.5]],
            constraint_bound=100.0,  # 100% relative
            constraint_op=ComparisonOp.GEQ,
            relative_constraint=True,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        # _should_add_constraint is True because no feasible points.
        self.assertTrue(t._should_add_constraint)

        # transform_optimization_config should raise ValueError due to
        # relative constraints.
        opt_config = deepcopy(adapter._experiment.optimization_config)
        assert opt_config is not None
        with self.assertRaisesRegex(
            ValueError,
            "ObjectiveAsConstraint requires all outcome constraints to be absolute",
        ):
            t.transform_optimization_config(opt_config, adapter)

    def test_relative_constraint_feasibility_check(self) -> None:
        """Test that _is_point_feasible correctly handles relative constraints.

        Relative constraints are evaluated relative to the status quo value.
        For a GEQ constraint with bound B%, the effective bound is sq_val * (1 + B/100).
        """
        # Relative constraint: m2 >= 50% (i.e., m2 >= sq_m2 * 1.5).
        # SQ has m2 = 2.0, so bound is 2.0 * 1.5 = 3.0.
        # Observation 1: m2 = 2.0 < 3.0 → infeasible.
        # Observation 2: m2 = 4.0 >= 3.0 → feasible.
        _, adapter, experiment_data = self._make_experiment_adapter_and_data(
            observations=[[1.0, 2.0], [2.0, 4.0]],
            constraint_bound=50.0,  # 50% relative
            constraint_op=ComparisonOp.GEQ,
            relative_constraint=True,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        # There are feasible points (observation 2: m2 = 4.0 >= 3.0).
        self.assertFalse(t._should_add_constraint)

    def test_leq_constraint_feasibility(self) -> None:
        """Test feasibility checking with LEQ constraints."""
        # m2 <= 0.3 constraint. Both observations have m2 > 0.3, so infeasible.
        _, adapter, experiment_data = self._make_experiment_adapter_and_data(
            observations=[[1.0, 0.5], [2.0, 5.0]],
            constraint_bound=0.3,
            constraint_op=ComparisonOp.LEQ,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        self.assertTrue(t._should_add_constraint)

    def test_leq_constraint_feasible(self) -> None:
        """Test that LEQ constraints with feasible points are correctly detected."""
        # m2 <= 10.0 constraint. Both observations have m2 <= 10.0, so feasible.
        _, adapter, experiment_data = self._make_experiment_adapter_and_data(
            observations=[[1.0, 0.5], [2.0, 5.0]],
            constraint_bound=10.0,
            constraint_op=ComparisonOp.LEQ,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        self.assertFalse(t._should_add_constraint)

    def test_no_op_for_experiment_data(self) -> None:
        """Test that transform_experiment_data is a no-op."""
        _, adapter, experiment_data = self._make_experiment_adapter_and_data(
            observations=[[1.0, 0.5], [2.0, 5.0]],
            constraint_bound=10.0,
            constraint_op=ComparisonOp.GEQ,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        original = deepcopy(experiment_data)
        result = t.transform_experiment_data(experiment_data)
        self.assertTrue(result.observation_data.equals(original.observation_data))


class ObjectiveAsConstraintScalarizedObjectiveTest(TestCase):
    """Tests for ScalarizedObjective support in ObjectiveAsConstraint."""

    def _make_scalarized_experiment_adapter_and_data(
        self,
        observations: list[list[float]],
        constraint_bound: float = 1.0,
        constraint_op: ComparisonOp = ComparisonOp.GEQ,
        minimize: bool = False,
        weights: list[float] | None = None,
        use_moo_config: bool = False,
    ) -> tuple[Experiment, Adapter, ExperimentData]:
        """Helper to create a ScalarizedObjective experiment.

        Creates an experiment with a ScalarizedObjective on m1 and m2, and
        an outcome constraint on m3.

        Args:
            observations: List of [m1_value, m2_value, m3_value] observations.
                The first observation is the status quo.
            constraint_bound: Bound for the outcome constraint on m3.
            constraint_op: Comparison op for the constraint.
            minimize: Whether the scalarized objective should be minimized.
            weights: Weights for the ScalarizedObjective.
            use_moo_config: If True, use MultiObjectiveOptimizationConfig.
        """
        scalarized_objective = ScalarizedObjective(
            metrics=[Metric("m1"), Metric("m2")],
            weights=weights or [1.0, 1.0],
            minimize=minimize,
        )

        outcome_constraints = [
            OutcomeConstraint(
                metric=Metric("m3", lower_is_better=True),
                op=constraint_op,
                bound=constraint_bound,
                relative=False,
            ),
        ]

        if use_moo_config:
            optimization_config = MultiObjectiveOptimizationConfig(
                objective=scalarized_objective,
                outcome_constraints=outcome_constraints,
            )
        else:
            optimization_config = OptimizationConfig(
                objective=scalarized_objective,
                outcome_constraints=outcome_constraints,
            )

        search_space = SearchSpace(
            parameters=[
                RangeParameter("x", ParameterType.FLOAT, 0.0, 10.0),
                RangeParameter("y", ParameterType.FLOAT, 0.0, 10.0),
            ]
        )
        sq_params = {"x": 0.0, "y": 0.0}
        parameterizations = [sq_params] + [
            {"x": float(j + 1), "y": float(j + 1)} for j in range(len(observations) - 1)
        ]

        experiment = get_experiment_with_observations(
            observations=observations,
            optimization_config=optimization_config,
            parameterizations=parameterizations,
            search_space=search_space,
            status_quo=Arm(parameters=sq_params, name="0_0"),
        )

        adapter = Adapter(experiment=experiment, generator=Generator())

        experiment_data = extract_experiment_data(
            experiment=experiment,
            data_loader_config=DataLoaderConfig(),
        )

        return experiment, adapter, experiment_data

    def test_scalarized_objective_soo_adds_scalarized_constraint(self) -> None:
        """Test that ScalarizedObjective in SOO config adds a single
        ScalarizedOutcomeConstraint with bound = scalarized SQ value."""
        # m3 >= 100.0 constraint — no point satisfies this.
        # SQ has m1=1.0, m2=2.0. Weights are [1.0, 1.0], minimize=False.
        # Scalarized SQ value = 1.0*1.0 + 1.0*2.0 = 3.0.
        # Expect a single ScalarizedOutcomeConstraint with bound=3.0, op=GEQ.
        _, adapter, experiment_data = self._make_scalarized_experiment_adapter_and_data(
            observations=[[1.0, 2.0, 0.5], [3.0, 4.0, 5.0]],
            constraint_bound=100.0,
            constraint_op=ComparisonOp.GEQ,
            minimize=False,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        self.assertTrue(t._should_add_constraint)

        opt_config = deepcopy(adapter._experiment.optimization_config)
        assert opt_config is not None
        transformed = t.transform_optimization_config(opt_config, adapter)
        # Original constraint on m3 + 1 new ScalarizedOutcomeConstraint.
        self.assertEqual(len(transformed.outcome_constraints), 2)

        scalarized_constraints = [
            c
            for c in transformed.outcome_constraints
            if isinstance(c, ScalarizedOutcomeConstraint)
        ]
        self.assertEqual(len(scalarized_constraints), 1)
        sc = scalarized_constraints[0]
        self.assertEqual(sc.op, ComparisonOp.GEQ)
        self.assertAlmostEqual(sc.bound, 3.0)
        self.assertEqual([m.name for m in sc.metrics], ["m1", "m2"])
        self.assertEqual(sc.weights, [1.0, 1.0])
        self.assertFalse(sc.relative)

    def test_scalarized_objective_soo_minimize(self) -> None:
        """Test ScalarizedObjective with minimize=True adds LEQ constraint."""
        _, adapter, experiment_data = self._make_scalarized_experiment_adapter_and_data(
            observations=[[1.0, 2.0, 0.5], [3.0, 4.0, 5.0]],
            constraint_bound=100.0,
            constraint_op=ComparisonOp.GEQ,
            minimize=True,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        self.assertTrue(t._should_add_constraint)

        opt_config = deepcopy(adapter._experiment.optimization_config)
        assert opt_config is not None
        transformed = t.transform_optimization_config(opt_config, adapter)

        scalarized_constraints = [
            c
            for c in transformed.outcome_constraints
            if isinstance(c, ScalarizedOutcomeConstraint)
        ]
        self.assertEqual(len(scalarized_constraints), 1)
        sc = scalarized_constraints[0]
        self.assertEqual(sc.op, ComparisonOp.LEQ)
        # SQ value = 1.0*1.0 + 1.0*2.0 = 3.0
        self.assertAlmostEqual(sc.bound, 3.0)

    def test_scalarized_objective_with_weights(self) -> None:
        """Test that custom weights affect the scalarized SQ bound."""
        # Weights [2.0, -1.0], minimize=False.
        # SQ value = 2.0*1.0 + (-1.0)*2.0 = 0.0
        _, adapter, experiment_data = self._make_scalarized_experiment_adapter_and_data(
            observations=[[1.0, 2.0, 0.5], [3.0, 4.0, 5.0]],
            constraint_bound=100.0,
            constraint_op=ComparisonOp.GEQ,
            minimize=False,
            weights=[2.0, -1.0],
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        self.assertTrue(t._should_add_constraint)

        opt_config = deepcopy(adapter._experiment.optimization_config)
        assert opt_config is not None
        transformed = t.transform_optimization_config(opt_config, adapter)

        scalarized_constraints = [
            c
            for c in transformed.outcome_constraints
            if isinstance(c, ScalarizedOutcomeConstraint)
        ]
        self.assertEqual(len(scalarized_constraints), 1)
        sc = scalarized_constraints[0]
        self.assertEqual(sc.op, ComparisonOp.GEQ)
        self.assertAlmostEqual(sc.bound, 0.0)
        self.assertEqual(sc.weights, [2.0, -1.0])

    def test_scalarized_objective_moo_config(self) -> None:
        """Test ScalarizedObjective in MultiObjectiveOptimizationConfig."""
        _, adapter, experiment_data = self._make_scalarized_experiment_adapter_and_data(
            observations=[[1.0, 2.0, 0.5], [3.0, 4.0, 5.0]],
            constraint_bound=100.0,
            constraint_op=ComparisonOp.GEQ,
            minimize=False,
            use_moo_config=True,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        self.assertTrue(t._should_add_constraint)

        opt_config = deepcopy(adapter._experiment.optimization_config)
        assert opt_config is not None
        transformed = t.transform_optimization_config(opt_config, adapter)
        # Original constraint on m3 + 1 new ScalarizedOutcomeConstraint.
        self.assertEqual(len(transformed.outcome_constraints), 2)

        scalarized_constraints = [
            c
            for c in transformed.outcome_constraints
            if isinstance(c, ScalarizedOutcomeConstraint)
        ]
        self.assertEqual(len(scalarized_constraints), 1)
        sc = scalarized_constraints[0]
        self.assertEqual(sc.op, ComparisonOp.GEQ)
        self.assertAlmostEqual(sc.bound, 3.0)

    def test_scalarized_objective_untransform(self) -> None:
        """Test that untransform removes the added ScalarizedOutcomeConstraint."""
        _, adapter, experiment_data = self._make_scalarized_experiment_adapter_and_data(
            observations=[[1.0, 2.0, 0.5], [3.0, 4.0, 5.0]],
            constraint_bound=100.0,
            constraint_op=ComparisonOp.GEQ,
            minimize=False,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        opt_config = deepcopy(adapter._experiment.optimization_config)
        assert opt_config is not None
        transformed = t.transform_optimization_config(opt_config, adapter)
        self.assertEqual(len(transformed.outcome_constraints), 2)

        untransformed = t.untransform_outcome_constraints(
            outcome_constraints=transformed.outcome_constraints,
        )
        self.assertEqual(len(untransformed), 1)
        self.assertEqual(untransformed[0].metric.name, "m3")

    def test_scalarized_objective_no_op_when_feasible(self) -> None:
        """Test no-op with ScalarizedObjective when feasible points exist."""
        # m3 >= 1.0 — observation [3.0, 4.0, 5.0] has m3=5.0 >= 1.0 → feasible.
        _, adapter, experiment_data = self._make_scalarized_experiment_adapter_and_data(
            observations=[[1.0, 2.0, 0.5], [3.0, 4.0, 5.0]],
            constraint_bound=1.0,
            constraint_op=ComparisonOp.GEQ,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        self.assertFalse(t._should_add_constraint)


class ObjectiveAsConstraintMOOTest(TestCase):
    """Tests for multi-objective optimization support in ObjectiveAsConstraint."""

    def _make_moo_experiment_adapter_and_data(
        self,
        observations: list[list[float]],
        constraint_bound: float | None = None,
        constraint_op: ComparisonOp = ComparisonOp.GEQ,
        objective_thresholds: list[tuple[float, bool]] | None = None,
        minimize_objs: tuple[bool, bool] = (False, False),
        relative_constraint: bool = False,
    ) -> tuple[Experiment, Adapter, ExperimentData]:
        """Helper to create an MOO experiment, adapter, and experiment data.

        Creates a multi-objective experiment with objectives on m1 and m2,
        and optionally a constraint on m3 and/or objective thresholds.

        Args:
            observations: List of [m1_value, m2_value, m3_value] observations.
                The first observation is the status quo.
            constraint_bound: Optional bound for the outcome constraint on m3.
            constraint_op: Comparison op for the constraint.
            objective_thresholds: Optional list of (bound, relative) tuples for
                objective thresholds on m1 and m2.
            minimize_objs: Tuple of (minimize_m1, minimize_m2).
            relative_constraint: Whether the constraint is relative.

        Returns:
            Tuple of (experiment, adapter, experiment_data).
        """
        objectives = [
            Objective(
                metric=Metric("m1", lower_is_better=minimize_objs[0]),
                minimize=minimize_objs[0],
            ),
            Objective(
                metric=Metric("m2", lower_is_better=minimize_objs[1]),
                minimize=minimize_objs[1],
            ),
        ]

        outcome_constraints = []
        if constraint_bound is not None:
            outcome_constraints.append(
                OutcomeConstraint(
                    metric=Metric("m3", lower_is_better=True),
                    op=constraint_op,
                    bound=constraint_bound,
                    relative=relative_constraint,
                )
            )

        obj_thresholds_list: list[ObjectiveThreshold] = []
        if objective_thresholds is not None:
            for i, (bound, relative) in enumerate(objective_thresholds):
                metric_name = f"m{i + 1}"
                # For maximization (minimize=False), threshold uses GEQ.
                # For minimization (minimize=True), threshold uses LEQ.
                op = ComparisonOp.LEQ if minimize_objs[i] else ComparisonOp.GEQ
                obj_thresholds_list.append(
                    ObjectiveThreshold(
                        metric=Metric(metric_name, lower_is_better=minimize_objs[i]),
                        op=op,
                        bound=bound,
                        relative=relative,
                    )
                )

        optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(objectives=objectives),
            outcome_constraints=outcome_constraints,
            objective_thresholds=obj_thresholds_list,
        )

        search_space = SearchSpace(
            parameters=[
                RangeParameter("x", ParameterType.FLOAT, 0.0, 10.0),
                RangeParameter("y", ParameterType.FLOAT, 0.0, 10.0),
            ]
        )
        sq_params = {"x": 0.0, "y": 0.0}
        parameterizations = [sq_params] + [
            {"x": float(j + 1), "y": float(j + 1)} for j in range(len(observations) - 1)
        ]

        experiment = get_experiment_with_observations(
            observations=observations,
            optimization_config=optimization_config,
            parameterizations=parameterizations,
            search_space=search_space,
            status_quo=Arm(parameters=sq_params, name="0_0"),
        )

        adapter = Adapter(experiment=experiment, generator=Generator())

        experiment_data = extract_experiment_data(
            experiment=experiment,
            data_loader_config=DataLoaderConfig(),
        )

        return experiment, adapter, experiment_data

    def test_moo_no_op_when_feasible_points_exist(self) -> None:
        """Test that MOO transform is a no-op when feasible points exist."""
        # m3 >= 1.0 constraint (no thresholds).
        # Observation [2.0, 2.0, 5.0]: m3=5.0 >= 1.0 ✓
        # There exists a feasible point.
        _, adapter, experiment_data = self._make_moo_experiment_adapter_and_data(
            observations=[[1.0, 1.0, 0.5], [2.0, 2.0, 5.0]],
            constraint_bound=1.0,
            constraint_op=ComparisonOp.GEQ,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        self.assertFalse(t._should_add_constraint)

    def test_moo_no_op_when_thresholds_specified(self) -> None:
        """Test that MOO transform is a no-op when objective thresholds are
        specified, regardless of feasibility."""
        # m3 >= 10.0 constraint (no point satisfies this), but thresholds
        # are specified so the transform should be a no-op.
        _, adapter, experiment_data = self._make_moo_experiment_adapter_and_data(
            observations=[[1.0, 1.0, 0.5], [2.0, 2.0, 5.0]],
            constraint_bound=10.0,
            constraint_op=ComparisonOp.GEQ,
            objective_thresholds=[(0.5, False), (0.5, False)],
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        # Should be a no-op because thresholds are specified.
        self.assertFalse(t._should_add_constraint)

    def test_moo_adds_constraints_when_no_feasible_points(self) -> None:
        """Test that constraints on all objectives are added when no points
        satisfy constraints and no thresholds are specified."""
        # m3 >= 10.0 constraint (no point satisfies this), no thresholds.
        # SQ has m1=1.0, m2=1.0.
        # Expect constraints: m1 >= 1.0, m2 >= 1.0.
        _, adapter, experiment_data = self._make_moo_experiment_adapter_and_data(
            observations=[[1.0, 1.0, 0.5], [2.0, 2.0, 5.0]],
            constraint_bound=10.0,
            constraint_op=ComparisonOp.GEQ,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        self.assertTrue(t._should_add_constraint)

        opt_config = deepcopy(adapter._experiment.optimization_config)
        assert opt_config is not None
        transformed = t.transform_optimization_config(opt_config, adapter)
        # Original constraint + 2 new constraints for m1 and m2.
        self.assertEqual(len(transformed.outcome_constraints), 3)

        # Check the new constraints.
        m1_constraint = next(
            c for c in transformed.outcome_constraints if c.metric.name == "m1"
        )
        self.assertEqual(m1_constraint.op, ComparisonOp.GEQ)
        self.assertEqual(m1_constraint.bound, 1.0)
        self.assertFalse(m1_constraint.relative)

        m2_constraint = next(
            c for c in transformed.outcome_constraints if c.metric.name == "m2"
        )
        self.assertEqual(m2_constraint.op, ComparisonOp.GEQ)
        self.assertEqual(m2_constraint.bound, 1.0)
        self.assertFalse(m2_constraint.relative)

    def test_moo_adds_leq_constraints_when_minimizing(self) -> None:
        """Test that LEQ constraints are added when objectives are minimized."""
        # Both objectives are minimized. SQ has m1=1.0, m2=2.0.
        # Expect constraints: m1 <= 1.0, m2 <= 2.0.
        _, adapter, experiment_data = self._make_moo_experiment_adapter_and_data(
            observations=[[1.0, 2.0, 0.5], [3.0, 4.0, 5.0]],
            constraint_bound=10.0,
            constraint_op=ComparisonOp.GEQ,
            minimize_objs=(True, True),
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        self.assertTrue(t._should_add_constraint)

        opt_config = deepcopy(adapter._experiment.optimization_config)
        assert opt_config is not None
        transformed = t.transform_optimization_config(opt_config, adapter)

        m1_constraint = next(
            c for c in transformed.outcome_constraints if c.metric.name == "m1"
        )
        self.assertEqual(m1_constraint.op, ComparisonOp.LEQ)
        self.assertEqual(m1_constraint.bound, 1.0)

        m2_constraint = next(
            c for c in transformed.outcome_constraints if c.metric.name == "m2"
        )
        self.assertEqual(m2_constraint.op, ComparisonOp.LEQ)
        self.assertEqual(m2_constraint.bound, 2.0)

    def test_moo_no_op_without_constraints(self) -> None:
        """Test that MOO transform is a no-op without constraints."""
        # No constraints and no thresholds - should be a no-op.
        # Only 2 values per observation since there's no m3 constraint.
        _, adapter, experiment_data = self._make_moo_experiment_adapter_and_data(
            observations=[[1.0, 1.0], [2.0, 2.0]],
            constraint_bound=None,
            objective_thresholds=None,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        # No constraints to check, so should be a no-op.
        self.assertFalse(t._should_add_constraint)

    def test_moo_untransform_removes_added_constraints(self) -> None:
        """Test that untransform removes all added objective constraints."""
        _, adapter, experiment_data = self._make_moo_experiment_adapter_and_data(
            observations=[[1.0, 1.0, 0.5], [2.0, 2.0, 5.0]],
            constraint_bound=10.0,
            constraint_op=ComparisonOp.GEQ,
        )

        t = ObjectiveAsConstraint(
            search_space=adapter._experiment.search_space,
            experiment_data=experiment_data,
            adapter=adapter,
        )

        opt_config = deepcopy(adapter._experiment.optimization_config)
        assert opt_config is not None
        transformed = t.transform_optimization_config(opt_config, adapter)
        self.assertEqual(len(transformed.outcome_constraints), 3)

        # Untransform should remove the added constraints for m1 and m2.
        untransformed = t.untransform_outcome_constraints(
            outcome_constraints=transformed.outcome_constraints,
        )
        self.assertEqual(len(untransformed), 1)
        self.assertEqual(untransformed[0].metric.name, "m3")
