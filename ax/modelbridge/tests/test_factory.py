#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.outcome_constraint import ComparisonOp, ObjectiveThreshold
from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.factory import (
    get_empirical_bayes_thompson,
    get_factorial,
    get_sobol,
    get_thompson,
    get_uniform,
)
from ax.modelbridge.random import RandomModelBridge
from ax.models.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ax.models.discrete.thompson import ThompsonSampler
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
    get_factorial_experiment,
)


# pyre-fixme[3]: Return type must be annotated.
def get_multi_obj_exp_and_opt_config():
    multi_obj_exp = get_branin_experiment_with_multi_objective(with_batch=True)
    # pyre-fixme[16]: Optional type has no attribute `objective`.
    metrics = multi_obj_exp.optimization_config.objective.metrics
    multi_objective_thresholds = [
        ObjectiveThreshold(
            metric=metrics[0], bound=5.0, relative=False, op=ComparisonOp.LEQ
        ),
        ObjectiveThreshold(
            metric=metrics[1], bound=10.0, relative=False, op=ComparisonOp.LEQ
        ),
    ]
    # pyre-fixme[16]: Optional type has no attribute `clone_with_args`.
    optimization_config = multi_obj_exp.optimization_config.clone_with_args(
        objective_thresholds=multi_objective_thresholds
    )
    return multi_obj_exp, optimization_config


class ModelBridgeFactoryTestSingleObjective(TestCase):
    def test_model_kwargs(self) -> None:
        """Tests that model kwargs are passed correctly."""
        exp = get_branin_experiment()
        sobol = get_sobol(
            search_space=exp.search_space, init_position=2, scramble=False, seed=239
        )
        self.assertIsInstance(sobol, RandomModelBridge)
        for _ in range(5):
            sobol_run = sobol.gen(1)
            exp.new_batch_trial().add_generator_run(sobol_run).run().mark_completed()
        with self.assertRaises(TypeError):
            # pyre-fixme[28]: Unexpected keyword argument `nonexistent`.
            get_sobol(search_space=exp.search_space, nonexistent=True)

    def test_factorial(self) -> None:
        """Tests factorial instantiation."""
        exp = get_factorial_experiment()
        factorial = get_factorial(exp.search_space)
        self.assertIsInstance(factorial, DiscreteModelBridge)
        factorial_run = factorial.gen(n=-1)
        self.assertEqual(len(factorial_run.arms), 24)

    def test_empirical_bayes_thompson(self) -> None:
        """Tests EB/TS instantiation."""
        exp = get_factorial_experiment()
        factorial = get_factorial(exp.search_space)
        self.assertIsInstance(factorial, DiscreteModelBridge)
        factorial_run = factorial.gen(n=-1)
        exp.new_batch_trial().add_generator_run(factorial_run).run().mark_completed()
        data = exp.fetch_data()
        eb_thompson = get_empirical_bayes_thompson(
            experiment=exp, data=data, min_weight=0.0
        )
        self.assertIsInstance(eb_thompson, DiscreteModelBridge)
        self.assertIsInstance(eb_thompson.model, EmpiricalBayesThompsonSampler)
        thompson_run = eb_thompson.gen(n=5)
        self.assertEqual(len(thompson_run.arms), 5)

    def test_thompson(self) -> None:
        """Tests TS instantiation."""
        exp = get_factorial_experiment()
        factorial = get_factorial(exp.search_space)
        self.assertIsInstance(factorial, DiscreteModelBridge)
        factorial_run = factorial.gen(n=-1)
        exp.new_batch_trial().add_generator_run(factorial_run).run().mark_completed()
        data = exp.fetch_data()
        thompson = get_thompson(experiment=exp, data=data)
        self.assertIsInstance(thompson.model, ThompsonSampler)

    def test_uniform(self) -> None:
        exp = get_branin_experiment()
        uniform = get_uniform(exp.search_space)
        self.assertIsInstance(uniform, RandomModelBridge)
        uniform_run = uniform.gen(n=5)
        self.assertEqual(len(uniform_run.arms), 5)
