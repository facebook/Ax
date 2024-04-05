#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest import mock

import pandas as pd
import torch
from ax.core.data import Data
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import ComparisonOp, ObjectiveThreshold
from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.factory import (
    get_botorch,
    get_empirical_bayes_thompson,
    get_factorial,
    get_GPEI,
    get_MOO_EHVI,
    get_MOO_NEHVI,
    get_MOO_PAREGO,
    get_MOO_RS,
    get_MTGP_LEGACY,
    get_MTGP_NEHVI,
    get_MTGP_PAREGO,
    get_sobol,
    get_thompson,
    get_uniform,
)
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.torch import TorchModelBridge
from ax.models.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ax.models.discrete.thompson import ThompsonSampler
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
    get_branin_optimization_config,
    get_factorial_experiment,
    get_multi_type_experiment,
    get_multi_type_experiment_with_multi_objective,
)
from ax.utils.testing.mock import fast_botorch_optimize
from botorch.models.multitask import MultiTaskGP
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list


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
    @fast_botorch_optimize
    def test_sobol_GPEI(self) -> None:
        """Tests sobol + GPEI instantiation."""
        exp = get_branin_experiment()
        # Check that factory generates a valid sobol modelbridge.
        sobol = get_sobol(search_space=exp.search_space)
        self.assertIsInstance(sobol, RandomModelBridge)
        for _ in range(5):
            sobol_run = sobol.gen(n=1)
            exp.new_batch_trial().add_generator_run(sobol_run).run().mark_completed()
        # Check that factory generates a valid GP+EI modelbridge.
        exp.optimization_config = get_branin_optimization_config()
        gpei = get_GPEI(experiment=exp, data=exp.fetch_data())
        self.assertIsInstance(gpei, TorchModelBridge)
        gpei = get_GPEI(
            experiment=exp, data=exp.fetch_data(), search_space=exp.search_space
        )
        self.assertIsInstance(gpei, TorchModelBridge)
        botorch = get_botorch(experiment=exp, data=exp.fetch_data())
        self.assertIsInstance(botorch, TorchModelBridge)

        # Check that .gen returns without failure
        gpei_run = gpei.gen(n=1)
        self.assertEqual(len(gpei_run.arms), 1)

    @fast_botorch_optimize
    def test_MTGP_LEGACY(self) -> None:
        """Tests MTGP instantiation."""
        # Test Multi-type MTGP
        exp = get_multi_type_experiment(add_trials=True)
        mtgp = get_MTGP_LEGACY(experiment=exp, data=exp.fetch_data())
        self.assertIsInstance(mtgp, TorchModelBridge)

        # Test Single-type MTGP
        exp = get_branin_experiment()
        # Check that factory generates a valid sobol modelbridge.
        sobol = get_sobol(search_space=exp.search_space)
        self.assertIsInstance(sobol, RandomModelBridge)
        for _ in range(5):
            sobol_run = sobol.gen(n=1)
            t = exp.new_batch_trial().add_generator_run(sobol_run)
            t.set_status_quo_with_weight(status_quo=t.arms[0], weight=0.5)
            t.run().mark_completed()
        mtgp = get_MTGP_LEGACY(experiment=exp, data=exp.fetch_data(), trial_index=0)
        self.assertIsInstance(mtgp, TorchModelBridge)
        # mtgp_run = mtgp.gen(
        #     n=1
        # )  # TODO[T110948251]: This is broken at the ChoiceEncode level

        with self.assertRaises(ValueError):
            get_MTGP_LEGACY(experiment=exp, data=exp.fetch_data(), trial_index=9)

        exp = get_branin_experiment()
        sobol = get_sobol(search_space=exp.search_space)
        self.assertIsInstance(sobol, RandomModelBridge)
        sobol_run = sobol.gen(n=1)
        t = exp.new_batch_trial().add_generator_run(sobol_run)
        t.run().mark_completed()

        with self.assertRaises(ValueError):
            get_MTGP_LEGACY(experiment=exp, data=exp.fetch_data(), trial_index=0)

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


class ModelBridgeFactoryTestMultiObjective(TestCase):
    # pyre-fixme[2]: Parameter must be annotated.
    def test_single_objective_error(self, factory_fn=get_MOO_RS) -> None:
        single_obj_exp = get_branin_experiment(with_batch=True)
        with self.assertRaises(ValueError):
            factory_fn(
                experiment=single_obj_exp,
                data=single_obj_exp.fetch_data(),
            )

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def test_data_error_and_get_multi_obj_exp(self, factory_fn=get_MOO_RS):
        multi_obj_exp = get_branin_experiment_with_multi_objective(with_batch=True)
        with self.assertRaises(ValueError):
            factory_fn(experiment=multi_obj_exp, data=multi_obj_exp.fetch_data())

        multi_obj_exp.trials[0].run().mark_completed()
        return multi_obj_exp

    @fast_botorch_optimize
    def test_MOO_RS(self) -> None:
        self.test_single_objective_error(get_MOO_RS)
        multi_obj_exp = self.test_data_error_and_get_multi_obj_exp(get_MOO_RS)
        moo_rs = get_MOO_RS(experiment=multi_obj_exp, data=multi_obj_exp.fetch_data())
        self.assertIsInstance(moo_rs, TorchModelBridge)
        self.assertEqual(
            {
                "acquisition_function_kwargs": {
                    "random_scalarization": True,
                }
            },
            moo_rs._default_model_gen_options,
        )
        with mock.patch(
            "ax.models.torch.botorch_moo_defaults.optimize_acqf_list",
            wraps=optimize_acqf_list,
        ) as mock_optimize_acqf_list:
            moo_rs_run = moo_rs.gen(n=2)
        mock_optimize_acqf_list.assert_called()
        self.assertEqual(len(moo_rs_run.arms), 2)

    @fast_botorch_optimize
    def test_MOO_PAREGO(self) -> None:
        self.test_single_objective_error(get_MOO_PAREGO)
        multi_obj_exp = self.test_data_error_and_get_multi_obj_exp(get_MOO_PAREGO)
        moo_parego = get_MOO_PAREGO(
            experiment=multi_obj_exp, data=multi_obj_exp.fetch_data()
        )
        self.assertIsInstance(moo_parego, TorchModelBridge)
        self.assertEqual(
            {
                "acquisition_function_kwargs": {
                    "chebyshev_scalarization": True,
                }
            },
            moo_parego._default_model_gen_options,
        )
        with mock.patch(
            "ax.models.torch.botorch_moo.infer_objective_thresholds"
        ) as mock_infer_ot:
            moo_parego_run = moo_parego.gen(n=2)
            mock_infer_ot.assert_not_called()
        self.assertEqual(len(moo_parego_run.arms), 2)

    @fast_botorch_optimize
    def test_MOO_EHVI(self) -> None:
        self.test_single_objective_error(get_MOO_EHVI)
        multi_obj_exp, optimization_config = get_multi_obj_exp_and_opt_config()
        with self.assertRaisesRegex(
            ValueError,
            "MultiObjectiveOptimization requires non-empty data.",
        ):
            get_MOO_EHVI(
                experiment=multi_obj_exp,
                data=multi_obj_exp.fetch_data(),
                optimization_config=optimization_config,
            )

        multi_obj_exp.trials[0].run().mark_completed()
        moo_ehvi = get_MOO_EHVI(
            experiment=multi_obj_exp,
            data=multi_obj_exp.fetch_data(),
            optimization_config=optimization_config,
        )
        self.assertIsInstance(moo_ehvi, TorchModelBridge)
        with mock.patch(
            "ax.models.torch.botorch_defaults.optimize_acqf", wraps=optimize_acqf
        ) as mock_optimize_acqf:
            moo_ehvi_run = moo_ehvi.gen(n=1)
        self.assertEqual(len(moo_ehvi_run.arms), 1)
        mock_optimize_acqf.assert_called_once()
        self.assertTrue(mock_optimize_acqf.call_args.kwargs["sequential"])

    @fast_botorch_optimize
    def test_MTGP_PAREGO(self) -> None:
        """Tests MTGP ParEGO instantiation."""
        self.test_single_objective_error(get_MTGP_PAREGO)
        multi_obj_exp, optimization_config = get_multi_obj_exp_and_opt_config()
        with self.assertRaises(ValueError):
            get_MTGP_PAREGO(
                experiment=multi_obj_exp,
                data=multi_obj_exp.fetch_data(),
                optimization_config=optimization_config,
            )

        multi_obj_exp.trials[0].run()
        sobol_generator = get_sobol(search_space=multi_obj_exp.search_space)
        sobol_run = sobol_generator.gen(n=3)
        multi_obj_exp.new_batch_trial(optimize_for_power=False).add_generator_run(
            sobol_run
        )
        multi_obj_exp.trials[1].run()
        mt_ehvi = get_MTGP_PAREGO(
            experiment=multi_obj_exp,
            data=multi_obj_exp.fetch_data(),
            trial_index=1,
            optimization_config=optimization_config,
        )
        self.assertIsInstance(mt_ehvi, TorchModelBridge)
        # pyre-fixme[16]: Optional type has no attribute `model`.
        self.assertIsInstance(mt_ehvi.model.model.models[0], MultiTaskGP)
        task_covar_factor = mt_ehvi.model.model.models[0].task_covar_module.covar_factor
        self.assertEqual(task_covar_factor.shape, torch.Size([2, 2]))
        with mock.patch(
            "ax.models.torch.botorch_moo_defaults.optimize_acqf_list",
            wraps=optimize_acqf_list,
        ) as mock_optimize_acqf_list:
            mt_ehvi_run = mt_ehvi.gen(
                n=1,
                fixed_features=ObservationFeatures(parameters={}, trial_index=1),
            )
        self.assertEqual(len(mt_ehvi_run.arms), 1)
        mock_optimize_acqf_list.assert_called_once()

        # Bad index given
        with self.assertRaises(ValueError):
            get_MTGP_PAREGO(
                experiment=multi_obj_exp,
                data=multi_obj_exp.fetch_data(),
                trial_index=999,
                optimization_config=optimization_config,
            )

        # Multi-type + multi-objective experiment
        multi_type_multi_obj_exp = get_multi_type_experiment_with_multi_objective(
            add_trials=True
        )
        data = multi_type_multi_obj_exp.fetch_data()
        mt_ehvi = get_MTGP_PAREGO(
            experiment=multi_type_multi_obj_exp,
            data=data,
            optimization_config=optimization_config,
        )

    @fast_botorch_optimize
    def test_MOO_NEHVI(self) -> None:
        self.test_single_objective_error(get_MOO_NEHVI)
        multi_obj_exp, optimization_config = get_multi_obj_exp_and_opt_config()
        with self.assertRaises(ValueError):
            get_MOO_NEHVI(
                experiment=multi_obj_exp,
                data=multi_obj_exp.fetch_data(),
                optimization_config=optimization_config,
            )

        multi_obj_exp.trials[0].run()
        moo_ehvi = get_MOO_NEHVI(
            experiment=multi_obj_exp,
            data=multi_obj_exp.fetch_data(),
            optimization_config=optimization_config,
        )
        self.assertIsInstance(moo_ehvi, TorchModelBridge)
        with mock.patch(
            "ax.models.torch.botorch_defaults.optimize_acqf", wraps=optimize_acqf
        ) as mock_optimize_acqf:
            moo_ehvi_run = moo_ehvi.gen(n=1)
        self.assertEqual(len(moo_ehvi_run.arms), 1)
        mock_optimize_acqf.assert_called_once()
        self.assertTrue(mock_optimize_acqf.call_args.kwargs["sequential"])

    @fast_botorch_optimize
    def test_MOO_with_more_outcomes_than_thresholds(self) -> None:
        experiment = get_branin_experiment_with_multi_objective(
            has_optimization_config=False
        )
        metric_c = Metric(name="c", lower_is_better=False)
        metric_a = Metric(name="a", lower_is_better=False)
        objective_thresholds = [
            ObjectiveThreshold(
                metric=metric_c,
                bound=2.0,
                relative=False,
            ),
            ObjectiveThreshold(
                metric=metric_a,
                bound=1.0,
                relative=False,
            ),
        ]
        experiment.optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                objectives=[
                    Objective(metric=metric_a),
                    Objective(metric=metric_c),
                ]
            ),
            objective_thresholds=objective_thresholds,
        )
        experiment.add_tracking_metric(Metric(name="b", lower_is_better=False))
        sobol = get_sobol(
            search_space=experiment.search_space,
        )
        sobol_run = sobol.gen(1)
        experiment.new_batch_trial().add_generator_run(sobol_run).run().mark_completed()
        data = Data(
            pd.DataFrame(
                data={
                    "arm_name": ["0_0", "0_0", "0_0"],
                    "metric_name": ["a", "b", "c"],
                    "mean": [1.0, 2.0, 3.0],
                    "trial_index": [0, 0, 0],
                    "sem": [0, 0, 0],
                }
            )
        )
        test_names_to_fns = {
            "MOO_NEHVI": get_MOO_NEHVI,
            "MOO_EHVI": get_MOO_NEHVI,
            "MOO_PAREGO": get_MOO_PAREGO,
            "MOO_RS": get_MOO_RS,
        }
        for test_name, factory_fn in test_names_to_fns.items():
            with self.subTest(test_name):
                moo_model = factory_fn(
                    experiment=experiment,
                    data=data,
                )
                moo_gr = moo_model.gen(n=1)
                # pyre-fixme[16]: Optional type has no attribute `__getitem__`.
                obj_t = moo_gr.gen_metadata["objective_thresholds"]
                self.assertEqual(obj_t[0], objective_thresholds[1])
                self.assertEqual(obj_t[1], objective_thresholds[0])
                self.assertEqual(len(obj_t), 2)

    @fast_botorch_optimize
    def test_MTGP_NEHVI(self) -> None:
        self.test_single_objective_error(get_MTGP_NEHVI)
        multi_obj_exp, optimization_config = get_multi_obj_exp_and_opt_config()
        with self.assertRaises(ValueError):
            get_MTGP_NEHVI(
                experiment=multi_obj_exp,
                data=multi_obj_exp.fetch_data(),
                optimization_config=optimization_config,
            )

        multi_obj_exp.trials[0].run()
        sobol_generator = get_sobol(search_space=multi_obj_exp.search_space)
        sobol_run = sobol_generator.gen(n=3)
        multi_obj_exp.new_batch_trial(optimize_for_power=False).add_generator_run(
            sobol_run
        )
        multi_obj_exp.trials[1].run()
        mt_ehvi = get_MTGP_NEHVI(
            experiment=multi_obj_exp,
            data=multi_obj_exp.fetch_data(),
            trial_index=1,
            optimization_config=optimization_config,
        )
        self.assertIsInstance(mt_ehvi, TorchModelBridge)
        # pyre-fixme[16]: Optional type has no attribute `model`.
        self.assertIsInstance(mt_ehvi.model.model.models[0], MultiTaskGP)
        task_covar_factor = mt_ehvi.model.model.models[0].task_covar_module.covar_factor
        self.assertEqual(task_covar_factor.shape, torch.Size([2, 2]))
        with mock.patch(
            "ax.models.torch.botorch_defaults.optimize_acqf", wraps=optimize_acqf
        ) as mock_optimize_acqf:
            mt_ehvi_run = mt_ehvi.gen(
                n=1,
                fixed_features=ObservationFeatures(parameters={}, trial_index=1),
            )
        mock_optimize_acqf.assert_called_once()
        self.assertTrue(mock_optimize_acqf.call_args.kwargs["sequential"])
        self.assertEqual(len(mt_ehvi_run.arms), 1)

        # Bad index given
        with self.assertRaises(ValueError):
            get_MTGP_NEHVI(
                experiment=multi_obj_exp,
                data=multi_obj_exp.fetch_data(),
                trial_index=999,
                optimization_config=optimization_config,
            )

        # Multi-type + multi-objective experiment
        multi_type_multi_obj_exp = get_multi_type_experiment_with_multi_objective(
            add_trials=True
        )
        data = multi_type_multi_obj_exp.fetch_data()
        mt_ehvi = get_MTGP_NEHVI(
            experiment=multi_type_multi_obj_exp,
            data=data,
            optimization_config=optimization_config,
        )
