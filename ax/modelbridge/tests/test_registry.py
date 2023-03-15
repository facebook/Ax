#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from ax.core.data import Data
from ax.core.observation import ObservationFeatures
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.core.types import ComparisonOp
from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.registry import (
    Cont_X_trans,
    get_model_from_generator_run,
    MODEL_KEY_TO_MODEL_SETUP,
    Models,
    Y_trans,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.models.base import Model
from ax.models.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ax.models.discrete.thompson import ThompsonSampler
from ax.models.random.alebo_initializer import ALEBOInitializer
from ax.models.torch.alebo import ALEBO
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.utils.common.constants import Keys
from ax.utils.common.kwargs import get_function_argument_names
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data,
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
    get_branin_optimization_config,
    get_factorial_experiment,
)
from ax.utils.testing.mock import fast_botorch_optimize
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models.gp_regression import FixedNoiseGP
from gpytorch.priors.torch_priors import GammaPrior


class ModelRegistryTest(TestCase):
    @fast_botorch_optimize
    def test_botorch_modular(self) -> None:
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run()
        gpei = Models.BOTORCH_MODULAR(
            # Model kwargs
            acquisition_class=Acquisition,
            botorch_acqf_class=qExpectedImprovement,
            acquisition_options={"best_f": 0.0},
            # Model bridge kwargs
            experiment=exp,
            data=exp.fetch_data(),
        )
        self.assertIsInstance(gpei, TorchModelBridge)
        self.assertIsInstance(gpei.model, BoTorchModel)
        self.assertEqual(gpei.model.botorch_acqf_class, qExpectedImprovement)
        self.assertEqual(gpei.model.acquisition_class, Acquisition)
        self.assertEqual(gpei.model.acquisition_options, {"best_f": 0.0})
        self.assertIsInstance(gpei.model.surrogates[Keys.AUTOSET_SURROGATE], Surrogate)
        # FixedNoiseGP should be picked since experiment data has fixed noise.
        self.assertIsInstance(
            gpei.model.surrogates[Keys.AUTOSET_SURROGATE].model, FixedNoiseGP
        )

        gr = gpei.gen(n=1)
        self.assertIsNotNone(gr.best_arm_predictions)

    @fast_botorch_optimize
    def test_enum_sobol_GPEI(self) -> None:
        """Tests Sobol and GPEI instantiation through the Models enum."""
        exp = get_branin_experiment()
        # Check that factory generates a valid sobol modelbridge.
        sobol = Models.SOBOL(search_space=exp.search_space)
        self.assertIsInstance(sobol, RandomModelBridge)
        for _ in range(5):
            sobol_run = sobol.gen(n=1)
            self.assertEqual(sobol_run._model_key, "Sobol")
            exp.new_batch_trial().add_generator_run(sobol_run).run()
        # Check that factory generates a valid GP+EI modelbridge.
        exp.optimization_config = get_branin_optimization_config()
        gpei = Models.GPEI(experiment=exp, data=exp.fetch_data())
        self.assertIsInstance(gpei, TorchModelBridge)
        self.assertEqual(gpei._model_key, "GPEI")
        botorch_defaults = "ax.models.torch.botorch_defaults"
        # Check that the callable kwargs and the torch kwargs were recorded.
        self.assertEqual(
            gpei._model_kwargs,
            {
                "acqf_constructor": {
                    "is_callable_as_path": True,
                    "value": f"{botorch_defaults}.get_NEI",
                },
                "acqf_optimizer": {
                    "is_callable_as_path": True,
                    "value": f"{botorch_defaults}.scipy_optimizer",
                },
                "model_constructor": {
                    "is_callable_as_path": True,
                    "value": f"{botorch_defaults}.get_and_fit_model",
                },
                "model_predictor": {
                    "is_callable_as_path": True,
                    "value": "ax.models.torch.utils.predict_from_model",
                },
                "best_point_recommender": {
                    "is_callable_as_path": True,
                    "value": f"{botorch_defaults}.recommend_best_observed_point",
                },
                "refit_on_cv": False,
                "refit_on_update": True,
                "warm_start_refitting": True,
                "use_input_warping": False,
                "use_loocv_pseudo_likelihood": False,
                "prior": None,
            },
        )
        self.assertEqual(
            gpei._bridge_kwargs,
            {
                "transform_configs": None,
                "torch_dtype": torch.float64,
                "torch_device": None,
                "status_quo_name": None,
                "status_quo_features": None,
                "optimization_config": None,
                "transforms": Cont_X_trans + Y_trans,
                "fit_out_of_design": False,
                "fit_abandoned": False,
                "default_model_gen_options": None,
            },
        )
        prior_kwargs = {"lengthscale_prior": GammaPrior(6.0, 6.0)}
        gpei = Models.GPEI(
            experiment=exp,
            data=exp.fetch_data(),
            search_space=exp.search_space,
            prior=prior_kwargs,
        )
        self.assertIsInstance(gpei, TorchModelBridge)
        self.assertEqual(
            gpei._model_kwargs["prior"],  # pyre-ignore
            prior_kwargs,
        )

    def test_enum_model_kwargs(self) -> None:
        """Tests that kwargs are passed correctly when instantiating through the
        Models enum."""
        exp = get_branin_experiment()
        sobol = Models.SOBOL(
            search_space=exp.search_space, init_position=2, scramble=False, seed=239
        )
        self.assertIsInstance(sobol, RandomModelBridge)
        for _ in range(5):
            sobol_run = sobol.gen(1)
            exp.new_batch_trial().add_generator_run(sobol_run).run()

    def test_enum_factorial(self) -> None:
        """Tests factorial instantiation through the Models enum."""
        exp = get_factorial_experiment()
        factorial = Models.FACTORIAL(exp.search_space)
        self.assertIsInstance(factorial, DiscreteModelBridge)
        factorial_run = factorial.gen(n=-1)
        self.assertEqual(len(factorial_run.arms), 24)

    def test_enum_empirical_bayes_thompson(self) -> None:
        """Tests EB/TS instantiation through the Models enum."""
        exp = get_factorial_experiment()
        factorial = Models.FACTORIAL(exp.search_space)
        self.assertIsInstance(factorial, DiscreteModelBridge)
        factorial_run = factorial.gen(n=-1)
        exp.new_batch_trial().add_generator_run(factorial_run).run().mark_completed()
        data = exp.fetch_data()
        eb_thompson = Models.EMPIRICAL_BAYES_THOMPSON(
            experiment=exp, data=data, min_weight=0.0
        )
        self.assertIsInstance(eb_thompson, DiscreteModelBridge)
        self.assertIsInstance(eb_thompson.model, EmpiricalBayesThompsonSampler)
        thompson_run = eb_thompson.gen(n=5)
        self.assertEqual(len(thompson_run.arms), 5)

    def test_enum_thompson(self) -> None:
        """Tests TS instantiation through the Models enum."""
        exp = get_factorial_experiment()
        factorial = Models.FACTORIAL(exp.search_space)
        self.assertIsInstance(factorial, DiscreteModelBridge)
        factorial_run = factorial.gen(n=-1)
        exp.new_batch_trial().add_generator_run(factorial_run).run().mark_completed()
        data = exp.fetch_data()
        thompson = Models.THOMPSON(experiment=exp, data=data)
        self.assertIsInstance(thompson.model, ThompsonSampler)

    def test_enum_uniform(self) -> None:
        """Tests uniform random instantiation through the Models enum."""
        exp = get_branin_experiment()
        uniform = Models.UNIFORM(exp.search_space)
        self.assertIsInstance(uniform, RandomModelBridge)
        uniform_run = uniform.gen(n=5)
        self.assertEqual(len(uniform_run.arms), 5)

    def test_view_defaults(self) -> None:
        """Checks that kwargs are correctly constructed from default kwargs +
        standard kwargs."""
        self.assertEqual(
            Models.SOBOL.view_defaults(),
            (
                {
                    "seed": None,
                    "deduplicate": True,
                    "init_position": 0,
                    "scramble": True,
                    "generated_points": None,
                    "fallback_to_sample_polytope": False,
                },
                {
                    "optimization_config": None,
                    "transforms": Cont_X_trans,
                    "transform_configs": None,
                    "status_quo_name": None,
                    "status_quo_features": None,
                    "fit_out_of_design": False,
                    "fit_abandoned": False,
                },
            ),
        )
        self.assertTrue(
            all(
                kw in Models.SOBOL.view_kwargs()[0]
                for kw in ["seed", "deduplicate", "init_position", "scramble"]
            ),
            all(
                kw in Models.SOBOL.view_kwargs()[1]
                for kw in [
                    "search_space",
                    "model",
                    "transforms",
                    "experiment",
                    "data",
                    "transform_configs",
                    "status_quo_name",
                    "status_quo_features",
                    "fit_out_of_design",
                    "fit_abandoned",
                ]
            ),
        )

    @fast_botorch_optimize
    def test_get_model_from_generator_run(self) -> None:
        """Tests that it is possible to restore a model from a generator run it
        produced, if `Models` registry was used.
        """
        exp = get_branin_experiment()
        initial_sobol = Models.SOBOL(experiment=exp, seed=239)
        gr = initial_sobol.gen(n=1)
        # Restore the model as it was before generation.
        sobol = get_model_from_generator_run(
            generator_run=gr,
            experiment=exp,
            data=exp.fetch_data(),
            models_enum=Models,
            after_gen=False,
        )
        self.assertEqual(sobol.model.init_position, 0)
        self.assertEqual(sobol.model.seed, 239)
        # Restore the model as it was after generation (to resume generation).
        sobol_after_gen = get_model_from_generator_run(
            generator_run=gr,
            experiment=exp,
            data=exp.fetch_data(),
            models_enum=Models,
        )
        self.assertEqual(sobol_after_gen.model.init_position, 1)
        self.assertEqual(sobol_after_gen.model.seed, 239)
        self.assertEqual(initial_sobol.gen(n=1).arms, sobol_after_gen.gen(n=1).arms)
        exp.new_trial(generator_run=gr)
        # Check restoration of GPEI, to ensure proper restoration of callable kwargs
        gpei = Models.GPEI(experiment=exp, data=get_branin_data())
        # Punch GPEI model + bridge kwargs into the Sobol generator run, to avoid
        # a slow call to `gpei.gen`, and remove Sobol's model state.
        gr._model_key = "GPEI"
        gr._model_kwargs = gpei._model_kwargs
        gr._bridge_kwargs = gpei._bridge_kwargs
        gr._model_state_after_gen = {}
        gpei_restored = get_model_from_generator_run(
            gr, experiment=exp, data=get_branin_data(), models_enum=Models
        )
        for key in gpei.__dict__:
            self.assertIn(key, gpei_restored.__dict__)
            original, restored = gpei.__dict__[key], gpei_restored.__dict__[key]
            # Fit times are set in instantiation so not same and model compared below.
            if key in ["fit_time", "fit_time_since_gen", "model", "training_data"]:
                continue  # Fit times are set in instantiation so won't be same.
            if isinstance(original, OrderedDict) and isinstance(restored, OrderedDict):
                original, restored = list(original.keys()), list(restored.keys())
            if isinstance(original, Model) and isinstance(restored, Model):
                continue  # Model equality is tough to compare.
                self.assertEqual(original, restored)

        for key in gpei.model.__dict__:
            self.assertIn(key, gpei_restored.model.__dict__)
            original, restored = (
                gpei.model.__dict__[key],
                gpei_restored.model.__dict__[key],
            )
            # Botorch model equality is tough to compare and training data
            # is unnecessary to compare, because data passed to model was the same
            if key in ["model", "warm_start_refitting", "Xs", "Ys"]:
                continue
            self.assertEqual(original, restored)

    def test_ModelSetups_do_not_share_kwargs(self) -> None:
        """Tests that none of the preset model and bridge combinations share a
        kwarg.
        """
        for model_setup_info in MODEL_KEY_TO_MODEL_SETUP.values():
            model_class = model_setup_info.model_class
            bridge_class = model_setup_info.bridge_class
            model_args = set(get_function_argument_names(model_class))
            bridge_args = set(get_function_argument_names(bridge_class))
            # Intersection of two sets should be empty
            self.assertEqual(model_args & bridge_args, set())

    @fast_botorch_optimize
    def test_ST_MTGP(self) -> None:
        """Tests single type MTGP instantiation."""
        # Test Single-type MTGP
        exp = get_branin_experiment()
        sobol = Models.SOBOL(search_space=exp.search_space)
        self.assertIsInstance(sobol, RandomModelBridge)
        for _ in range(5):
            sobol_run = sobol.gen(n=1)
            t = exp.new_batch_trial().add_generator_run(sobol_run)
            t.set_status_quo_with_weight(status_quo=t.arms[0], weight=0.5)
            t.run().mark_completed()
        status_quo_features = ObservationFeatures(
            # pyre-fixme[16]: `BaseTrial` has no attribute `status_quo`.
            parameters=exp.trials[0].status_quo.parameters,
            # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
            trial_index=0,
        )
        mtgp = Models.ST_MTGP(
            experiment=exp,
            data=exp.fetch_data(),
            status_quo_features=status_quo_features,
        )
        self.assertIsInstance(mtgp, TorchModelBridge)

        exp = get_branin_experiment()
        sobol = Models.SOBOL(search_space=exp.search_space)
        self.assertIsInstance(sobol, RandomModelBridge)
        sobol_run = sobol.gen(n=1)
        t = exp.new_batch_trial().add_generator_run(sobol_run)
        t.set_status_quo_with_weight(status_quo=t.arms[0], weight=0.5)
        t.run().mark_completed()

        with self.assertRaises(ValueError):
            status_quo_features = ObservationFeatures(
                parameters=exp.trials[0].status_quo.parameters,
                # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
                trial_index=0,
            )
            Models.ST_MTGP(
                experiment=exp,
                data=exp.fetch_data(),
                status_quo_features=status_quo_features,
            )

    @fast_botorch_optimize
    def test_ALEBO(self) -> None:
        """Tests Alebo fitting and generations"""
        experiment = get_branin_experiment(with_batch=True)
        B = np.array([[1.0, 2.0]])
        data = Data(
            pd.DataFrame(
                {
                    "arm_name": ["0_0", "0_1", "0_2"],
                    "metric_name": "y",
                    "mean": [-1.0, 0.0, 1.0],
                    "sem": 0.1,
                }
            )
        )
        m = Models.ALEBO(
            experiment=experiment,
            search_space=None,
            data=data,
            B=torch.from_numpy(B).double(),
        )
        self.assertIsInstance(m, TorchModelBridge)
        self.assertIsInstance(m.model, ALEBO)
        self.assertTrue(np.array_equal(m.model.B.numpy(), B))

    def test_ALEBO_Initializer(self) -> None:
        """Tests Alebo Initializer generations"""
        experiment = get_branin_experiment(with_batch=True)
        B = np.array([[1.0, 2.0]])
        m = Models.ALEBO_INITIALIZER(
            experiment=experiment,
            search_space=None,
            B=B,
        )
        self.assertIsInstance(m, RandomModelBridge)
        self.assertIsInstance(m.model, ALEBOInitializer)

        gr = m.gen(n=2)
        self.assertEqual(len(gr.arms), 2)

    @fast_botorch_optimize
    def test_ST_MTGP_NEHVI(self) -> None:
        """Tests single type MTGP NEHVI instantiation."""
        multi_obj_exp = get_branin_experiment_with_multi_objective(
            with_batch=True,
            with_status_quo=True,
        )
        # pyre-fixme[16]: Optional type has no attribute `objective`.
        metrics = multi_obj_exp.optimization_config.objective.metrics
        multi_objective_thresholds = [
            ObjectiveThreshold(
                metric=metrics[0], bound=0.0, relative=False, op=ComparisonOp.GEQ
            ),
            ObjectiveThreshold(
                metric=metrics[1], bound=0.0, relative=False, op=ComparisonOp.GEQ
            ),
        ]
        sobol = Models.SOBOL(search_space=multi_obj_exp.search_space)
        self.assertIsInstance(sobol, RandomModelBridge)
        for _ in range(2):
            sobol_run = sobol.gen(n=1)
            t = multi_obj_exp.new_batch_trial().add_generator_run(sobol_run)
            t.set_status_quo_with_weight(status_quo=t.arms[0], weight=0.5)
            t.run().mark_completed()
        status_quo_features = ObservationFeatures(
            # pyre-fixme[16]: `BaseTrial` has no attribute `status_quo`.
            parameters=multi_obj_exp.trials[0].status_quo.parameters,
            # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
            trial_index=0,
        )
        # pyre-fixme[16]: Optional type has no attribute `clone_with_args`.
        optimization_config = multi_obj_exp.optimization_config.clone_with_args(
            objective_thresholds=multi_objective_thresholds
        )
        mtgp = Models.ST_MTGP_NEHVI(
            experiment=multi_obj_exp,
            data=multi_obj_exp.fetch_data(),
            status_quo_features=status_quo_features,
            optimization_config=optimization_config,
        )
        self.assertIsInstance(mtgp, TorchModelBridge)
        self.assertIsInstance(mtgp.model, MultiObjectiveBotorchModel)

        # test it can generate
        mtgp_run = mtgp.gen(
            n=1,
            # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
            fixed_features=ObservationFeatures(parameters={}, trial_index=1),
        )
        self.assertEqual(len(mtgp_run.arms), 1)

        # test a generated trial can be completed
        t = multi_obj_exp.new_batch_trial().add_generator_run(mtgp_run)
        t.set_status_quo_with_weight(status_quo=t.arms[0], weight=0.5)
        t.run().mark_completed()
