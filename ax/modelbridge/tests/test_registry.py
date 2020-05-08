#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.registry import (
    Cont_X_trans,
    Models,
    Y_trans,
    get_model_from_generator_run,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.models.base import Model
from ax.models.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ax.models.discrete.thompson import ThompsonSampler
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data,
    get_branin_experiment,
    get_branin_optimization_config,
    get_factorial_experiment,
)
from torch import device as torch_device, float64 as torch_float64


class ModelRegistryTest(TestCase):
    def test_enum_sobol_GPEI(self):
        """Tests Soboland GPEI instantiation through the Models enum."""
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
                    "value": f"{botorch_defaults}.predict_from_model",
                },
                "best_point_recommender": {
                    "is_callable_as_path": True,
                    "value": f"{botorch_defaults}.recommend_best_observed_point",
                },
                "refit_on_cv": False,
                "refit_on_update": True,
                "warm_start_refitting": True,
            },
        )
        self.assertEqual(
            gpei._bridge_kwargs,
            {
                "transform_configs": None,
                "torch_dtype": torch_float64,
                "torch_device": torch_device(type="cpu"),
                "status_quo_name": None,
                "status_quo_features": None,
                "optimization_config": None,
                "transforms": Cont_X_trans + Y_trans,
                "fit_out_of_design": False,
            },
        )
        gpei = Models.GPEI(
            experiment=exp, data=exp.fetch_data(), search_space=exp.search_space
        )
        self.assertIsInstance(gpei, TorchModelBridge)

    def test_enum_model_kwargs(self):
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

    def test_enum_factorial(self):
        """Tests factorial instantiation through the Models enum."""
        exp = get_factorial_experiment()
        factorial = Models.FACTORIAL(exp.search_space)
        self.assertIsInstance(factorial, DiscreteModelBridge)
        factorial_run = factorial.gen(n=-1)
        self.assertEqual(len(factorial_run.arms), 24)

    def test_enum_empirical_bayes_thompson(self):
        """Tests EB/TS instantiation through the Models enum."""
        exp = get_factorial_experiment()
        factorial = Models.FACTORIAL(exp.search_space)
        self.assertIsInstance(factorial, DiscreteModelBridge)
        factorial_run = factorial.gen(n=-1)
        exp.new_batch_trial().add_generator_run(factorial_run).run()
        data = exp.fetch_data()
        eb_thompson = Models.EMPIRICAL_BAYES_THOMPSON(
            experiment=exp, data=data, min_weight=0.0
        )
        self.assertIsInstance(eb_thompson, DiscreteModelBridge)
        self.assertIsInstance(eb_thompson.model, EmpiricalBayesThompsonSampler)
        thompson_run = eb_thompson.gen(n=5)
        self.assertEqual(len(thompson_run.arms), 5)

    def test_enum_thompson(self):
        """Tests TS instantiation through the Models enum."""
        exp = get_factorial_experiment()
        factorial = Models.FACTORIAL(exp.search_space)
        self.assertIsInstance(factorial, DiscreteModelBridge)
        factorial_run = factorial.gen(n=-1)
        exp.new_batch_trial().add_generator_run(factorial_run).run()
        data = exp.fetch_data()
        thompson = Models.THOMPSON(experiment=exp, data=data)
        self.assertIsInstance(thompson.model, ThompsonSampler)

    def test_enum_uniform(self):
        """Tests uniform random instantiation through the Models enum."""
        exp = get_branin_experiment()
        uniform = Models.UNIFORM(exp.search_space)
        self.assertIsInstance(uniform, RandomModelBridge)
        uniform_run = uniform.gen(n=5)
        self.assertEqual(len(uniform_run.arms), 5)

    def test_view_defaults(self):
        """Checks that kwargs are correctly constructed from default kwargs +
        standard kwargs."""
        self.assertEqual(
            Models.SOBOL.view_defaults(),
            (
                {
                    "seed": None,
                    "deduplicate": False,
                    "init_position": 0,
                    "scramble": True,
                    "generated_points": None,
                },
                {
                    "optimization_config": None,
                    "transforms": Cont_X_trans,
                    "transform_configs": None,
                    "status_quo_name": None,
                    "status_quo_features": None,
                    "fit_out_of_design": False,
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
                ]
            ),
        )

    def test_get_model_from_generator_run(self):
        """Tests that it is possible to restore a model from a generator run it
        produced, if `Models` registry was used.
        """
        exp = get_branin_experiment()
        initial_sobol = Models.SOBOL(experiment=exp, seed=239)
        gr = initial_sobol.gen(n=1)
        # Restore the model as it was before generation.
        sobol = get_model_from_generator_run(
            generator_run=gr, experiment=exp, data=exp.fetch_data()
        )
        self.assertEqual(sobol.model.init_position, 0)
        self.assertEqual(sobol.model.seed, 239)
        # Restore the model as it was after generation (to resume generation).
        sobol_after_gen = get_model_from_generator_run(
            generator_run=gr, experiment=exp, data=exp.fetch_data(), after_gen=True
        )
        self.assertEqual(sobol_after_gen.model.init_position, 1)
        self.assertEqual(sobol_after_gen.model.seed, 239)
        self.assertEqual(initial_sobol.gen(n=1).arms, sobol_after_gen.gen(n=1).arms)
        exp.new_trial(generator_run=gr)
        # Check restoration of GPEI, to ensure proper restoration of callable kwargs
        gpei = Models.GPEI(experiment=exp, data=get_branin_data())
        # Punch GPEI model + bridge kwargs into the Sobol generator run, to avoid
        # a slow call to `gpei.gen`.
        gr._model_key = "GPEI"
        gr._model_kwargs = gpei._model_kwargs
        gr._bridge_kwargs = gpei._bridge_kwargs
        gpei_restored = get_model_from_generator_run(
            gr, experiment=exp, data=get_branin_data()
        )
        for key in gpei.__dict__:
            self.assertIn(key, gpei_restored.__dict__)
            original, restored = gpei.__dict__[key], gpei_restored.__dict__[key]
            # Fit times are set in instantiation so not same and model compared below
            if key in ["fit_time", "fit_time_since_gen", "model"]:
                continue  # Fit times are set in instantiation so won't be same
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
