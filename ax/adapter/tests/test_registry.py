#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.adapter.discrete import DiscreteAdapter
from ax.adapter.random import RandomAdapter
from ax.adapter.registry import (
    _extract_model_state_after_gen,
    Cont_X_trans,
    Generators,
    MODEL_KEY_TO_MODEL_SETUP,
)
from ax.adapter.torch import TorchAdapter
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.exceptions.core import UserInputError
from ax.generators.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ax.generators.discrete.thompson import ThompsonSampler
from ax.generators.random.sobol import SobolGenerator
from ax.generators.torch.botorch_modular.acquisition import Acquisition
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.generators.torch.botorch_modular.kernels import ScaleMaternKernel
from ax.generators.torch.botorch_modular.surrogate import (
    ModelConfig,
    Surrogate,
    SurrogateSpec,
)
from ax.utils.common.kwargs import get_function_argument_names
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_status_quo_trials,
    get_branin_search_space,
    get_factorial_experiment,
)
from ax.utils.testing.mock import mock_botorch_optimize
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.utils.types import DEFAULT
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior
from pyre_extensions import assert_is_instance


class ModelRegistryTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.maxDiff = None

    @mock_botorch_optimize
    def test_botorch_modular(self) -> None:
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run()
        gpei = Generators.BOTORCH_MODULAR(
            # Model kwargs
            acquisition_class=Acquisition,
            botorch_acqf_class=qExpectedImprovement,
            botorch_acqf_options={"best_f": 0.0},
            # Adapter kwargs
            experiment=exp,
            data=exp.fetch_data(),
        )
        self.assertIsInstance(gpei, TorchAdapter)
        generator = assert_is_instance(gpei.generator, BoTorchGenerator)
        self.assertEqual(generator.botorch_acqf_class, qExpectedImprovement)
        self.assertEqual(generator.acquisition_class, Acquisition)
        self.assertEqual(generator._botorch_acqf_options, {"best_f": 0.0})
        self.assertIsInstance(generator.surrogate, Surrogate)
        # SingleTaskGP should be picked.
        self.assertIsInstance(generator.surrogate.model, SingleTaskGP)

        gr = gpei.gen(n=1)
        self.assertIsNotNone(gr.best_arm_predictions)

    @mock_botorch_optimize
    def test_SAASBO(self) -> None:
        exp = get_branin_experiment()
        sobol = Generators.SOBOL(experiment=exp)
        self.assertIsInstance(sobol, RandomAdapter)
        for _ in range(5):
            sobol_run = sobol.gen(n=1)
            self.assertEqual(sobol_run._model_key, "Sobol")
            exp.new_batch_trial().add_generator_run(sobol_run).run()
        saasbo = Generators.SAASBO(experiment=exp, data=exp.fetch_data())
        self.assertIsInstance(saasbo, TorchAdapter)
        self.assertEqual(saasbo._model_key, "SAASBO")
        generator = assert_is_instance(saasbo.generator, BoTorchGenerator)
        surrogate_spec = generator.surrogate_spec
        self.assertEqual(
            surrogate_spec,
            SurrogateSpec(
                model_configs=[
                    ModelConfig(
                        botorch_model_class=SaasFullyBayesianSingleTaskGP, name="SAASBO"
                    )
                ]
            ),
        )
        self.assertEqual(
            generator.surrogate.surrogate_spec.model_configs[0].botorch_model_class,
            SaasFullyBayesianSingleTaskGP,
        )

    def test_enum_model_kwargs(self) -> None:
        """Tests that kwargs are passed correctly when instantiating through the
        Generators enum."""
        exp = get_branin_experiment()
        sobol = Generators.SOBOL(
            experiment=exp, init_position=2, scramble=False, seed=239
        )
        self.assertIsInstance(sobol, RandomAdapter)
        for _ in range(5):
            sobol_run = sobol.gen(1)
            exp.new_batch_trial().add_generator_run(sobol_run).run()

    def test_enum_factorial(self) -> None:
        """Tests factorial instantiation through the Generators enum."""
        exp = get_factorial_experiment()
        factorial = Generators.FACTORIAL(experiment=exp)
        self.assertIsInstance(factorial, DiscreteAdapter)
        factorial_run = factorial.gen(n=-1)
        self.assertEqual(len(factorial_run.arms), 24)

    def test_enum_empirical_bayes_thompson(self) -> None:
        """Tests EB/TS instantiation through the Generators enum."""
        exp = get_factorial_experiment()
        factorial = Generators.FACTORIAL(experiment=exp)
        self.assertIsInstance(factorial, DiscreteAdapter)
        factorial_run = factorial.gen(n=-1)
        exp.new_batch_trial().add_generator_run(factorial_run).run().mark_completed()
        data = exp.fetch_data()
        eb_thompson = Generators.EMPIRICAL_BAYES_THOMPSON(
            experiment=exp, data=data, min_weight=0.0
        )
        self.assertIsInstance(eb_thompson, DiscreteAdapter)
        self.assertIsInstance(eb_thompson.generator, EmpiricalBayesThompsonSampler)
        thompson_run = eb_thompson.gen(n=5)
        self.assertEqual(len(thompson_run.arms), 5)

    def test_enum_thompson(self) -> None:
        """Tests TS instantiation through the Generators enum."""
        exp = get_factorial_experiment()
        factorial = Generators.FACTORIAL(experiment=exp)
        self.assertIsInstance(factorial, DiscreteAdapter)
        factorial_run = factorial.gen(n=-1)
        exp.new_batch_trial().add_generator_run(factorial_run).run().mark_completed()
        data = exp.fetch_data()
        thompson = Generators.THOMPSON(experiment=exp, data=data)
        self.assertIsInstance(thompson.generator, ThompsonSampler)

    def test_enum_uniform(self) -> None:
        """Tests uniform random instantiation through the Generators enum."""
        exp = get_branin_experiment()
        uniform = Generators.UNIFORM(experiment=exp)
        self.assertIsInstance(uniform, RandomAdapter)
        uniform_run = uniform.gen(n=5)
        self.assertEqual(len(uniform_run.arms), 5)

    def test_view_defaults(self) -> None:
        """Checks that kwargs are correctly constructed from default kwargs +
        standard kwargs."""
        self.assertEqual(
            Generators.SOBOL.view_defaults(),
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
                    "fit_abandoned": None,  # False by DataLoaderConfig default
                    "data_loader_config": None,
                    "fit_tracking_metrics": True,
                    "fit_on_init": True,
                },
            ),
        )
        self.assertTrue(
            all(
                kw in Generators.SOBOL.view_kwargs()[0]
                for kw in ["seed", "deduplicate", "init_position", "scramble"]
            ),
            all(
                kw in Generators.SOBOL.view_kwargs()[1]
                for kw in [
                    "search_space",
                    "model",
                    "transforms",
                    "experiment",
                    "data",
                    "transform_configs",
                    "expand_model_space",
                    "fit_abandoned",
                    "fit_tracking_metrics",
                    "fit_on_init",
                ]
            ),
        )

    def test_ModelSetups_do_not_share_kwargs(self) -> None:
        """Tests that none of the preset model and adapter combinations share a
        kwarg.
        """
        for model_setup_info in MODEL_KEY_TO_MODEL_SETUP.values():
            model_class = model_setup_info.model_class
            adapter_class = model_setup_info.adapter_class
            model_args = set(get_function_argument_names(model_class))
            bridge_args = set(get_function_argument_names(adapter_class))
            # Intersection of two sets should be empty
            self.assertEqual(model_args & bridge_args, set())

    @mock_botorch_optimize
    def test_ST_MTGP(self, use_saas: bool = False) -> None:
        """Tests single type MTGP via Modular BoTorch instantiation
        with both single & multi objective optimization."""
        for exp in [
            get_branin_experiment_with_status_quo_trials(num_sobol_trials=2),
            get_branin_experiment_with_status_quo_trials(
                num_sobol_trials=2, multi_objective=True
            ),
        ]:
            # testing custom and default kernel for a surrogate
            surrogates = (
                [None]
                if use_saas
                else [
                    Surrogate(
                        surrogate_spec=SurrogateSpec(
                            model_configs=[
                                ModelConfig(
                                    botorch_model_class=MultiTaskGP,
                                    mll_class=ExactMarginalLogLikelihood,
                                    covar_module_class=ScaleMaternKernel,
                                    covar_module_options={
                                        "ard_num_dims": DEFAULT,
                                        "lengthscale_prior": GammaPrior(6.0, 3.0),
                                        "outputscale_prior": GammaPrior(2.0, 0.15),
                                        "batch_shape": DEFAULT,
                                    },
                                    model_options={},
                                )
                            ],
                            allow_batched_models=False,
                        )
                    ),
                    None,
                ]
            )

            for surrogate, default_model in zip(surrogates, (False, True)):
                constructor = Generators.SAAS_MTGP if use_saas else Generators.ST_MTGP
                mtgp = constructor(
                    experiment=exp,
                    data=exp.fetch_data(),
                    surrogate=surrogate,
                )
                self.assertIsInstance(mtgp, TorchAdapter)
                generator = assert_is_instance(mtgp.generator, BoTorchGenerator)
                self.assertEqual(generator.acquisition_class, Acquisition)
                is_moo = isinstance(
                    exp.optimization_config, MultiObjectiveOptimizationConfig
                )
                if is_moo:
                    self.assertIsInstance(generator.surrogate.model, ModelListGP)
                    models = generator.surrogate.model.models
                else:
                    models = [generator.surrogate.model]

                for model in models:
                    self.assertIsInstance(
                        model,
                        SaasFullyBayesianMultiTaskGP if use_saas else MultiTaskGP,
                    )
                    data_covar_module, task_covar_module = model.covar_module.kernels
                    if use_saas is False and default_model is False:
                        self.assertIsInstance(data_covar_module, ScaleKernel)
                        base_kernel = data_covar_module.base_kernel
                        self.assertIsInstance(base_kernel, MaternKernel)
                        self.assertEqual(
                            base_kernel.lengthscale_prior.concentration, 6.0
                        )
                        self.assertEqual(base_kernel.lengthscale_prior.rate, 3.0)
                    elif use_saas is False:
                        self.assertIsInstance(data_covar_module, RBFKernel)
                        self.assertIsInstance(
                            data_covar_module.lengthscale_prior, LogNormalPrior
                        )

                gr = mtgp.gen(
                    n=1,
                    fixed_features=ObservationFeatures({}, trial_index=1),
                )
                self.assertEqual(len(gr.arms), 1)

    def test_SAAS_MTGP(self) -> None:
        self.test_ST_MTGP(use_saas=True)

    def test_extract_model_state_after_gen(self) -> None:
        # Test with actual state.
        exp = get_branin_experiment()
        sobol = Generators.SOBOL(experiment=exp)
        gr = sobol.gen(n=1)
        expected_state = sobol.generator._get_state()
        self.assertEqual(gr._model_state_after_gen, expected_state)
        extracted = _extract_model_state_after_gen(
            generator_run=gr, model_class=SobolGenerator
        )
        self.assertEqual(extracted, expected_state)
        # Test with empty state.
        gr._model_state_after_gen = None
        extracted = _extract_model_state_after_gen(
            generator_run=gr, model_class=SobolGenerator
        )
        self.assertEqual(extracted, {})

    def test_initialize_from_search_space(self) -> None:
        search_space = get_branin_search_space()
        with self.assertWarnsRegex(
            DeprecationWarning, "Passing in a `search_space` to initialize"
        ):
            adapter = Generators.SOBOL(search_space=search_space)
        self.assertEqual(adapter._model_space, search_space)
        self.assertIsNotNone(adapter._experiment)
        with self.assertRaisesRegex(
            UserInputError,
            "`experiment` is required to initialize a model from registry.",
        ):
            Generators.BOTORCH_MODULAR(search_space=search_space)
