#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from itertools import product
from typing import cast

import numpy as np
from ax.core.experiment import Experiment
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.branin import BraninMetric
from ax.modelbridge.cross_validation import (
    _predict_on_cross_validation_data,
    _predict_on_training_data,
    compute_model_fit_metrics_from_modelbridge,
    get_fit_and_std_quality_and_generalization_dict,
)
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from ax.service.scheduler import get_fitted_model_bridge, Scheduler, SchedulerOptions
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.stats.model_fit_stats import _entropy_via_kde, entropy_of_observations
from ax.utils.testing.core_stubs import get_branin_experiment, get_branin_search_space

NUM_SOBOL = 5


class TestModelBridgeFitMetrics(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # setting up experiment and generation strategy
        self.runner = SyntheticRunner()
        self.branin_experiment = Experiment(
            name="branin_test_experiment",
            search_space=get_branin_search_space(),
            runner=self.runner,
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=BraninMetric(name="branin", param_names=["x1", "x2"]),
                    minimize=True,
                ),
            ),
            is_test=True,
        )
        self.branin_experiment._properties[Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF] = (
            True
        )
        self.generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL, num_trials=NUM_SOBOL, max_parallelism=NUM_SOBOL
                ),
                GenerationStep(model=Models.BOTORCH_MODULAR, num_trials=-1),
            ]
        )

    def test_model_fit_metrics(self) -> None:
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=self.generation_strategy,
            options=SchedulerOptions(),
        )
        # need to run some trials to initialize the ModelBridge
        scheduler.run_n_trials(max_trials=NUM_SOBOL + 1)

        model_bridge = get_fitted_model_bridge(scheduler)
        self.assertEqual(len(model_bridge.get_training_data()), NUM_SOBOL)

        model_bridge = get_fitted_model_bridge(scheduler, force_refit=True)
        self.assertEqual(len(model_bridge.get_training_data()), NUM_SOBOL + 1)

        # testing compute_model_fit_metrics_from_modelbridge with default metrics
        fit_metrics = compute_model_fit_metrics_from_modelbridge(
            model_bridge=model_bridge,
            untransform=False,
        )
        r2 = fit_metrics.get("coefficient_of_determination")
        self.assertIsInstance(r2, dict)
        r2 = cast(dict[str, float], r2)
        self.assertTrue("branin" in r2)
        r2_branin = r2["branin"]
        self.assertIsInstance(r2_branin, float)

        std = fit_metrics.get("std_of_the_standardized_error")
        self.assertIsInstance(std, dict)
        std = cast(dict[str, float], std)
        self.assertTrue("branin" in std)
        std_branin = std["branin"]
        self.assertIsInstance(std_branin, float)

        # checking non-default model-fit-metric
        for untransform, generalization in product([True, False], [True, False]):
            with self.subTest(untransform=untransform):
                fit_metrics = compute_model_fit_metrics_from_modelbridge(
                    model_bridge=model_bridge,
                    generalization=generalization,
                    untransform=untransform,
                    fit_metrics_dict={"Entropy": entropy_of_observations},
                )
                entropy = fit_metrics.get("Entropy")
                self.assertIsInstance(entropy, dict)
                entropy = cast(dict[str, float], entropy)
                self.assertTrue("branin" in entropy)
                entropy_branin = entropy["branin"]
                self.assertIsInstance(entropy_branin, float)

                predict = (
                    _predict_on_cross_validation_data
                    if generalization
                    else _predict_on_training_data
                )
                y_obs, _, _ = predict(
                    model_bridge=model_bridge, untransform=untransform
                )
                y_obs_branin = np.array(y_obs["branin"])[:, np.newaxis]
                entropy_truth = _entropy_via_kde(y_obs_branin)
                self.assertAlmostEqual(entropy_branin, entropy_truth)

                # testing with empty metrics
                empty_metrics = compute_model_fit_metrics_from_modelbridge(
                    model_bridge=model_bridge,
                    fit_metrics_dict={},
                )
                self.assertIsInstance(empty_metrics, dict)
                self.assertTrue(len(empty_metrics) == 0)

                # testing log filtering
                with warnings.catch_warnings(record=True) as ws:
                    fit_metrics = compute_model_fit_metrics_from_modelbridge(
                        model_bridge=model_bridge,
                        untransform=untransform,
                        generalization=generalization,
                    )
                self.assertFalse(
                    any("Data is not standardized" in str(w.message) for w in ws)
                )


class TestGetFitAndStdQualityAndGeneralizationDict(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.experiment = get_branin_experiment()
        self.sobol = Models.SOBOL(search_space=self.experiment.search_space)

    def test_it_returns_empty_data_for_sobol(self) -> None:
        results = get_fit_and_std_quality_and_generalization_dict(
            fitted_model_bridge=self.sobol,
        )
        expected = {
            "model_fit_quality": None,
            "model_std_quality": None,
            "model_fit_generalization": None,
            "model_std_generalization": None,
        }
        self.assertDictEqual(results, expected)

    def test_it_returns_float_values_when_fit_can_be_evaluated(self) -> None:
        # GIVEN we have a model whose CV can be evaluated
        sobol_run = self.sobol.gen(n=20)
        self.experiment.new_batch_trial().add_generator_run(
            sobol_run
        ).run().mark_completed()
        data = self.experiment.fetch_data()
        botorch_modelbridge = Models.BOTORCH_MODULAR(
            experiment=self.experiment, data=data
        )

        # WHEN we call get_fit_and_std_quality_and_generalization_dict
        results = get_fit_and_std_quality_and_generalization_dict(
            fitted_model_bridge=botorch_modelbridge,
        )

        # THEN we get expected results
        # CALCULATE EXPECTED RESULTS
        fit_metrics = compute_model_fit_metrics_from_modelbridge(
            model_bridge=botorch_modelbridge,
            generalization=False,
            untransform=False,
        )
        # checking fit metrics
        r2 = fit_metrics.get("coefficient_of_determination")
        r2 = cast(dict[str, float], r2)

        std = fit_metrics.get("std_of_the_standardized_error")
        std = cast(dict[str, float], std)
        std_branin = std["branin"]

        model_std_quality = 1 / std_branin

        # check generalization metrics
        gen_metrics = compute_model_fit_metrics_from_modelbridge(
            model_bridge=botorch_modelbridge,
            generalization=True,
            untransform=False,
        )
        r2_gen = gen_metrics.get("coefficient_of_determination")
        r2_gen = cast(dict[str, float], r2_gen)
        gen_std = gen_metrics.get("std_of_the_standardized_error")
        gen_std = cast(dict[str, float], gen_std)
        gen_std_branin = gen_std["branin"]
        model_std_generalization = 1 / gen_std_branin

        expected = {
            "model_fit_quality": min(r2.values()),
            "model_std_quality": model_std_quality,
            "model_fit_generalization": min(r2_gen.values()),
            "model_std_generalization": model_std_generalization,
        }
        # END CALCULATE EXPECTED RESULTS

        self.assertDictsAlmostEqual(results, expected)
