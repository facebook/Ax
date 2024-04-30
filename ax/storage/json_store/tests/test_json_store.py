#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import tempfile
from functools import partial

import numpy as np
import torch
from ax.benchmark.metrics.jenatton import JenattonMetric
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.runner import Runner
from ax.exceptions.core import AxStorageWarning
from ax.exceptions.storage import JSONDecodeError, JSONEncodeError
from ax.modelbridge.generation_node import GenerationStep
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.registry import Models
from ax.storage.json_store.decoder import (
    generation_strategy_from_json,
    object_from_json,
)
from ax.storage.json_store.decoders import botorch_component_from_json, class_from_json
from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.encoders import (
    botorch_component_to_dict,
    botorch_modular_to_dict,
    metric_to_dict,
    runner_to_dict,
)
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.registry import (
    CORE_CLASS_DECODER_REGISTRY,
    CORE_CLASS_ENCODER_REGISTRY,
    CORE_DECODER_REGISTRY,
    CORE_ENCODER_REGISTRY,
)
from ax.storage.json_store.save import save_experiment
from ax.storage.registry_bundle import RegistryBundle
from ax.utils.common.testutils import TestCase
from ax.utils.testing.benchmark_stubs import (
    get_aggregated_benchmark_result,
    get_benchmark_problem,
    get_benchmark_result,
    get_multi_objective_benchmark_problem,
    get_single_objective_benchmark_problem,
    get_sobol_gpei_benchmark_method,
)
from ax.utils.testing.core_stubs import (
    get_abandoned_arm,
    get_acquisition_function_type,
    get_acquisition_type,
    get_and_early_stopping_strategy,
    get_arm,
    get_augmented_branin_metric,
    get_augmented_hartmann_metric,
    get_batch_trial,
    get_botorch_model,
    get_botorch_model_with_default_acquisition_class,
    get_botorch_model_with_surrogate_specs,
    get_branin_data,
    get_branin_experiment,
    get_branin_experiment_with_timestamp_map_metric,
    get_branin_metric,
    get_chained_input_transform,
    get_choice_parameter,
    get_default_scheduler_options,
    get_dict_lookup_metric,
    get_experiment_with_batch_and_single_trial,
    get_experiment_with_data,
    get_experiment_with_map_data,
    get_experiment_with_map_data_type,
    get_experiment_with_trial_with_ttl,
    get_factorial_metric,
    get_fixed_parameter,
    get_gamma_prior,
    get_generator_run,
    get_hartmann_metric,
    get_hierarchical_search_space,
    get_improvement_global_stopping_strategy,
    get_interval,
    get_map_data,
    get_map_key_info,
    get_metric,
    get_mll_type,
    get_model_type,
    get_multi_objective,
    get_multi_objective_optimization_config,
    get_multi_type_experiment,
    get_objective,
    get_objective_threshold,
    get_optimization_config,
    get_or_early_stopping_strategy,
    get_order_constraint,
    get_outcome_constraint,
    get_parameter_constraint,
    get_parameter_distribution,
    get_pathlib_path,
    get_percentile_early_stopping_strategy,
    get_percentile_early_stopping_strategy_with_non_objective_metric_name,
    get_range_parameter,
    get_risk_measure,
    get_robust_search_space,
    get_scalarized_objective,
    get_scheduler_options_batch_trial,
    get_search_space,
    get_sebo_acquisition_class,
    get_sum_constraint1,
    get_sum_constraint2,
    get_surrogate,
    get_synthetic_runner,
    get_threshold_early_stopping_strategy,
    get_trial,
    get_trial_based_criterion,
    get_winsorization_config,
)
from ax.utils.testing.modeling_stubs import (
    get_generation_strategy,
    get_input_transform_type,
    get_observation_features,
    get_outcome_transfrom_type,
    get_transform_type,
)
from ax.utils.testing.utils import generic_equals


# pyre-fixme[5]: Global expression must be annotated.
TEST_CASES = [
    ("AbandonedArm", get_abandoned_arm),
    ("AggregatedBenchmarkResult", get_aggregated_benchmark_result),
    ("AndEarlyStoppingStrategy", get_and_early_stopping_strategy),
    ("Arm", get_arm),
    ("AugmentedBraninMetric", get_augmented_branin_metric),
    ("AugmentedHartmannMetric", get_augmented_hartmann_metric),
    ("BatchTrial", get_batch_trial),
    ("BenchmarkMethod", get_sobol_gpei_benchmark_method),
    ("BenchmarkProblem", get_benchmark_problem),
    ("BenchmarkResult", get_benchmark_result),
    ("BoTorchModel", get_botorch_model),
    ("BoTorchModel", get_botorch_model_with_default_acquisition_class),
    ("BoTorchModel", get_botorch_model_with_surrogate_specs),
    ("BraninMetric", get_branin_metric),
    ("ChainedInputTransform", get_chained_input_transform),
    ("ChoiceParameter", get_choice_parameter),
    ("DictLookupMetric", get_dict_lookup_metric),
    ("Experiment", get_experiment_with_batch_and_single_trial),
    ("Experiment", get_experiment_with_trial_with_ttl),
    ("Experiment", get_experiment_with_data),
    ("Experiment", get_experiment_with_map_data_type),
    ("Experiment", get_branin_experiment_with_timestamp_map_metric),
    ("Experiment", get_experiment_with_map_data),
    ("FactorialMetric", get_factorial_metric),
    ("FixedParameter", get_fixed_parameter),
    ("GammaPrior", get_gamma_prior),
    ("GenerationStrategy", partial(get_generation_strategy, with_experiment=True)),
    (
        "GenerationStrategy",
        partial(
            get_generation_strategy, with_experiment=True, with_completion_criteria=3
        ),
    ),
    (
        "GenerationStrategy",
        partial(
            get_generation_strategy,
            with_experiment=True,
            with_generation_nodes=True,
            with_callable_model_kwarg=False,
        ),
    ),
    ("GeneratorRun", get_generator_run),
    ("Hartmann6Metric", get_hartmann_metric),
    ("HierarchicalSearchSpace", get_hierarchical_search_space),
    ("ImprovementGlobalStoppingStrategy", get_improvement_global_stopping_strategy),
    ("Interval", get_interval),
    ("JenattonMetric", JenattonMetric),
    ("MapData", get_map_data),
    ("MapData", get_map_data),
    ("MapKeyInfo", get_map_key_info),
    ("Metric", get_metric),
    ("MultiObjective", get_multi_objective),
    ("MultiObjectiveBenchmarkProblem", get_multi_objective_benchmark_problem),
    ("MultiObjectiveOptimizationConfig", get_multi_objective_optimization_config),
    ("MultiTypeExperiment", get_multi_type_experiment),
    ("ObservationFeatures", get_observation_features),
    ("Objective", get_objective),
    ("ObjectiveThreshold", get_objective_threshold),
    ("OptimizationConfig", get_optimization_config),
    ("OrEarlyStoppingStrategy", get_or_early_stopping_strategy),
    ("OrderConstraint", get_order_constraint),
    ("OutcomeConstraint", get_outcome_constraint),
    ("Path", get_pathlib_path),
    ("PercentileEarlyStoppingStrategy", get_percentile_early_stopping_strategy),
    (
        "PercentileEarlyStoppingStrategy",
        get_percentile_early_stopping_strategy_with_non_objective_metric_name,
    ),
    ("ParameterConstraint", get_parameter_constraint),
    ("ParameterDistribution", get_parameter_distribution),
    ("RangeParameter", get_range_parameter),
    ("RiskMeasure", get_risk_measure),
    ("RobustSearchSpace", get_robust_search_space),
    ("ScalarizedObjective", get_scalarized_objective),
    ("SchedulerOptions", get_default_scheduler_options),
    ("SchedulerOptions", get_scheduler_options_batch_trial),
    ("SearchSpace", get_search_space),
    ("SingleObjectiveBenchmarkProblem", get_single_objective_benchmark_problem),
    ("SumConstraint", get_sum_constraint1),
    ("SumConstraint", get_sum_constraint2),
    ("Surrogate", get_surrogate),
    ("SyntheticRunner", get_synthetic_runner),
    ("Type[Acquisition]", get_acquisition_type),
    ("Type[AcquisitionFunction]", get_acquisition_function_type),
    ("Type[Model]", get_model_type),
    ("Type[MarginalLogLikelihood]", get_mll_type),
    ("Type[Transform]", get_transform_type),
    ("Type[InputTransform]", get_input_transform_type),
    ("Type[OutcomeTransform]", get_outcome_transfrom_type),
    ("TransitionCriterionList", get_trial_based_criterion),
    ("ThresholdEarlyStoppingStrategy", get_threshold_early_stopping_strategy),
    ("Trial", get_trial),
    ("WinsorizationConfig", get_winsorization_config),
    ("SEBOAcquisition", get_sebo_acquisition_class),
]


class JSONStoreTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.experiment = get_experiment_with_batch_and_single_trial()

    def test_JSONEncodeFailure(self) -> None:
        with self.assertRaises(JSONEncodeError):
            object_to_json(
                obj=RuntimeError("foobar"),
                encoder_registry=CORE_ENCODER_REGISTRY,
                class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
            )

    def test_JSONDecodeFailure(self) -> None:
        self.assertRaises(
            JSONDecodeError,
            object_from_json,
            RuntimeError("foobar"),
            CORE_DECODER_REGISTRY,
            CORE_CLASS_DECODER_REGISTRY,
        )
        self.assertRaises(
            JSONDecodeError,
            object_from_json,
            {"__type": "foobar"},
            CORE_DECODER_REGISTRY,
            CORE_CLASS_DECODER_REGISTRY,
        )

    def test_SaveAndLoad(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
            save_experiment(
                self.experiment,
                f.name,
                encoder_registry=CORE_ENCODER_REGISTRY,
                class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
            )
            loaded_experiment = load_experiment(
                f.name,
                decoder_registry=CORE_DECODER_REGISTRY,
                class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
            )
            self.assertEqual(loaded_experiment, self.experiment)
            os.remove(f.name)

    def test_SaveValidation(self) -> None:
        with self.assertRaises(ValueError):
            save_experiment(
                self.experiment.trials[0],
                "test.json",
                encoder_registry=CORE_ENCODER_REGISTRY,
                class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
            )

    def test_ValidateFilename(self) -> None:
        bad_filename = "test"
        self.assertRaises(
            ValueError,
            save_experiment,
            self.experiment,
            bad_filename,
            CORE_ENCODER_REGISTRY,
            CORE_CLASS_ENCODER_REGISTRY,
        )

    def test_EncodeDecode(self) -> None:
        for class_, fake_func in TEST_CASES:
            # Can't load trials from JSON, because a batch needs an experiment
            # in order to be initialized
            if class_ == "BatchTrial" or class_ == "Trial":
                continue

            # Can't load parameter constraints from JSON, because they require
            # a SearchSpace in order to be initialized
            if class_ == "OrderConstraint" or class_ == "SumConstraint":
                continue

            original_object = fake_func()

            json_object = object_to_json(
                original_object,
                encoder_registry=CORE_ENCODER_REGISTRY,
                class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
            )

            converted_object = object_from_json(
                json_object,
                decoder_registry=CORE_DECODER_REGISTRY,
                class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
            )

            if class_ == "SimpleExperiment":
                # Evaluation functions will be different, so need to do
                # this so equality test passes
                with self.assertRaises(RuntimeError):
                    converted_object.evaluation_function(parameterization={})

                original_object.evaluation_function = None
                converted_object.evaluation_function = None
            if class_ == "BenchmarkMethod":
                # Some fields of the reloaded GS are not expected to be set (both will
                # be set during next model fitting call), so we unset them on the
                # original GS as well.
                original_object.generation_strategy._unset_non_persistent_state_fields()
            if isinstance(original_object, torch.nn.Module):
                self.assertIsInstance(
                    converted_object,
                    original_object.__class__,
                    msg=f"Error encoding/decoding {class_}.",
                )
                original_object = original_object.state_dict()
                converted_object = converted_object.state_dict()
            if isinstance(original_object, GenerationStrategy):
                original_object._unset_non_persistent_state_fields()
                # for the test, completion criterion are set post init
                # and therefore do not become transition criterion, unset
                # for this specific test only
                if "with_completion_criteria" in fake_func.keywords:
                    for step in original_object._steps:
                        step._transition_criteria = None
                    for step in converted_object._steps:
                        step._transition_criteria = None
                    # also unset the `transition_to` field for the same reason
                    for criterion in converted_object._steps[0].completion_criteria:
                        if criterion.criterion_class == "MinimumPreferenceOccurances":
                            criterion._transition_to = None
            try:
                self.assertEqual(
                    original_object,
                    converted_object,
                    msg=f"Error encoding/decoding {class_}.",
                )
            except RuntimeError as e:
                if "Tensor with more than one value" in str(e):
                    self.assertTrue(
                        generic_equals(first=original_object, second=converted_object)
                    )
                else:
                    raise e

    def test_EncodeDecodeTorchTensor(self) -> None:
        x = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, device=torch.device("cpu")
        )
        expected_json = {
            "__type": "Tensor",
            "value": [[1.0, 2.0], [3.0, 4.0]],
            "dtype": {"__type": "torch_dtype", "value": "torch.float64"},
            "device": {"__type": "torch_device", "value": "cpu"},
        }
        x_json = object_to_json(
            x,
            encoder_registry=CORE_ENCODER_REGISTRY,
            class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
        )
        self.assertEqual(expected_json, x_json)
        x2 = object_from_json(
            x_json,
            decoder_registry=CORE_DECODER_REGISTRY,
            class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
        )
        self.assertTrue(torch.equal(x, x2))

        # Warning on large tensor.
        with self.assertWarnsRegex(AxStorageWarning, "serialize a tensor"):
            object_to_json(
                torch.ones(99_999),
                encoder_registry=CORE_ENCODER_REGISTRY,
                class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
            )

        # Key error on desearialization.
        x_json = object_to_json(
            x,
            encoder_registry=CORE_ENCODER_REGISTRY,
            class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
        )
        x_json.pop("dtype")
        with self.assertRaisesRegex(JSONDecodeError, "construct a tensor"):
            object_from_json(
                x_json,
                decoder_registry=CORE_DECODER_REGISTRY,
                class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
            )

    def test_DecodeGenerationStrategy(self) -> None:
        generation_strategy = get_generation_strategy()
        experiment = get_branin_experiment()
        gs_json = object_to_json(
            generation_strategy,
            encoder_registry=CORE_ENCODER_REGISTRY,
            class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
        )
        new_generation_strategy = generation_strategy_from_json(
            gs_json,
            decoder_registry=CORE_DECODER_REGISTRY,
            class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
        )
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(generation_strategy, new_generation_strategy)
        self.assertGreater(len(new_generation_strategy._steps), 0)
        self.assertIsInstance(new_generation_strategy._steps[0].model, Models)
        # Model has not yet been initialized on this GS since it hasn't generated
        # anything yet.
        self.assertIsNone(new_generation_strategy.model)

        # Check that we can encode and decode the generation strategy after
        # it has generated some generator runs. Since we now need to `gen`,
        # we remove the fake callable kwarg we added, since model does not
        # expect it.
        generation_strategy = get_generation_strategy(with_callable_model_kwarg=False)
        gr = generation_strategy.gen(experiment)
        gs_json = object_to_json(
            generation_strategy,
            encoder_registry=CORE_ENCODER_REGISTRY,
            class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
        )
        new_generation_strategy = generation_strategy_from_json(
            gs_json,
            decoder_registry=CORE_DECODER_REGISTRY,
            class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
        )
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(generation_strategy, new_generation_strategy)
        self.assertIsInstance(new_generation_strategy._steps[0].model, Models)

        # Check that we can encode and decode the generation strategy after
        # it has generated some trials and been updated with some data.
        generation_strategy = new_generation_strategy
        experiment.new_trial(gr)  # Add previously generated GR as trial.
        # Make generation strategy aware of the trial's data via `gen`.
        generation_strategy.gen(experiment, data=get_branin_data())
        gs_json = object_to_json(
            generation_strategy,
            encoder_registry=CORE_ENCODER_REGISTRY,
            class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
        )
        new_generation_strategy = generation_strategy_from_json(
            gs_json,
            decoder_registry=CORE_DECODER_REGISTRY,
            class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
        )
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(generation_strategy, new_generation_strategy)
        self.assertIsInstance(new_generation_strategy._steps[0].model, Models)

    def test_EncodeDecodeNumpy(self) -> None:
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(
            np.array_equal(
                arr,
                object_from_json(
                    object_to_json(
                        arr,
                        encoder_registry=CORE_ENCODER_REGISTRY,
                        class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
                    ),
                    decoder_registry=CORE_DECODER_REGISTRY,
                    class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
                ),
            )
        )

    def test_EncodeDecodeSet(self) -> None:
        a = {"a", 1, False}
        self.assertEqual(
            a,
            object_from_json(
                object_to_json(
                    a,
                    encoder_registry=CORE_ENCODER_REGISTRY,
                    class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
                ),
                decoder_registry=CORE_DECODER_REGISTRY,
                class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
            ),
        )

    def test_RegistryAdditions(self) -> None:
        class MyRunner(Runner):
            def run():
                pass

            def staging_required():
                return False

        class MyMetric(Metric):
            pass

        encoder_registry = {
            MyMetric: metric_to_dict,
            MyRunner: runner_to_dict,
            **CORE_ENCODER_REGISTRY,
        }
        decoder_registry = {
            MyMetric.__name__: MyMetric,
            MyRunner.__name__: MyRunner,
            **CORE_DECODER_REGISTRY,
        }

        experiment = get_experiment_with_batch_and_single_trial()
        experiment.runner = MyRunner()
        experiment.add_tracking_metric(MyMetric(name="my_metric"))
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
            save_experiment(
                experiment,
                f.name,
                encoder_registry=encoder_registry,
                class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
            )
            loaded_experiment = load_experiment(
                f.name,
                decoder_registry=decoder_registry,
                class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
            )
            self.assertEqual(loaded_experiment, experiment)
            os.remove(f.name)

    def test_RegistryBundle(self) -> None:
        class MyMetric(Metric):
            pass

        class MyRunner(Runner):
            def run():
                pass

            def staging_required():
                return False

        bundle = RegistryBundle(
            metric_clss={MyMetric: 1998}, runner_clss={MyRunner: None}
        )

        experiment = get_experiment_with_batch_and_single_trial()
        experiment.runner = MyRunner()
        experiment.add_tracking_metric(MyMetric(name="my_metric"))
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
            save_experiment(
                experiment,
                f.name,
                encoder_registry=bundle.encoder_registry,
            )
            loaded_experiment = load_experiment(
                f.name,
                decoder_registry=bundle.decoder_registry,
            )
            self.assertEqual(loaded_experiment, experiment)
            os.remove(f.name)

    def test_EncodeUnknownClassToDict(self) -> None:
        # Cannot encode `UnknownClass` type because it is not registered in the
        # CLASS_ENCODER_REGISTRY.
        class UnknownClass:
            def __init__(self) -> None:
                pass

        with self.assertRaisesRegex(
            ValueError, "is a class. Add it to the CLASS_ENCODER_REGISTRY"
        ):
            object_to_json(
                UnknownClass,
                encoder_registry=CORE_ENCODER_REGISTRY,
                class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
            )
        # `UnknownClass` type is registered in the CLASS_ENCODER_REGISTRY and uses the
        # `botorch_modular_to_dict` encoder, but `UnknownClass` is not registered in
        # the `botorch_modular_registry.py` file.
        CORE_CLASS_ENCODER_REGISTRY[UnknownClass] = botorch_modular_to_dict
        with self.assertRaisesRegex(
            ValueError,
            "does not have a corresponding parent class in CLASS_TO_REGISTRY",
        ):
            object_to_json(
                UnknownClass,
                encoder_registry=CORE_ENCODER_REGISTRY,
                class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
            )

    def test_DecodeUnknownClassFromJson(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "does not have a corresponding entry in CLASS_TO_REVERSE_REGISTRY",
        ):
            class_from_json({"index": 0, "class": "unknown_path"})

    def test_unregistered_model_not_supported_in_nodes(self) -> None:
        """Support for callables within model kwargs on ModelSpecs stored on
        GenerationNodes is currently not supported. This is supported for
        GenerationSteps due to legacy compatibility.
        """
        with self.assertRaisesRegex(
            JSONEncodeError,
            "is not registered with a corresponding encoder",
        ):
            gs = get_generation_strategy(
                with_experiment=True,
                with_generation_nodes=True,
                with_callable_model_kwarg=True,
                with_completion_criteria=0,
            )
            object_to_json(
                gs,
                encoder_registry=CORE_ENCODER_REGISTRY,
                class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
            )

    def test_BadStateDict(self) -> None:
        interval = get_interval()
        expected_json = botorch_component_to_dict(interval)
        with self.assertRaisesRegex(ValueError, "Received unused args"):
            expected_json = botorch_component_to_dict(interval)
            expected_json["state_dict"]["foo"] = "bar"
            botorch_component_from_json(interval.__class__, expected_json)
        with self.assertRaisesRegex(ValueError, "Missing required initialization args"):
            expected_json = botorch_component_to_dict(interval)
            del expected_json["state_dict"]["lower_bound"]
            botorch_component_from_json(interval.__class__, expected_json)

    def test_objective_backwards_compatibility(self) -> None:
        # Test that we can load an objective that has conflicting
        # ``lower_is_better`` and ``minimize`` fields.
        objective = get_objective(minimize=True)
        objective.metric.lower_is_better = False  # for conflict!
        objective_json = object_to_json(objective)
        self.assertTrue(objective_json["minimize"])
        self.assertFalse(objective_json["metric"]["lower_is_better"])
        objective_loaded = object_from_json(objective_json)
        self.assertIsInstance(objective_loaded, Objective)
        self.assertNotEqual(objective, objective_loaded)
        self.assertTrue(objective_loaded.minimize)
        self.assertTrue(objective_loaded.metric.lower_is_better)

    def test_generation_step_backwards_compatibility(self) -> None:
        # Test that we can load a generation step with fit_on_update.
        json = {
            "__type": "GenerationStep",
            "model": {"__type": "Models", "name": "BOTORCH_MODULAR"},
            "num_trials": 5,
            "min_trials_observed": 0,
            "completion_criteria": [],
            "max_parallelism": None,
            "use_update": False,
            "enforce_num_trials": True,
            "model_kwargs": {"fit_on_update": False, "other_kwarg": 5},
            "model_gen_kwargs": {},
            "index": -1,
            "should_deduplicate": False,
        }
        generation_step = object_from_json(json)
        self.assertIsInstance(generation_step, GenerationStep)
        self.assertEqual(generation_step.model_kwargs, {"other_kwarg": 5})
