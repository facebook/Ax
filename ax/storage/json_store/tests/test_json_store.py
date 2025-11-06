#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
import json
import os
import tempfile
import warnings
from functools import partial
from math import nan

import numpy as np
import torch
from ax.adapter.base import DataLoaderConfig
from ax.adapter.registry import Generators
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.log import Log
from ax.adapter.transforms.one_hot import OneHot
from ax.benchmark.methods.sobol import get_sobol_benchmark_method
from ax.benchmark.testing.benchmark_stubs import (
    get_aggregated_benchmark_result,
    get_benchmark_map_metric,
    get_benchmark_map_unavailable_while_running_metric,
    get_benchmark_metric,
    get_benchmark_result,
    get_benchmark_time_varying_metric,
)
from ax.core.auxiliary import AuxiliaryExperimentPurpose
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun
from ax.core.map_data import MAP_KEY, MapData
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
    PreferenceOptimizationConfig,
)
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.runner import Runner
from ax.exceptions.core import AxStorageWarning, UnsupportedError
from ax.exceptions.storage import JSONDecodeError, JSONEncodeError
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.generation_node import GenerationNode, GenerationStep
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generators.torch.botorch_modular.kernels import ScaleMaternKernel
from ax.generators.torch.botorch_modular.surrogate import Surrogate, SurrogateSpec
from ax.generators.torch.botorch_modular.utils import ModelConfig
from ax.storage.json_store.decoder import (
    _DEPRECATED_MODEL_TO_REPLACEMENT,
    generation_node_from_json,
    generation_strategy_from_json,
    object_from_json,
)
from ax.storage.json_store.decoders import (
    botorch_component_from_json,
    class_from_json,
    multi_objective_from_json,
)
from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.encoders import (
    botorch_component_to_dict,
    botorch_modular_to_dict,
    choice_parameter_to_dict,
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
from ax.utils.testing.core_stubs import (
    get_abandoned_arm,
    get_acquisition_function_type,
    get_acquisition_type,
    get_and_early_stopping_strategy,
    get_arm,
    get_auxiliary_experiment,
    get_batch_trial,
    get_botorch_model,
    get_botorch_model_with_default_acquisition_class,
    get_botorch_model_with_surrogate_spec,
    get_branin_data,
    get_branin_experiment,
    get_branin_experiment_with_timestamp_map_metric,
    get_branin_metric,
    get_chained_input_transform,
    get_choice_parameter,
    get_default_orchestrator_options,
    get_derived_parameter,
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
    get_hierarchical_choice_parameter,
    get_hierarchical_search_space,
    get_improvement_global_stopping_strategy,
    get_interval,
    get_map_data,
    get_map_metric,
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
    get_orchestrator_options_batch_trial,
    get_order_constraint,
    get_outcome_constraint,
    get_parameter_constraint,
    get_pathlib_path,
    get_percentile_early_stopping_strategy,
    get_percentile_early_stopping_strategy_with_non_objective_metric_signature,
    get_range_parameter,
    get_scalarized_objective,
    get_search_space,
    get_sorted_choice_parameter,
    get_sum_constraint1,
    get_sum_constraint2,
    get_surrogate,
    get_surrogate_spec_with_default,
    get_surrogate_spec_with_lognormal,
    get_synthetic_runner,
    get_threshold_early_stopping_strategy,
    get_trial,
    get_trial_based_criterion,
    get_winsorization_config,
)
from ax.utils.testing.modeling_stubs import (
    get_generation_strategy,
    get_input_transform_type,
    get_legacy_list_surrogate_generation_step_as_dict,
    get_observation_features,
    get_outcome_transfrom_type,
    get_surrogate_as_dict,
    get_surrogate_generation_step,
    get_surrogate_spec_as_dict,
    get_to_new_sq_transform_type,
    get_transform_type,
    sobol_gpei_generation_node_gs,
)
from ax.utils.testing.utils import generic_equals
from ax.utils.testing.utils_testing_stubs import get_backend_simulator_with_trials
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from pyre_extensions import none_throws


# pyre-fixme[5]: Global expression must be annotated.
TEST_CASES = [
    ("AbandonedArm", get_abandoned_arm),
    ("AggregatedBenchmarkResult", get_aggregated_benchmark_result),
    ("AndEarlyStoppingStrategy", get_and_early_stopping_strategy),
    ("Arm", get_arm),
    ("AuxiliaryExperiment", get_auxiliary_experiment),
    ("AuxiliaryExperimentPurpose", lambda: AuxiliaryExperimentPurpose.PE_EXPERIMENT),
    ("BackendSimulator", get_backend_simulator_with_trials),
    ("BatchTrial", get_batch_trial),
    ("BenchmarkMethod", get_sobol_benchmark_method),
    ("BenchmarkMetric", get_benchmark_metric),
    ("BenchmarkMapMetric", get_benchmark_map_metric),
    ("BenchmarkTimeVaryingMetric", get_benchmark_time_varying_metric),
    (
        "BenchmarkMapUnavailableWhileRunningMetric",
        get_benchmark_map_unavailable_while_running_metric,
    ),
    ("BenchmarkResult", get_benchmark_result),
    ("BoTorchGenerator", get_botorch_model),
    ("BoTorchGenerator", get_botorch_model_with_default_acquisition_class),
    ("BoTorchGenerator", get_botorch_model_with_surrogate_spec),
    ("BraninMetric", get_branin_metric),
    ("CenterGenerationNode", partial(CenterGenerationNode, next_node_name="SOBOL")),
    ("ChainedInputTransform", get_chained_input_transform),
    ("ChoiceParameter", get_choice_parameter),
    ("ChoiceParameter", get_sorted_choice_parameter),
    (
        "ChoiceParameter",
        partial(get_hierarchical_choice_parameter, parameter_type=ParameterType.BOOL),
    ),
    (
        "ChoiceParameter",
        partial(get_hierarchical_choice_parameter, parameter_type=ParameterType.INT),
    ),
    (
        "ChoiceParameter",
        partial(get_hierarchical_choice_parameter, parameter_type=ParameterType.FLOAT),
    ),
    (
        "ChoiceParameter",
        partial(get_hierarchical_choice_parameter, parameter_type=ParameterType.STRING),
    ),
    # testing with non-default argument
    (
        "DataLoaderConfig",
        partial(DataLoaderConfig, fit_only_completed_map_metrics=True),
    ),
    ("DerivedParameter", get_derived_parameter),
    ("Experiment", get_experiment_with_batch_and_single_trial),
    ("Experiment", get_experiment_with_trial_with_ttl),
    ("Experiment", get_experiment_with_data),
    ("Experiment", get_experiment_with_map_data_type),
    ("Experiment", get_branin_experiment_with_timestamp_map_metric),
    ("Experiment", get_experiment_with_map_data),
    ("FactorialMetric", get_factorial_metric),
    ("FixedParameter", get_fixed_parameter),
    ("FixedParameter", partial(get_fixed_parameter, with_dependents=True)),
    ("GammaPrior", get_gamma_prior),
    (
        "GenerationStep",
        partial(
            GenerationStep,
            generator=Generators.SOBOL,
            num_trials=5,
            min_trials_observed=3,
            use_all_trials_in_exp=True,
        ),
    ),
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
    (
        "GenerationStrategy",
        partial(sobol_gpei_generation_node_gs, with_model_selection=True),
    ),
    (
        "GenerationStrategy",
        partial(sobol_gpei_generation_node_gs, with_auto_transition=True),
    ),
    (
        "GenerationStrategy",
        partial(sobol_gpei_generation_node_gs, with_previous_node=True),
    ),
    (
        "GenerationStrategy",
        partial(sobol_gpei_generation_node_gs, with_trial_type=True),
    ),
    (
        "GenerationStrategy",
        partial(sobol_gpei_generation_node_gs, with_input_constructors_all_n=True),
    ),
    (
        "GenerationStrategy",
        partial(
            sobol_gpei_generation_node_gs, with_input_constructors_remaining_n=True
        ),
    ),
    (
        "GenerationStrategy",
        partial(sobol_gpei_generation_node_gs, with_input_constructors_repeat_n=True),
    ),
    (
        "GenerationStrategy",
        partial(
            sobol_gpei_generation_node_gs, with_input_constructors_target_trial=True
        ),
    ),
    (
        "GenerationStrategy",
        partial(sobol_gpei_generation_node_gs, with_unlimited_gen_mbm=True),
    ),
    (
        "GenerationStrategy",
        partial(sobol_gpei_generation_node_gs, with_is_SOO_transition=True),
    ),
    ("GeneratorRun", get_generator_run),
    ("Hartmann6Metric", get_hartmann_metric),
    ("HierarchicalSearchSpace", get_hierarchical_search_space),
    ("ImprovementGlobalStoppingStrategy", get_improvement_global_stopping_strategy),
    ("Interval", get_interval),
    ("MapData", get_map_data),
    ("MapMetric", partial(get_map_metric, name="test")),
    ("Metric", get_metric),
    ("MultiObjective", get_multi_objective),
    ("MultiObjectiveOptimizationConfig", get_multi_objective_optimization_config),
    ("MultiTypeExperiment", get_multi_type_experiment),
    ("MultiTypeExperiment", partial(get_multi_type_experiment, add_trials=True)),
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
        get_percentile_early_stopping_strategy_with_non_objective_metric_signature,
    ),
    ("ParameterConstraint", get_parameter_constraint),
    ("RangeParameter", get_range_parameter),
    ("ScalarizedObjective", get_scalarized_objective),
    ("OrchestratorOptions", get_default_orchestrator_options),
    ("OrchestratorOptions", get_orchestrator_options_batch_trial),
    ("SearchSpace", get_search_space),
    ("SumConstraint", get_sum_constraint1),
    ("SumConstraint", get_sum_constraint2),
    ("Surrogate", get_surrogate),
    ("SyntheticRunner", get_synthetic_runner),
    ("Type[Acquisition]", get_acquisition_type),
    ("Type[AcquisitionFunction]", get_acquisition_function_type),
    ("Type[Model]", get_model_type),
    ("Type[MarginalLogLikelihood]", get_mll_type),
    ("Type[Transform]", get_transform_type),
    ("Type[Transform]", lambda: Transform),
    ("Type[InputTransform]", get_input_transform_type),
    ("Type[OutcomeTransform]", get_outcome_transfrom_type),
    ("Type[TransformToNewSQ]", get_to_new_sq_transform_type),
    ("TransitionCriterionList", get_trial_based_criterion),
    ("ThresholdEarlyStoppingStrategy", get_threshold_early_stopping_strategy),
    ("Trial", get_trial),
    ("WinsorizationConfig", get_winsorization_config),
]


class JSONStoreTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.maxDiff = None
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

            # Dump and reload the json_object to simulate serialization round-trip.
            json_str = json.dumps(json_object)
            json_object = json.loads(json_str)

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
                        step._transition_criteria = []
                    for step in converted_object._steps:
                        step._transition_criteria = []
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

    def test_EncodeDecode_dataclass_with_initvar(self) -> None:
        @dataclasses.dataclass
        class TestDataclass:
            a_field: int
            not_a_field: dataclasses.InitVar[int | None] = None

            def __post_init__(self, doesnt_serialize: None) -> None:
                self.not_a_field = 1

        obj = TestDataclass(a_field=-1)
        as_json = object_to_json(obj=obj)
        self.assertEqual(as_json, {"__type": "TestDataclass", "a_field": -1})
        recovered = object_from_json(
            object_json=as_json, decoder_registry={"TestDataclass": TestDataclass}
        )
        self.assertEqual(recovered.a_field, -1)
        self.assertEqual(recovered.not_a_field, 1)
        self.assertEqual(obj, recovered)

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
        self.assertIsInstance(new_generation_strategy._steps[0].generator, Generators)
        # Model has not yet been initialized on this GS since it hasn't generated
        # anything yet.
        self.assertIsNone(new_generation_strategy.adapter)

        # Check that we can encode and decode the generation strategy after
        # it has generated some generator runs. Since we now need to `gen`,
        # we remove the fake callable kwarg we added, since model does not
        # expect it.
        generation_strategy = get_generation_strategy(with_callable_model_kwarg=False)
        gr = generation_strategy.gen_single_trial(experiment)
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
        self.assertIsInstance(new_generation_strategy._steps[0].generator, Generators)

        # Check that we can encode and decode the generation strategy after
        # it has generated some trials and been updated with some data.
        generation_strategy = new_generation_strategy
        experiment.new_trial(gr)  # Add previously generated GR as trial.
        # Make generation strategy aware of the trial's data via `gen`.
        generation_strategy.gen_single_trial(experiment, data=get_branin_data())
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
        self.assertIsInstance(new_generation_strategy._steps[0].generator, Generators)

    def test_decode_map_data_backward_compatible(self) -> None:
        with self.subTest("Multiple map keys"):
            data_with_two_map_keys_json = {
                "df": {
                    "__type": "DataFrame",
                    "value": (
                        '{"trial_index":{"0":0,"1":0},"arm_name":{"0":"0_0","1":"0_0"},'
                        '"metric_name":{"0":"a","1":"a"},"mean":{"0":0.0,"1":0.0},'
                        '"sem":{"0":0.0,"1":0.0},"epoch":{"0":0.0,"1":1.0},'
                        '"timestamps":{"0":3.0,"1":4.0}}'
                    ),
                },
                "map_key_infos": [
                    {"key": "epoch", "default_value": nan},
                    {"key": "timestamps", "default_value": nan},
                ],
                "__type": "MapData",
            }
            with warnings.catch_warnings(record=True) as warning_list:
                map_data = object_from_json(
                    data_with_two_map_keys_json,
                    decoder_registry=CORE_DECODER_REGISTRY,
                    class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
                )
            self.assertIn(
                "Received multiple map keys. All except ", str(warning_list[0].message)
            )
            self.assertIn("will be renamed to step", str(warning_list[1].message))

            # The "timestamp" map key will be silently dropped, and "epoch" will
            # be renamed to MAP_KEY
            self.assertIn(MAP_KEY, map_data.full_df.columns)
            # Either 'epoch' or 'timestamps' could have been kept
            progression = map_data.full_df[MAP_KEY].tolist()
            self.assertTrue(progression == [0.0, 1.0] or progression == [3.0, 4.0])

        with self.subTest("Single map key"):
            data_json = {
                "df": {
                    "__type": "DataFrame",
                    "value": (
                        '{"trial_index":{"0":0,"1":0},"arm_name":{"0":"0_0","1":"0_0"},'
                        '"metric_name":{"0":"a","1":"a"},"mean":{"0":0.0,"1":0.0},'
                        '"sem":{"0":0.0,"1":0.0},"epoch":{"0":0.0,"1":1.0}}'
                    ),
                },
                "map_key_infos": [{"key": "epoch", "default_value": nan}],
                "__type": "MapData",
            }
            with warnings.catch_warnings(record=True) as warning_list:
                map_data = object_from_json(
                    data_json,
                    decoder_registry=CORE_DECODER_REGISTRY,
                    class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
                )
            self.assertIn(
                f"epoch will be renamed to {MAP_KEY}", str(warning_list[0].message)
            )
            self.assertIn(MAP_KEY, map_data.full_df.columns)
            self.assertEqual(map_data.full_df[MAP_KEY].tolist(), [0.0, 1.0])
            # No warning about multiple map keys
            self.assertFalse(any("Received multiple" in str(w) for w in warning_list))

        with self.subTest("No map key"):
            data_json = {
                "df": {
                    "__type": "DataFrame",
                    "value": (
                        '{"metric_name":{},"arm_name":{},"trial_index":{},"mean":{}'
                        ',"sem":{}}'
                    ),
                },
                "map_key_infos": [],
                "__type": "MapData",
            }
            map_data = object_from_json(
                data_json,
                decoder_registry=CORE_DECODER_REGISTRY,
                class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
            )
            self.assertIsInstance(map_data, MapData)
            self.assertEqual(len(map_data.df), 0)

    def test_decode_data_backward_compatible(self) -> None:
        empty_df_json = {
            "__type": "DataFrame",
            "value": (
                '{"metric_name":{},"arm_name":{},"trial_index":{},"mean":{}'
                ',"sem":{}}'
            ),
        }
        with self.subTest("Description is None"):
            data_json = {"df": empty_df_json, "description": None, "__type": "Data"}
            data = object_from_json(
                data_json,
                decoder_registry=CORE_DECODER_REGISTRY,
                class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
            )
            self.assertIsInstance(data, Data)

        with self.subTest("Description is not None"):
            data_json = {
                "df": empty_df_json,
                "description": "description",
                "__type": "Data",
            }
            data = object_from_json(
                data_json,
                decoder_registry=CORE_DECODER_REGISTRY,
                class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
            )
            self.assertIsInstance(data, Data)

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

    def test_encode_decode_surrogate_spec(self) -> None:
        # Test SurrogateSpec separately since the GPyTorch components
        # fail simple equality checks.
        for org_object in (
            get_surrogate_spec_with_default(),
            get_surrogate_spec_with_lognormal(),
        ):
            converted_object = object_from_json(
                object_to_json(
                    org_object,
                    encoder_registry=CORE_ENCODER_REGISTRY,
                    class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
                ),
                decoder_registry=CORE_DECODER_REGISTRY,
                class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
            )
            org_as_dict = dataclasses.asdict(org_object)["model_configs"][0]
            converted_as_dict = dataclasses.asdict(converted_object)["model_configs"][0]
            # Covar module kwargs will fail comparison. Manually compare.
            org_covar_kwargs = org_as_dict.pop("covar_module_options")
            converted_covar_kwargs = converted_as_dict.pop("covar_module_options")
            self.assertEqual(org_covar_kwargs.keys(), converted_covar_kwargs.keys())
            for k in org_covar_kwargs:
                org_ = org_covar_kwargs[k]
                converted_ = converted_covar_kwargs[k]
                if isinstance(org_, torch.nn.Module):
                    self.assertEqual(org_.__class__, converted_.__class__)
                    self.assertEqual(org_.state_dict(), converted_.state_dict())
                else:
                    self.assertEqual(org_, converted_)
            # Compare the rest.
            self.assertEqual(org_as_dict, converted_as_dict)

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
        """Support for callables within model kwargs on GeneratorSpecs stored on
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

    def test_observation_features_backward_compatibility(self) -> None:
        json = {
            "__type": "ObservationFeatures",
            "parameters": {"x1": 0.0},
            "trial_index": 0,
            "random_split": 4,
        }
        with self.assertLogs(logger="ax", level="WARNING") as cm:
            decoded = object_from_json(object_json=json)
        self.assertTrue(any("random_split" in w for w in cm.output))
        self.assertIsInstance(decoded, ObservationFeatures)
        self.assertEqual(decoded.parameters, {"x1": 0.0})
        self.assertEqual(decoded.trial_index, 0)

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
        # Test that we can load a generation step with deprecated kwargs.
        json = {
            "__type": "GenerationStep",
            "model": {"__type": "Generators", "name": "BOTORCH_MODULAR"},
            "num_trials": 5,
            "min_trials_observed": 0,
            "completion_criteria": [],
            "max_parallelism": None,
            "use_update": False,
            "enforce_num_trials": True,
            "model_kwargs": {
                "fit_on_update": False,
                "torch_dtype": torch.double,
                "status_quo_name": "status_quo",
                "status_quo_features": None,
                "other_kwarg": 5,
                "fit_out_of_design": True,
                "fit_abandoned": True,
                "fit_only_completed_map_metrics": True,
            },
            "model_gen_kwargs": {},
            "index": -1,
            "should_deduplicate": False,
        }
        generation_step = object_from_json(json)
        self.assertIsInstance(generation_step, GenerationStep)
        self.assertEqual(generation_step.model_kwargs, {"other_kwarg": 5})
        self.assertEqual(generation_step.generator, Generators.BOTORCH_MODULAR)

    def test_generator_run_backwards_compatibility(self) -> None:
        # Test that we can load a generator run with deprecated kwargs.
        json = {
            "__type": "GeneratorRun",
            "arms": [
                {
                    "__type": "Arm",
                    "parameters": {"x1": 0.17783968150615692, "x2": 0.8026256756857038},
                    "name": None,
                }
            ],
            "weights": [1.0],
            "optimization_config": None,
            "search_space": None,
            "time_created": {
                "__type": "datetime",
                "value": "2025-02-27 07:06:36.675760",
            },
            "model_predictions": None,
            "best_arm_predictions": None,
            "generator_run_type": None,
            "index": None,
            "fit_time": 0.00037617841735482216,
            "gen_time": 0.00448690727353096,
            "model_key": "Sobol",
            "model_kwargs": {
                "deduplicate": False,
                "seed": None,
                "torch_dtype": None,
            },
            "bridge_kwargs": {
                "transforms": {},
                "transform_configs": None,
                "status_quo_name": None,
                "status_quo_features": None,
                "optimization_config": None,
                "fit_on_update": False,
                "fit_out_of_design": False,
                "fit_abandoned": False,
                "fit_tracking_metrics": True,
                "fit_on_init": True,
            },
            "gen_metadata": {
                "model_fit_quality": None,
            },
            "model_state_after_gen": None,
            "generation_step_index": None,
            "candidate_metadata_by_arm_signature": None,
            "generation_node_name": None,
        }
        generator_run = object_from_json(json)
        self.assertIsInstance(generator_run, GeneratorRun)
        self.assertEqual(
            generator_run._model_kwargs,
            {"deduplicate": False, "seed": None},
        )
        self.assertEqual(
            generator_run._bridge_kwargs,
            {
                "transforms": {},
                "transform_configs": None,
                "optimization_config": None,
                "fit_tracking_metrics": True,
                "fit_on_init": True,
            },
        )

    def test_generation_node_backwards_compatibility(self) -> None:
        # Checks that deprecated input constructors are discarded gracefully.
        json = {
            "node_name": "Test",
            "model_specs": [
                {
                    "__type": "GeneratorSpec",
                    "model_enum": {"__type": "Generators", "name": "BOTORCH_MODULAR"},
                    "model_kwargs": {
                        "transforms": [
                            {
                                "__type": "Type[Transform]",
                                "index_in_registry": 6,
                                "transform_type": (
                                    "<class 'ax.adapter.transforms" ".one_hot.OneHot'>"
                                ),
                            },
                            {
                                "__type": "Type[Transform]",
                                "index_in_registry": 5,
                                "transform_type": (
                                    "<class 'ax.adapter.transforms.log.Log'>"
                                ),
                            },
                        ]
                    },
                    "model_gen_kwargs": {
                        "model_gen_options": {
                            "optimizer_kwargs": {"num_restarts": 10},
                            "acquisition_function_kwargs": {},
                        }
                    },
                }
            ],
            "best_model_selector": None,
            "should_deduplicate": False,
            "transition_criteria": [
                {
                    "transition_to": "BOTORCH_MODULAR",
                    "auxiliary_experiment_purposes_to_include": None,
                    "auxiliary_experiment_purposes_to_exclude": [],
                    "block_transition_if_unmet": True,
                    "block_gen_if_met": False,
                    "continue_trial_generation": False,
                    "__type": "AuxiliaryExperimentCheck",
                }
            ],
            "model_spec_to_gen_from": None,
            "previous_node_name": None,
            "trial_type": {"__type": "Keys", "name": "SHORT_RUN"},
            "input_constructors": {
                "N": {"__type": "NodeInputConstructors", "name": "REMAINING_N"},
                "FIXED_FEATURES": {
                    "__type": "NodeInputConstructors",
                    "name": "TARGET_TRIAL_FIXED_FEATURES",
                },
                "STATUS_QUO_FEATURES": {
                    "__type": "NodeInputConstructors",
                    "name": "STATUS_QUO_FEATURES",
                },
            },
        }
        node = generation_node_from_json(json)
        self.assertIsInstance(node, GenerationNode)
        self.assertEqual(node.name, "Test")
        self.assertEqual(len(node.transition_criteria), 1)
        # Status quo is discarded, so we have 2 input constructors left.
        self.assertEqual(len(node.input_constructors), 2)
        # Check that transforms got correctly deserialized.
        self.assertEqual(
            node.generator_specs[0].model_kwargs["transforms"],
            [OneHot, Log],
        )

    def test_SobolQMCNormalSampler(self) -> None:
        # This fails default equality checks, so testing it separately.
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]))
        sampler_json = object_to_json(
            sampler,
            encoder_registry=CORE_ENCODER_REGISTRY,
            class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
        )
        sampler_loaded = object_from_json(
            sampler_json,
            decoder_registry=CORE_DECODER_REGISTRY,
            class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
        )
        self.assertIsInstance(sampler_loaded, SobolQMCNormalSampler)
        self.assertEqual(sampler.sample_shape, sampler_loaded.sample_shape)
        self.assertEqual(sampler.seed, sampler_loaded.seed)

    def test_mbm_backwards_compatibility(self) -> None:
        # This is json of get_botorch_model_with_surrogate_specs() before D64875988.
        object_json = {
            "__type": "BoTorchModel",
            "acquisition_class": {
                "__type": "Type[Acquisition]",
                "index": "Acquisition",
                "class": (
                    "<class 'ax.models.torch.botorch_modular.acquisition.Acquisition'>"
                ),
            },
            "acquisition_options": {},
            "surrogate": None,
            "surrogate_specs": {
                "name": {
                    "__type": "SurrogateSpec",
                    "botorch_model_class": None,
                    "botorch_model_kwargs": {"some_option": "some_value"},
                    "mll_class": {
                        "__type": "Type[MarginalLogLikelihood]",
                        "index": "ExactMarginalLogLikelihood",
                        "class": (
                            "<class 'gpytorch.mlls.marginal_log_likelihood."
                            "MarginalLogLikelihood'>"
                        ),
                    },
                    "mll_kwargs": {},
                    "covar_module_class": None,
                    "covar_module_kwargs": None,
                    "likelihood_class": None,
                    "likelihood_kwargs": None,
                    "input_transform_classes": None,
                    "input_transform_options": None,
                    "outcome_transform_classes": None,
                    "outcome_transform_options": None,
                    "allow_batched_models": True,
                    "outcomes": [],
                }
            },
            "botorch_acqf_class": None,
            "refit_on_cv": False,
            "warm_start_refit": True,
        }
        expected_object = get_botorch_model_with_surrogate_spec(with_covar_module=False)
        expected_object.surrogate_spec.model_configs[0].input_transform_classes = None
        expected_object.surrogate_spec.model_configs[0].name = "from deprecated args"
        # The new default value is None; we need to manually set it to the old value
        self.assertIsNone(
            none_throws(expected_object.surrogate_spec).model_configs[0].mll_class
        )
        expected_object.surrogate_spec.model_configs[
            0
        ].mll_class = ExactMarginalLogLikelihood
        self.assertEqual(object_from_json(object_json), expected_object)

    def test_mbm_backwards_compatibility_2(self) -> None:
        # Ensure Modular BoTorch Generators saved before the Multi-surrogate
        # MBM refactor in D41637384 can be loaded and converted from using a
        # ListSurrogate to a Surrogate
        converted_object = object_from_json(
            get_legacy_list_surrogate_generation_step_as_dict()
        )
        new_object = get_surrogate_generation_step()
        # Converted object is a generation step without a strategy associated with it;
        # unset the generation strategy of the new object too, to match.
        new_object._generation_strategy = None
        self.assertEqual(converted_object, new_object)

        # Check that we can deserialize Surrogate with input_transform
        # & outcome_transform kwargs.
        converted_object = object_from_json(get_surrogate_as_dict())
        new_object = Surrogate(
            surrogate_spec=SurrogateSpec(
                model_configs=[
                    ModelConfig(
                        mll_class=ExactMarginalLogLikelihood,
                        input_transform_classes=None,
                        name="from deprecated args",
                    )
                ],
                allow_batched_models=False,
            ),
        )
        self.assertEqual(converted_object, new_object)

        # Check with SurrogateSpec.
        for model_class, legacy_input_transform in [
            (None, False),  # None maps to SingleTaskGP.
            ("FixedNoiseGP", True),
        ]:
            converted_object = object_from_json(
                get_surrogate_spec_as_dict(
                    model_class=model_class,
                    with_legacy_input_transform=legacy_input_transform,
                ),
            )
            extra_args = {}
            if legacy_input_transform:
                extra_args["input_transform_classes"] = [Normalize]
                extra_args["input_transform_options"] = {
                    "Normalize": {
                        "d": 7,
                        "indices": None,
                        "bounds": None,
                        "batch_shape": torch.Size([]),
                        "transform_on_train": True,
                        "transform_on_eval": True,
                        "transform_on_fantasize": True,
                        "reverse": False,
                        "min_range": 1e-08,
                        "learn_bounds": False,
                    }
                }
            else:
                extra_args["input_transform_classes"] = None
            new_object = SurrogateSpec(
                model_configs=[
                    ModelConfig(
                        botorch_model_class=SingleTaskGP,
                        mll_class=ExactMarginalLogLikelihood,
                        name="from deprecated args",
                        **extra_args,
                    )
                ],
                allow_batched_models=False,
            )
            self.assertEqual(converted_object, new_object)

    def test_multi_objective_backwards_compatibility(self) -> None:
        object_json = {
            "__type": "MultiObjective",
            "objectives": [
                {
                    "__type": "Objective",
                    "metric": {
                        "name": "m1",
                        "lower_is_better": None,
                        "properties": {},
                        "__type": "Metric",
                    },
                    "minimize": False,
                },
                {
                    "__type": "Objective",
                    "metric": {
                        "name": "m3",
                        "lower_is_better": True,
                        "properties": {},
                        "__type": "Metric",
                    },
                    "minimize": True,
                },
            ],
            "weights": [1.0, 1.0],
        }
        deserialized_object = object_from_json(object_json)
        expected_object = get_multi_objective()
        self.assertEqual(deserialized_object, expected_object)

    def test_optimization_config_with_pruning_target_json_roundtrip(self) -> None:
        # Test that OptimizationConfig with pruning_target_parameterization can
        # be serialized/deserialized correctly

        # Setup: create OptimizationConfig with pruning_target_parameterization
        pruning_target_parameterization = get_arm()
        optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric("test_metric"), minimize=False),
            pruning_target_parameterization=pruning_target_parameterization,
        )

        # Execute: serialize and deserialize through JSON
        json_data = object_to_json(
            optimization_config,
            encoder_registry=CORE_ENCODER_REGISTRY,
            class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
        )

        # Simulate full serialization round-trip
        json_str = json.dumps(json_data)
        json_data = json.loads(json_str)

        deserialized_config = object_from_json(
            json_data,
            decoder_registry=CORE_DECODER_REGISTRY,
            class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
        )

        # Assert: confirm pruning_target_parameterization is preserved correctly
        self.assertEqual(optimization_config, deserialized_config)
        self.assertIsNotNone(deserialized_config.pruning_target_parameterization)
        self.assertEqual(
            optimization_config.pruning_target_parameterization,
            deserialized_config.pruning_target_parameterization,
        )

    def test_multi_objective_optimization_config_with_pruning_target_json_roundtrip(
        self,
    ) -> None:
        # Test that MultiObjectiveOptimizationConfig with
        # pruning_target_parameterization can be
        # serialized/deserialized correctly

        # Setup: create MultiObjectiveOptimizationConfig with
        # pruning_target_parameterization
        pruning_target_parameterization = get_arm()
        multi_objective_config = MultiObjectiveOptimizationConfig(
            objective=get_multi_objective(),
            pruning_target_parameterization=pruning_target_parameterization,
        )

        # Execute: serialize and deserialize through JSON
        json_data = object_to_json(
            multi_objective_config,
            encoder_registry=CORE_ENCODER_REGISTRY,
            class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
        )

        # Simulate full serialization round-trip
        json_str = json.dumps(json_data)
        json_data = json.loads(json_str)

        deserialized_config = object_from_json(
            json_data,
            decoder_registry=CORE_DECODER_REGISTRY,
            class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
        )

        # Assert: confirm pruning_target_parameterization is preserved correctly
        self.assertEqual(multi_objective_config, deserialized_config)
        self.assertIsNotNone(deserialized_config.pruning_target_parameterization)
        self.assertEqual(
            multi_objective_config.pruning_target_parameterization,
            deserialized_config.pruning_target_parameterization,
        )

    def test_preference_optimization_config_with_pruning_target_json_roundtrip(
        self,
    ) -> None:
        # Test that PreferenceOptimizationConfig with
        # pruning_target_parameterization can be
        # serialized/deserialized correctly

        # Setup: create PreferenceOptimizationConfig with
        # pruning_target_parameterization
        pruning_target_parameterization = get_arm()
        preference_config = PreferenceOptimizationConfig(
            objective=get_multi_objective(),
            pruning_target_parameterization=pruning_target_parameterization,
            preference_profile_name="default",
        )

        # Execute: serialize and deserialize through JSON
        json_data = object_to_json(
            preference_config,
            encoder_registry=CORE_ENCODER_REGISTRY,
            class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
        )

        # Simulate full serialization round-trip
        json_str = json.dumps(json_data)
        json_data = json.loads(json_str)

        deserialized_config = object_from_json(
            json_data,
            decoder_registry=CORE_DECODER_REGISTRY,
            class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
        )

        # Assert: confirm pruning_target_parameterization is preserved correctly
        self.assertEqual(preference_config, deserialized_config)
        self.assertIsNotNone(deserialized_config.pruning_target_parameterization)
        self.assertEqual(
            preference_config.pruning_target_parameterization,
            deserialized_config.pruning_target_parameterization,
        )

    def test_optimization_config_with_none_pruning_target_json_roundtrip(self) -> None:
        # Test that OptimizationConfig with
        # pruning_target_parameterization=None is handled correctly

        # Setup: create OptimizationConfig without
        # pruning_target_parameterization
        optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric("test_metric"), minimize=False),
            pruning_target_parameterization=None,
        )

        # Execute: serialize and deserialize through JSON
        json_data = object_to_json(
            optimization_config,
            encoder_registry=CORE_ENCODER_REGISTRY,
            class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
        )

        # Simulate full serialization round-trip
        json_str = json.dumps(json_data)
        json_data = json.loads(json_str)

        deserialized_config = object_from_json(
            json_data,
            decoder_registry=CORE_DECODER_REGISTRY,
            class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
        )

        # Assert: confirm pruning_target_parameterization remains None
        self.assertEqual(optimization_config, deserialized_config)
        self.assertIsNone(deserialized_config.pruning_target_parameterization)

    def test_experiment_with_pruning_target_json_roundtrip(self) -> None:
        # Test that Experiment with optimization_config containing
        # pruning_target_parameterization is
        # serialized correctly

        # Setup: create experiment with pruning_target_parameterization in optimization
        # config
        experiment = get_branin_experiment()
        pruning_target_parameterization = get_arm()
        optimization_config = none_throws(
            experiment.optimization_config
        ).clone_with_args(
            pruning_target_parameterization=pruning_target_parameterization
        )
        experiment.optimization_config = optimization_config

        # Execute: save and load experiment through JSON
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
            save_experiment(
                experiment,
                f.name,
                encoder_registry=CORE_ENCODER_REGISTRY,
                class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
            )
            loaded_experiment = load_experiment(
                f.name,
                decoder_registry=CORE_DECODER_REGISTRY,
                class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
            )

        # Cleanup
        os.remove(f.name)

        # Assert: confirm experiment and pruning_target_parameterization
        # are preserved correctly
        self.assertEqual(experiment, loaded_experiment)
        self.assertIsNotNone(
            none_throws(
                loaded_experiment.optimization_config
            ).pruning_target_parameterization
        )
        self.assertEqual(
            none_throws(experiment.optimization_config).pruning_target_parameterization,
            none_throws(
                loaded_experiment.optimization_config
            ).pruning_target_parameterization,
        )

    def test_multi_objective_from_json_warning(self) -> None:
        objectives = [get_objective()]

        # Test that warning is logged when deprecated kwargs are passed
        with self.assertLogs("ax.utils.common.kwargs", level="WARNING") as cm:
            multi_objective_from_json(
                objectives=objectives,
                weights=[1.0],
                metrics=["test_metric"],
                minimize=True,
            )

        # Verify the warning message
        self.assertTrue(
            any("Found unexpected kwargs" in warning for warning in cm.output)
        )

    def test_choice_parameter_bypass_cardinality_check_encode_failure(self) -> None:
        choice_parameter = ChoiceParameter(
            name="test_choice",
            parameter_type=ParameterType.INT,
            values=[1, 2, 3],
            bypass_cardinality_check=True,
        )
        with self.assertRaisesRegex(
            UnsupportedError,
            "`bypass_cardinality_check` should only be set to True "
            "when constructing parameters within the modeling layer. It is not "
            "supported for storage.",
        ):
            choice_parameter_to_dict(choice_parameter)

    def test_surrogate_spec_backwards_compatibility(self) -> None:
        # This is an invalid example that has both deprecated args
        # and model config specified. Deprecated args will be ignored.
        object_json = {
            "__type": "SurrogateSpec",
            "botorch_model_class": {
                "__type": "Type[Model]",
                "index": "MultiTaskGP",
                "class": "<class 'botorch.models.model.Model'>",
            },
            "botorch_model_kwargs": {"dummy": 5},
            "mll_class": {
                "__type": "Type[MarginalLogLikelihood]",
                "index": "ExactMarginalLogLikelihood",
                "class": (
                    "<class 'gpytorch.mlls.marginal_log_likelihood."
                    "MarginalLogLikelihood'>"
                ),
            },
            "mll_kwargs": {},
            "covar_module_class": None,
            "covar_module_kwargs": None,
            "likelihood_class": None,
            "likelihood_kwargs": None,
            "input_transform_classes": None,
            "input_transform_options": None,
            "outcome_transform_classes": None,
            "outcome_transform_options": None,
            "allow_batched_models": True,
            "model_configs": [
                {
                    "__type": "ModelConfig",
                    "botorch_model_class": {
                        "__type": "Type[Model]",
                        "index": "SingleTaskGP",
                        "class": "<class 'botorch.models.model.Model'>",
                    },
                    "model_options": {},
                    "mll_class": {
                        "__type": "Type[MarginalLogLikelihood]",
                        "index": "ExactMarginalLogLikelihood",
                        "class": (
                            "<class 'gpytorch.mlls.marginal_log_likelihood."
                            "MarginalLogLikelihood'>"
                        ),
                    },
                    "mll_options": {},
                    "input_transform_classes": None,
                    "input_transform_options": {},
                    "outcome_transform_classes": [
                        {
                            "__type": "Type[OutcomeTransform]",
                            "index": "Standardize",
                            "class": (
                                "<class 'botorch.models.transforms.outcome."
                                "OutcomeTransform'>"
                            ),
                        }
                    ],
                    "outcome_transform_options": {},
                    "covar_module_class": {
                        "__type": "Type[Kernel]",
                        "index": "ScaleMaternKernel",
                        "class": "<class 'gpytorch.kernels.kernel.Kernel'>",
                    },
                    "covar_module_options": {},
                    "likelihood_class": None,
                    "likelihood_options": {},
                }
            ],
            "metric_to_model_configs": {},
            "eval_criterion": "Rank correlation",
            "outcomes": [],
            "use_posterior_predictive": False,
        }
        deserialized_object = object_from_json(object_json)
        expected_object = SurrogateSpec(
            model_configs=[
                ModelConfig(
                    botorch_model_class=SingleTaskGP,
                    covar_module_class=ScaleMaternKernel,
                    mll_class=ExactMarginalLogLikelihood,
                    outcome_transform_classes=[Standardize],
                    input_transform_classes=None,
                )
            ]
        )
        self.assertEqual(deserialized_object, expected_object)

    def test_model_registry_backwards_compatibility(self) -> None:
        # Check that deprecated model registry entries can be loaded.
        # Check for models with listed replacements.
        for name, replacement in _DEPRECATED_MODEL_TO_REPLACEMENT.items():
            with self.assertLogs(logger="ax", level="ERROR"):
                from_json = object_from_json({"__type": "Generators", "name": name})
            self.assertEqual(from_json, Generators[replacement])
        # Check for non-deprecated models.
        from_json = object_from_json({"__type": "Models", "name": "BO_MIXED"})
        self.assertEqual(from_json, Generators.BO_MIXED)
        # Check for models with no replacement.
        with self.assertRaisesRegex(KeyError, "nonexistent"):
            object_from_json({"__type": "Models", "name": "nonexistent_model"})

    def test_optimization_config_backwards_compatibility(self) -> None:
        # Check that opt config json with risk measure can be loaded.
        opt_config = get_optimization_config()
        opt_config_json = object_to_json(opt_config)
        # Add risk measure.
        opt_config_json["risk_measure"] = None
        # Decode and compare.
        decoded_opt_config = object_from_json(opt_config_json)
        self.assertEqual(opt_config, decoded_opt_config)
