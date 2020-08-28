#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from functools import partial

import numpy as np
import torch
from ax.benchmark.benchmark_problem import SimpleBenchmarkProblem
from ax.core.metric import Metric
from ax.core.runner import Runner
from ax.exceptions.storage import JSONDecodeError, JSONEncodeError
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.registry import Models
from ax.storage.json_store.decoder import (
    generation_strategy_from_json,
    object_from_json,
)
from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.storage.utils import EncodeDecodeFieldsMap, remove_prefix
from ax.utils.common.testutils import TestCase
from ax.utils.measurement.synthetic_functions import ackley, branin, from_botorch
from ax.utils.testing.benchmark_stubs import (
    get_branin_benchmark_problem,
    get_branin_simple_benchmark_problem,
    get_mult_simple_benchmark_problem,
    get_sum_simple_benchmark_problem,
)
from ax.utils.testing.core_stubs import (
    get_acquisition_function_type,
    get_acquisition_type,
    get_arm,
    get_augmented_branin_metric,
    get_augmented_hartmann_metric,
    get_batch_trial,
    get_botorch_model,
    get_botorch_model_with_default_acquisition_class,
    get_branin_data,
    get_branin_experiment,
    get_branin_metric,
    get_choice_parameter,
    get_experiment_with_batch_and_single_trial,
    get_experiment_with_data,
    get_factorial_metric,
    get_fixed_parameter,
    get_generator_run,
    get_hartmann_metric,
    get_metric,
    get_mll_type,
    get_model_type,
    get_multi_objective,
    get_multi_type_experiment,
    get_objective,
    get_optimization_config,
    get_order_constraint,
    get_outcome_constraint,
    get_parameter_constraint,
    get_range_parameter,
    get_scalarized_objective,
    get_search_space,
    get_simple_experiment_with_batch_trial,
    get_sum_constraint1,
    get_sum_constraint2,
    get_surrogate,
    get_synthetic_runner,
    get_trial,
)
from ax.utils.testing.modeling_stubs import (
    get_generation_strategy,
    get_observation_features,
    get_transform_type,
)
from botorch.test_functions.synthetic import Ackley


TEST_CASES = [
    ("Arm", get_arm),
    ("AugmentedBraninMetric", get_augmented_branin_metric),
    ("AugmentedHartmannMetric", get_augmented_hartmann_metric),
    ("BatchTrial", get_batch_trial),
    ("BenchmarkProblem", get_branin_benchmark_problem),
    ("BoTorchModel", get_botorch_model),
    (
        "BoTorchModelWithDefaultAcquisitionClass",
        get_botorch_model_with_default_acquisition_class,
    ),
    ("BraninMetric", get_branin_metric),
    ("ChoiceParameter", get_choice_parameter),
    ("Experiment", get_experiment_with_batch_and_single_trial),
    ("Experiment", get_experiment_with_data),
    ("FactorialMetric", get_factorial_metric),
    ("FixedParameter", get_fixed_parameter),
    ("Hartmann6Metric", get_hartmann_metric),
    ("GenerationStrategy", partial(get_generation_strategy, with_experiment=True)),
    ("GeneratorRun", get_generator_run),
    ("Metric", get_metric),
    ("MultiObjective", get_multi_objective),
    ("MultiTypeExperiment", get_multi_type_experiment),
    ("ObservationFeatures", get_observation_features),
    ("Objective", get_objective),
    ("OptimizationConfig", get_optimization_config),
    ("OrderConstraint", get_order_constraint),
    ("OutcomeConstraint", get_outcome_constraint),
    ("ParameterConstraint", get_parameter_constraint),
    ("RangeParameter", get_range_parameter),
    ("ScalarizedObjective", get_scalarized_objective),
    ("SearchSpace", get_search_space),
    ("SimpleBenchmarkProblem", get_mult_simple_benchmark_problem),
    ("SimpleBenchmarkProblem", get_branin_simple_benchmark_problem),
    ("SimpleBenchmarkProblem", get_sum_simple_benchmark_problem),
    ("SimpleExperiment", get_simple_experiment_with_batch_trial),
    ("SumConstraint", get_sum_constraint1),
    ("SumConstraint", get_sum_constraint2),
    ("Surrogate", get_surrogate),
    ("SyntheticRunner", get_synthetic_runner),
    ("Type[Acquisition]", get_acquisition_type),
    ("Type[AcquisitionFunction]", get_acquisition_function_type),
    ("Type[Model]", get_model_type),
    ("Type[MarginalLogLikelihood]", get_mll_type),
    ("Type[Transform]", get_transform_type),
    ("Trial", get_trial),
]

# This map records discrepancies between Python and JSON representations,
# so that we can validate that the JSON representation is complete
# -- Sometimes a field appears in the Python object but not the JSON
#    (because it is not strictly necessary to store)
# -- Sometimes a field appears in both places but with different names
#    (because the name used by JSON is the name used by the constructor,
#    might not be the same used by the attribute)
ENCODE_DECODE_FIELD_MAPS = {
    "Experiment": EncodeDecodeFieldsMap(
        python_only=[
            "arms_by_signature",
            "arms_by_name",
            "trial_indices_by_status",
            "trials_have_ttl",
        ]
    ),
    "BatchTrial": EncodeDecodeFieldsMap(
        python_only=["experiment"], python_to_encoded={"BaseTrial__status": "status"}
    ),
    "SimpleBenchmarkProblem": EncodeDecodeFieldsMap(encoded_only=["function_name"]),
    "GenerationStrategy": EncodeDecodeFieldsMap(
        python_only=["uses_registered_models", "seen_trial_indices_by_status"],
        encoded_only=["had_initialized_model", "db_id"],
        python_to_encoded={"curr": "curr_index"},
    ),
    "GeneratorRun": EncodeDecodeFieldsMap(
        encoded_only=["arms", "weights"], python_only=["arm_weight_table"]
    ),
    "MultiTypeExperiment": EncodeDecodeFieldsMap(
        python_only=[
            "arms_by_signature",
            "arms_by_name",
            "metric_to_canonical_name",
            "metric_to_trial_type",
            "trial_indices_by_status",
            "trials_have_ttl",
            "trial_type_to_runner",
        ],
        encoded_only=[
            "_metric_to_canonical_name",
            "_metric_to_trial_type",
            "_trial_type_to_runner",
        ],
    ),
    "OrderConstraint": EncodeDecodeFieldsMap(
        python_only=["bound"],
        python_to_encoded={
            "lower_parameter": "lower_name",
            "upper_parameter": "upper_name",
        },
    ),
    "SimpleExperiment": EncodeDecodeFieldsMap(
        python_only=[
            "arms_by_signature",
            "arms_by_name",
            "evaluation_function",
            "trial_indices_by_status",
            "trials_have_ttl",
        ]
    ),
    "SumConstraint": EncodeDecodeFieldsMap(
        python_only=["constraint_dict", "parameters"]
    ),
    "Trial": EncodeDecodeFieldsMap(
        python_only=["experiment"], python_to_encoded={"BaseTrial__status": "status"}
    ),
    "Type[Acquisition]": EncodeDecodeFieldsMap(
        python_only=["_module__", "_doc__", "default_botorch_acqf_class"],
        encoded_only=["index", "class"],
    ),
    "Type[AcquisitionFunction]": EncodeDecodeFieldsMap(
        python_only=[
            "_module__",
            "_doc__",
            "_init__",
            "_abstractmethods__",
            "forward",
            "abc_impl",
        ],
        encoded_only=["index", "class"],
    ),
    "Type[Model]": EncodeDecodeFieldsMap(
        python_only=[
            "_module__",
            "_doc__",
            "_init__",
            "_abstractmethods__",
            "forward",
            "construct_inputs",
            "abc_impl",
        ],
        encoded_only=["index", "class"],
    ),
    "Type[MarginalLogLikelihood]": EncodeDecodeFieldsMap(
        python_only=["_module__", "_doc__", "_init__", "forward", "pyro_factor"],
        encoded_only=["index", "class"],
    ),
    "Type[Transform]": EncodeDecodeFieldsMap(
        python_only=[
            "transform_observation_features",
            "_module__",
            "_init__",
            "_doc__",
            "transform_search_space",
            "untransform_observation_features",
        ],
        encoded_only=["transform_type", "index_in_registry"],
    ),
}


class JSONStoreTest(TestCase):
    def setUp(self):
        self.experiment = get_experiment_with_batch_and_single_trial()

    def testJSONEncodeFailure(self):
        self.assertRaises(JSONEncodeError, object_to_json, Exception("foobar"))

    def testJSONDecodeFailure(self):
        self.assertRaises(JSONDecodeError, object_from_json, Exception("foobar"))
        self.assertRaises(JSONDecodeError, object_from_json, {"__type": "foobar"})

    def testSaveAndLoad(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
            save_experiment(self.experiment, f.name)
            loaded_experiment = load_experiment(f.name)
            self.assertEqual(loaded_experiment, self.experiment)
            os.remove(f.name)

    def testSaveValidation(self):
        with self.assertRaises(ValueError):
            save_experiment(self.experiment.trials[0], "test.json")

    def testValidateFilename(self):
        bad_filename = "test"
        self.assertRaises(ValueError, save_experiment, self.experiment, bad_filename)

    def testEncodeDecode(self):
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
            json_object = object_to_json(original_object)
            converted_object = object_from_json(json_object)

            if class_ == "SimpleExperiment":
                # Evaluation functions will be different, so need to do
                # this so equality test passes
                with self.assertRaises(Exception):
                    converted_object.evaluation_function()

                original_object.evaluation_function = None
                converted_object.evaluation_function = None

            self.assertEqual(
                original_object,
                converted_object,
                msg=f"Error encoding/decoding {class_}.",
            )

    def testEncoders(self):
        for class_, fake_func in TEST_CASES:
            original_object = fake_func()

            # We can skip metrics and runners; the encoders will automatically
            # handle the addition of new fields to these classes
            if isinstance(original_object, Metric) or isinstance(
                original_object, Runner
            ):
                continue

            json_object = object_to_json(original_object)

            object_keys = {
                remove_prefix(key, "_") for key in original_object.__dict__.keys()
            }
            json_keys = {key for key in json_object.keys() if key != "__type"}

            # Account for fields that appear in the Python object but not the JSON
            # and for fields that appear in both places but with different names
            if class_ in ENCODE_DECODE_FIELD_MAPS:
                map = ENCODE_DECODE_FIELD_MAPS[class_]
                for field in map.python_only:
                    json_keys.add(field)
                for field in map.encoded_only:
                    object_keys.add(field)
                for python, encoded in map.python_to_encoded.items():
                    json_keys.remove(encoded)
                    json_keys.add(python)
            # TODO: Remove this check if able. `_slotnames__` is not a class attribute
            # when testing locally, but it is a class attribute on Travis.
            if class_ == "Type[Model]":
                object_keys.discard("_slotnames__")
            self.assertEqual(
                object_keys,
                json_keys,
                msg=f"Mismatch between Python and JSON representation in {class_}.",
            )

    def testEncodeDecodeTorchTensor(self):
        x = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, device=torch.device("cpu")
        )
        expected_json = {
            "__type": "Tensor",
            "value": [[1.0, 2.0], [3.0, 4.0]],
            "dtype": {"__type": "torch_dtype", "value": "torch.float64"},
            "device": {"__type": "torch_device", "value": "cpu"},
        }
        x_json = object_to_json(x)
        self.assertEqual(expected_json, x_json)
        x2 = object_from_json(x_json)
        self.assertTrue(torch.equal(x, x2))

    def testDecodeGenerationStrategy(self):
        generation_strategy = get_generation_strategy()
        experiment = get_branin_experiment()
        gs_json = object_to_json(generation_strategy)
        new_generation_strategy = generation_strategy_from_json(gs_json)
        self.assertEqual(generation_strategy, new_generation_strategy)
        self.assertGreater(len(new_generation_strategy._steps), 0)
        self.assertIsInstance(new_generation_strategy._steps[0].model, Models)
        # Model has not yet been initialized on this GS since it hasn't generated
        # anything yet.
        self.assertIsNone(new_generation_strategy.model)

        # Check that we can encode and decode the generation strategy after
        # it has generated some generator runs.
        generation_strategy = new_generation_strategy
        gr = generation_strategy.gen(experiment)
        gs_json = object_to_json(generation_strategy)
        new_generation_strategy = generation_strategy_from_json(gs_json)
        self.assertEqual(generation_strategy, new_generation_strategy)
        self.assertIsInstance(new_generation_strategy._steps[0].model, Models)
        # Since this GS has now generated one generator run, model should have
        # been initialized and restored when decoding from JSON.
        self.assertIsInstance(new_generation_strategy.model, ModelBridge)

        # Check that we can encode and decode the generation strategy after
        # it has generated some trials and been updated with some data.
        generation_strategy = new_generation_strategy
        experiment.new_trial(gr)  # Add previously generated GR as trial.
        # Make generation strategy aware of the trial's data via `gen`.
        generation_strategy.gen(experiment, data=get_branin_data())
        gs_json = object_to_json(generation_strategy)
        new_generation_strategy = generation_strategy_from_json(gs_json)
        self.assertEqual(generation_strategy, new_generation_strategy)
        self.assertIsInstance(new_generation_strategy._steps[0].model, Models)
        self.assertIsInstance(new_generation_strategy.model, ModelBridge)

    def test_encode_decode_numpy(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(np.array_equal(arr, object_from_json(object_to_json(arr))))

    def testEncodeDecodeSimpleBenchmarkProblem(self):
        branin_problem = get_branin_simple_benchmark_problem()
        sum_problem = get_sum_simple_benchmark_problem()
        new_branin_problem = object_from_json(object_to_json(branin_problem))
        new_sum_problem = object_from_json(object_to_json(sum_problem))
        self.assertEqual(
            branin_problem.f(1, 2), new_branin_problem.f(1, 2), branin(1, 2)
        )
        self.assertEqual(sum_problem.f([1, 2]), new_sum_problem.f([1, 2]), 3)
        # Test using `from_botorch`.
        ackley_problem = SimpleBenchmarkProblem(
            f=from_botorch(Ackley()), noise_sd=0.0, minimize=True
        )
        new_ackley_problem = object_from_json(object_to_json(ackley_problem))
        self.assertEqual(
            ackley_problem.f(1, 2), new_ackley_problem.f(1, 2), ackley(1, 2)
        )

    def testRegistryAdditions(self):
        class MyRunner(Runner):
            def run():
                pass

            def staging_required():
                return False

        class MyMetric(Metric):
            pass

        register_metric(MyMetric)
        register_runner(MyRunner)

        experiment = get_experiment_with_batch_and_single_trial()
        experiment.runner = MyRunner()
        experiment.add_tracking_metric(MyMetric(name="my_metric"))
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
            save_experiment(experiment, f.name)
            loaded_experiment = load_experiment(f.name)
            self.assertEqual(loaded_experiment, experiment)
            os.remove(f.name)
