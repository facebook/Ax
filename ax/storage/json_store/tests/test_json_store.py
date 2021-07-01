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
from ax.storage.json_store.decoders import class_from_json
from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.encoders import botorch_modular_to_dict
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.registry import CLASS_ENCODER_REGISTRY
from ax.storage.json_store.save import save_experiment
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.utils.common.testutils import TestCase
from ax.utils.measurement.synthetic_functions import ackley, branin, from_botorch
from ax.utils.testing.benchmark_stubs import (
    get_branin_benchmark_problem,
    get_branin_simple_benchmark_problem,
    get_mult_simple_benchmark_problem,
    get_sum_simple_benchmark_problem,
)
from ax.utils.testing.core_stubs import (
    get_abandoned_arm,
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
    get_experiment_with_trial_with_ttl,
    get_experiment_with_map_data_type,
    get_factorial_metric,
    get_fixed_parameter,
    get_generator_run,
    get_map_data,
    get_hartmann_metric,
    get_list_surrogate,
    get_metric,
    get_mll_type,
    get_model_type,
    get_multi_objective,
    get_multi_objective_optimization_config,
    get_multi_type_experiment,
    get_objective,
    get_objective_threshold,
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
    ("AbandonedArm", get_abandoned_arm),
    ("Arm", get_arm),
    ("AugmentedBraninMetric", get_augmented_branin_metric),
    ("AugmentedHartmannMetric", get_augmented_hartmann_metric),
    ("BatchTrial", get_batch_trial),
    ("BenchmarkProblem", get_branin_benchmark_problem),
    ("BoTorchModel", get_botorch_model),
    ("BoTorchModel", get_botorch_model_with_default_acquisition_class),
    ("BraninMetric", get_branin_metric),
    ("ChoiceParameter", get_choice_parameter),
    ("Experiment", get_experiment_with_batch_and_single_trial),
    ("Experiment", get_experiment_with_trial_with_ttl),
    ("Experiment", get_experiment_with_data),
    ("Experiment", get_experiment_with_map_data_type),
    ("FactorialMetric", get_factorial_metric),
    ("FixedParameter", get_fixed_parameter),
    ("Hartmann6Metric", get_hartmann_metric),
    ("GenerationStrategy", partial(get_generation_strategy, with_experiment=True)),
    ("GeneratorRun", get_generator_run),
    ("ListSurrogate", get_list_surrogate),
    ("MapData", get_map_data),
    ("Metric", get_metric),
    ("MultiObjective", get_multi_objective),
    ("MultiObjectiveOptimizationConfig", get_multi_objective_optimization_config),
    ("MultiTypeExperiment", get_multi_type_experiment),
    ("ObservationFeatures", get_observation_features),
    ("Objective", get_objective),
    ("ObjectiveThreshold", get_objective_threshold),
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


class JSONStoreTest(TestCase):
    def setUp(self):
        self.experiment = get_experiment_with_batch_and_single_trial()

    def testJSONEncodeFailure(self):
        self.assertRaises(JSONEncodeError, object_to_json, RuntimeError("foobar"))

    def testJSONDecodeFailure(self):
        self.assertRaises(JSONDecodeError, object_from_json, RuntimeError("foobar"))
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
                with self.assertRaises(RuntimeError):
                    converted_object.evaluation_function(parameterization={})

                original_object.evaluation_function = None
                converted_object.evaluation_function = None

            self.assertEqual(
                original_object,
                converted_object,
                msg=f"Error encoding/decoding {class_}.",
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
        # it has generated some generator runs. Since we now need to `gen`,
        # we remove the fake callable kwarg we added, since model does not
        # expect it.
        generation_strategy = get_generation_strategy(with_callable_model_kwarg=False)
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

    def testEncodeDecodeNumpy(self):
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

    def testEncodeUnknownClassToDict(self):
        # Cannot encode `UnknownClass` type because it is not registered in the
        # CLASS_ENCODER_REGISTRY.
        class UnknownClass:
            def __init__(self):
                pass

        with self.assertRaisesRegex(
            ValueError, "is a class. Add it to the CLASS_ENCODER_REGISTRY"
        ):
            object_to_json(UnknownClass)
        # `UnknownClass` type is registered in the CLASS_ENCODER_REGISTRY and uses the
        # `botorch_modular_to_dict` encoder, but `UnknownClass` is not registered in
        # the `botorch_modular_registry.py` file.
        CLASS_ENCODER_REGISTRY[UnknownClass] = botorch_modular_to_dict
        with self.assertRaisesRegex(
            ValueError,
            "does not have a corresponding parent class in CLASS_TO_REGISTRY",
        ):
            object_to_json(UnknownClass)

    def testDecodeUnknownClassFromJson(self):
        with self.assertRaisesRegex(
            ValueError,
            "does not have a corresponding entry in CLASS_TO_REVERSE_REGISTRY",
        ):
            class_from_json({"index": 0, "class": "unknown_path"})
