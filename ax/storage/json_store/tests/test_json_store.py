#!/usr/bin/env python3

import os
import tempfile

from ax.core.metric import Metric
from ax.core.runner import Runner
from ax.exceptions.storage import JSONDecodeError, JSONEncodeError
from ax.storage.json_store.decoder import object_from_json
from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment
from ax.storage.utils import EncodeDecodeFieldsMap, remove_prefix
from ax.tests.fake import (
    get_arm,
    get_batch_trial,
    get_branin_metric,
    get_choice_parameter,
    get_experiment_with_batch_and_single_trial,
    get_experiment_with_data,
    get_factorial_metric,
    get_fixed_parameter,
    get_generator_run,
    get_hartmann_metric,
    get_metric,
    get_objective,
    get_optimization_config,
    get_order_constraint,
    get_outcome_constraint,
    get_parameter_constraint,
    get_range_parameter,
    get_search_space,
    get_simple_experiment,
    get_sum_constraint1,
    get_sum_constraint2,
    get_synthetic_runner,
    get_trial,
)
from ax.utils.common.testutils import TestCase


TEST_CASES = [
    ("BatchTrial", get_batch_trial),
    ("BraninMetric", get_branin_metric),
    ("ChoiceParameter", get_choice_parameter),
    ("Arm", get_arm),
    ("Experiment", get_experiment_with_batch_and_single_trial),
    ("Experiment", get_experiment_with_data),
    ("FactorialMetric", get_factorial_metric),
    ("FixedParameter", get_fixed_parameter),
    ("Hartmann6Metric", get_hartmann_metric),
    ("GeneratorRun", get_generator_run),
    ("Metric", get_metric),
    ("Objective", get_objective),
    ("OptimizationConfig", get_optimization_config),
    ("OrderConstraint", get_order_constraint),
    ("OutcomeConstraint", get_outcome_constraint),
    ("ParameterConstraint", get_parameter_constraint),
    ("RangeParameter", get_range_parameter),
    ("SearchSpace", get_search_space),
    ("SimpleExperiment", get_simple_experiment),
    ("SumConstraint", get_sum_constraint1),
    ("SumConstraint", get_sum_constraint2),
    ("SyntheticRunner", get_synthetic_runner),
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
    "BatchTrial": EncodeDecodeFieldsMap(python_only=["experiment"]),
    "GeneratorRun": EncodeDecodeFieldsMap(
        encoded_only=["arms", "weights"], python_only=["arm_weight_table"]
    ),
    "OrderConstraint": EncodeDecodeFieldsMap(python_only=["bound"]),
    "SimpleExperiment": EncodeDecodeFieldsMap(python_only=["evaluation_function"]),
    "SumConstraint": EncodeDecodeFieldsMap(python_only=["constraint_dict"]),
    "Trial": EncodeDecodeFieldsMap(python_only=["experiment"]),
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

    def testValidateFilename(self):
        bad_filename = "test"
        self.assertRaises(ValueError, save_experiment, self.experiment, bad_filename)

    def testEncodeDecode(self):
        for class_, fake_func in TEST_CASES:
            # Can't load trials from JSON, because a batch needs an experiment
            # in order to be initialized
            if class_ == "BatchTrial" or class_ == "Trial":
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

            self.assertEqual(
                object_keys,
                json_keys,
                msg=f"Mismatch between Python and JSON representation in {class_}.",
            )
