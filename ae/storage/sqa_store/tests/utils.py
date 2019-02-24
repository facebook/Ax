#!/usr/bin/env python3

from ae.lazarus.ae.storage.sqa_store.base_decoder import Decoder
from ae.lazarus.ae.storage.sqa_store.base_encoder import Encoder
from ae.lazarus.ae.storage.utils import EncodeDecodeFieldsMap
from ae.lazarus.ae.tests.fake import (
    get_abandoned_arm,
    get_arm,
    get_batch_trial,
    get_branin_metric,
    get_branin_objective,
    get_branin_outcome_constraint,
    get_choice_parameter,
    get_experiment_with_batch_and_single_trial,
    get_experiment_with_batch_trial,
    get_fixed_parameter,
    get_generator_run,
    get_generator_run2,
    get_metric,
    get_objective,
    get_order_constraint,
    get_outcome_constraint,
    get_parameter_constraint,
    get_range_parameter,
    get_sum_constraint1,
    get_sum_constraint2,
    get_synthetic_runner,
    get_trial,
)


TEST_CASES = [
    (
        "AbandonedArm",
        get_abandoned_arm,
        Encoder.abandoned_arm_to_sqa,
        Decoder.abandoned_arm_from_sqa,
    ),
    ("BatchTrial", get_batch_trial, Encoder.trial_to_sqa, Decoder.trial_from_sqa),
    ("BraninMetric", get_branin_metric, Encoder.metric_to_sqa, Decoder.metric_from_sqa),
    (
        "BraninObjective",
        get_branin_objective,
        Encoder.objective_to_sqa,
        Decoder.metric_from_sqa,
    ),
    (
        "BraninOutcomeConstraint",
        get_branin_outcome_constraint,
        Encoder.outcome_constraint_to_sqa,
        Decoder.metric_from_sqa,
    ),
    (
        "ChoiceParameter",
        get_choice_parameter,
        Encoder.parameter_to_sqa,
        Decoder.parameter_from_sqa,
    ),
    ("Arm", get_arm, Encoder.arm_to_sqa, Decoder.arm_from_sqa),
    (
        "Experiment",
        get_experiment_with_batch_trial,
        Encoder.experiment_to_sqa,
        Decoder.experiment_from_sqa,
    ),
    (
        "Experiment",
        get_experiment_with_batch_and_single_trial,
        Encoder.experiment_to_sqa,
        Decoder.experiment_from_sqa,
    ),
    (
        "FixedParameter",
        get_fixed_parameter,
        Encoder.parameter_to_sqa,
        Decoder.parameter_from_sqa,
    ),
    (
        "GeneratorRun",
        get_generator_run,
        Encoder.generator_run_to_sqa,
        Decoder.generator_run_from_sqa,
    ),
    (
        "GeneratorRun",
        get_generator_run2,
        Encoder.generator_run_to_sqa,
        Decoder.generator_run_from_sqa,
    ),
    (
        "OrderConstraint",
        get_order_constraint,
        Encoder.parameter_constraint_to_sqa,
        Decoder.parameter_constraint_from_sqa,
    ),
    (
        "ParameterConstraint",
        get_parameter_constraint,
        Encoder.parameter_constraint_to_sqa,
        Decoder.parameter_constraint_from_sqa,
    ),
    ("Metric", get_metric, Encoder.metric_to_sqa, Decoder.metric_from_sqa),
    ("Objective", get_objective, Encoder.objective_to_sqa, Decoder.metric_from_sqa),
    (
        "OutcomeConstraint",
        get_outcome_constraint,
        Encoder.outcome_constraint_to_sqa,
        Decoder.metric_from_sqa,
    ),
    (
        "RangeParameter",
        get_range_parameter,
        Encoder.parameter_to_sqa,
        Decoder.parameter_from_sqa,
    ),
    (
        "SyntheticRunner",
        get_synthetic_runner,
        Encoder.runner_to_sqa,
        Decoder.runner_from_sqa,
    ),
    (
        "SumConstraint",
        get_sum_constraint1,
        Encoder.parameter_constraint_to_sqa,
        Decoder.parameter_constraint_from_sqa,
    ),
    (
        "SumConstraint",
        get_sum_constraint2,
        Encoder.parameter_constraint_to_sqa,
        Decoder.parameter_constraint_from_sqa,
    ),
    ("Trial", get_trial, Encoder.trial_to_sqa, Decoder.trial_from_sqa),
]


# This map records discrepancies between Python and SQA representations,
# so that we can validate that the SQA representation is complete
ENCODE_DECODE_FIELD_MAPS = {
    "AbandonedArm": EncodeDecodeFieldsMap(
        python_to_encoded={"reason": "abandoned_reason", "time": "time_abandoned"}
    ),
    "BatchTrial": EncodeDecodeFieldsMap(
        python_to_encoded={
            "generator_run_structs": "generator_runs",
            "abandoned_arms_metadata": "abandoned_arms",
            "num_arms_created": "num_arms_created",
        },
        python_only=["experiment", "status_quo", "status_quo_weight"],
        encoded_only=["is_batch", "status_quo_name"],
    ),
    "BraninMetric": EncodeDecodeFieldsMap(
        python_only=["param_names", "noise_sd"],
        encoded_only=[
            "intent",
            "properties",
            "metric_type",
            "minimize",
            "op",
            "bound",
            "relative",
        ],
    ),
    "BraninObjective": EncodeDecodeFieldsMap(
        python_only=["metric"],
        encoded_only=[
            "metric_type",
            "intent",
            "name",
            "lower_is_better",
            "properties",
            "op",
            "bound",
            "relative",
        ],
    ),
    "BraninOutcomeConstraint": EncodeDecodeFieldsMap(
        python_only=["metric"],
        encoded_only=[
            "metric_type",
            "intent",
            "name",
            "lower_is_better",
            "properties",
            "minimize",
        ],
    ),
    "Arm": EncodeDecodeFieldsMap(
        encoded_only=["weight"], python_to_encoded={"params": "parameters"}
    ),
    "ChoiceParameter": EncodeDecodeFieldsMap(
        encoded_only=[
            "domain_type",
            "log_scale",
            "upper",
            "digits",
            "fixed_value",
            "lower",
        ],
        python_to_encoded={"values": "choice_values"},
    ),
    "Experiment": EncodeDecodeFieldsMap(
        encoded_only=[
            "parameter_constraints",
            "parameters",
            "status_quo_name",
            "status_quo_parameters",
        ],
        python_only=[
            "search_space",
            "data_by_trial",
            "runner",
            "optimization_config",
            "status_quo",
        ],
    ),
    "FixedParameter": EncodeDecodeFieldsMap(
        encoded_only=[
            "domain_type",
            "log_scale",
            "upper",
            "digits",
            "lower",
            "choice_values",
            "is_task",
            "is_ordered",
        ],
        python_to_encoded={"value": "fixed_value"},
    ),
    "GeneratorRun": EncodeDecodeFieldsMap(
        encoded_only=[
            "arms",
            "metrics",
            "parameters",
            "parameter_constraints",
            "weight",
            "best_arm_name",
            "best_arm_parameters",
            "best_arm_predictions",
        ],
        python_only=[
            "arm_weight_table",
            "optimization_config",
            "search_space",
            "best_arm_predictions",  # TODO[drfreund, lilidworkin] T38936022
        ],
    ),
    "Metric": EncodeDecodeFieldsMap(
        encoded_only=[
            "intent",
            "properties",
            "metric_type",
            "op",
            "relative",
            "bound",
            "minimize",
        ]
    ),
    "Objective": EncodeDecodeFieldsMap(
        python_only=["metric"],
        encoded_only=[
            "metric_type",
            "intent",
            "name",
            "lower_is_better",
            "properties",
            "op",
            "relative",
            "bound",
        ],
    ),
    "OrderConstraint": EncodeDecodeFieldsMap(
        python_only=["lower_name", "upper_name"],
        encoded_only=["constraint_dict", "type"],
    ),
    "OutcomeConstraint": EncodeDecodeFieldsMap(
        python_only=["metric"],
        encoded_only=[
            "metric_type",
            "intent",
            "name",
            "lower_is_better",
            "properties",
            "minimize",
        ],
    ),
    "ParameterConstraint": EncodeDecodeFieldsMap(encoded_only=["type"]),
    "RangeParameter": EncodeDecodeFieldsMap(
        encoded_only=[
            "domain_type",
            "choice_values",
            "fixed_value",
            "is_task",
            "is_ordered",
        ]
    ),
    "SumConstraint": EncodeDecodeFieldsMap(
        python_only=["parameter_names", "is_upper_bound"], encoded_only=["type"]
    ),
    "SyntheticRunner": EncodeDecodeFieldsMap(
        encoded_only=["runner_type", "properties"], python_only=["dummy_metadata"]
    ),
    "Trial": EncodeDecodeFieldsMap(
        python_to_encoded={
            "generator_run": "generator_runs",
            "num_arms_created": "num_arms_created",
        },
        python_only=["experiment"],
        encoded_only=["is_batch", "abandoned_arms", "status_quo_name"],
    ),
}
