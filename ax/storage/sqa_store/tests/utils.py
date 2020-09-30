#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.utils import EncodeDecodeFieldsMap
from ax.utils.testing.core_stubs import (
    get_abandoned_arm,
    get_arm,
    get_batch_trial,
    get_branin_metric,
    get_branin_objective,
    get_branin_outcome_constraint,
    get_choice_parameter,
    get_experiment_with_batch_and_single_trial,
    get_experiment_with_batch_trial,
    get_experiment_with_data,
    get_experiment_with_multi_objective,
    get_experiment_with_scalarized_objective,
    get_factorial_metric,
    get_fixed_parameter,
    get_generator_run,
    get_generator_run2,
    get_hartmann_metric,
    get_metric,
    get_objective,
    get_order_constraint,
    get_outcome_constraint,
    get_parameter_constraint,
    get_range_parameter,
    get_scalarized_objective,
    get_simple_experiment_with_batch_trial,
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
    ("Arm", get_arm, Encoder.arm_to_sqa, Decoder.arm_from_sqa),
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
        "Experiment",
        get_experiment_with_data,
        Encoder.experiment_to_sqa,
        Decoder.experiment_from_sqa,
    ),
    (
        "Experiment",
        get_experiment_with_multi_objective,
        Encoder.experiment_to_sqa,
        Decoder.experiment_from_sqa,
    ),
    (
        "Experiment",
        get_experiment_with_scalarized_objective,
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
        "FactorialMetric",
        get_factorial_metric,
        Encoder.metric_to_sqa,
        Decoder.metric_from_sqa,
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
        "HartmannMetric",
        get_hartmann_metric,
        Encoder.metric_to_sqa,
        Decoder.metric_from_sqa,
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
        "ScalarizedObjective",
        get_scalarized_objective,
        Encoder.objective_to_sqa,
        Decoder.metric_from_sqa,
    ),
    (
        "ScalarizedObjective",
        get_scalarized_objective,
        Encoder.objective_to_sqa,
        Decoder.metric_from_sqa,
    ),
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
        "SimpleExperiment",
        get_simple_experiment_with_batch_trial,
        Encoder.experiment_to_sqa,
        Decoder.experiment_from_sqa,
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
    "Metric": EncodeDecodeFieldsMap(
        encoded_only={
            "scalarized_objective_children_metrics",
            "scalarized_objective_weight",
        }
    ),
    "AbandonedArm": EncodeDecodeFieldsMap(
        python_to_encoded={"reason": "abandoned_reason", "time": "time_abandoned"}
    ),
    "Arm": EncodeDecodeFieldsMap(
        encoded_only=["weight"], python_to_encoded={"parameters": "parameters"}
    ),
    "BatchTrial": EncodeDecodeFieldsMap(
        python_to_encoded={
            "generator_run_structs": "generator_runs",
            "abandoned_arms_metadata": "abandoned_arms",
            "num_arms_created": "num_arms_created",
            # Dunder fields in Py get prepended with class name (BaseTrial in this
            # case because that's the one that carries the `__status` attribute,
            # which BatchTrial inherits)
            "BaseTrial__status": "status",
        },
        python_only=["experiment", "status_quo", "status_quo_weight_override"],
        encoded_only=["is_batch", "status_quo_name", "deployed_name", "properties"],
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
            "trial_type",
            "canonical_name",
            "scalarized_objective_children_metrics",
            "scalarized_objective_weight",
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
            "trial_type",
            "canonical_name",
            "scalarized_objective_children_metrics",
            "scalarized_objective_weight",
        ],
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
            "properties",
            "status_quo_name",
            "status_quo_parameters",
            "default_trial_type",
        ],
        python_only=[
            "arms_by_signature",
            "arms_by_name",
            "search_space",
            "runner",
            "optimization_config",
            "status_quo",
            "trial_indices_by_status",
            "trials_have_ttl",
        ],
        python_to_encoded={
            "data_by_trial": "data",
            "tracking_metrics": "metrics",
            "runner": "runners",
        },
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
            "best_arm_predictions",
            "model_key",
            "model_kwargs",
            "bridge_kwargs",
        ],
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
            "trial_type",
            "canonical_name",
            "scalarized_objective_children_metrics",
            "scalarized_objective_weight",
        ],
    ),
    "MultiObjective": EncodeDecodeFieldsMap(
        encoded_only=[
            "metric_type",
            "intent",
            "name",
            "lower_is_better",
            "properties",
            "op",
            "relative",
            "bound",
            "trial_type",
            "canonical_name",
            "scalarized_objective_weight",
        ],
        python_only=["weights"],
        python_to_encoded={"metrics": "scalarized_objective_children_metrics"},
    ),
    "ScalarizedObjective": EncodeDecodeFieldsMap(
        encoded_only=[
            "metric_type",
            "intent",
            "name",
            "lower_is_better",
            "properties",
            "op",
            "relative",
            "bound",
            "trial_type",
            "canonical_name",
        ],
        python_to_encoded={
            "metrics": "scalarized_objective_children_metrics",
            "weights": "scalarized_objective_weight",
        },
    ),
    "OrderConstraint": EncodeDecodeFieldsMap(
        encoded_only=["constraint_dict", "type"],
        python_only=["lower_parameter", "upper_parameter"],
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
            "trial_type",
            "canonical_name",
            "scalarized_objective_children_metrics",
            "scalarized_objective_weight",
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
    "SimpleExperiment": EncodeDecodeFieldsMap(
        encoded_only=[
            "parameter_constraints",
            "parameters",
            "properties",
            "status_quo_name",
            "status_quo_parameters",
            "default_trial_type",
            "runners",
        ],
        python_only=[
            "arms_by_signature",
            "arms_by_name",
            "search_space",
            "runner",
            "optimization_config",
            "status_quo",
            "evaluation_function",
            "trial_indices_by_status",
            "trials_have_ttl",
        ],
        python_to_encoded={"data_by_trial": "data", "tracking_metrics": "metrics"},
    ),
    "SumConstraint": EncodeDecodeFieldsMap(
        python_only=["is_upper_bound", "parameters", "parameter_names"],
        encoded_only=["type"],
    ),
    "Trial": EncodeDecodeFieldsMap(
        python_to_encoded={
            "generator_run": "generator_runs",
            "num_arms_created": "num_arms_created",
            # Dunder fields in Py get prepended with class name (BaseTrial in this
            # case because that's the one that carries the `__status` attribute,
            # which Trial inherits)
            "BaseTrial__status": "status",
        },
        python_only=["experiment"],
        encoded_only=[
            "is_batch",
            "abandoned_arms",
            "status_quo_name",
            "deployed_name",
            "optimize_for_power",
            "properties",
        ],
    ),
}
