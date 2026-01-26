#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from enum import Enum
from logging import Logger
from typing import Any, cast, TypeVar
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
from ax.adapter.registry import Generators
from ax.analysis.markdown.markdown_analysis import MarkdownAnalysisCard
from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
from ax.api.utils.generation_strategy_dispatch import (
    choose_generation_strategy,
    GenerationStrategyDispatchStruct,
)
from ax.core.analysis_card import AnalysisCard, AnalysisCardGroup
from ax.core.arm import Arm
from ax.core.auxiliary import (
    AuxiliaryExperiment,
    AuxiliaryExperimentPurpose,
    TransferLearningMetadata,
)
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
    PreferenceOptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.runner import Runner
from ax.core.trial import Trial
from ax.core.trial_status import TrialStatus
from ax.core.types import ComparisonOp
from ax.exceptions.core import (
    MisconfiguredExperiment,
    ObjectNotFoundError,
    TrialMutationError,
    UnsupportedError,
)
from ax.exceptions.storage import JSONDecodeError, SQADecodeError, SQAEncodeError
from ax.generation_strategy.dispatch_utils import choose_generation_strategy_legacy
from ax.generation_strategy.transition_criterion import MaxGenerationParallelism
from ax.generators.torch.botorch_modular.surrogate import Surrogate, SurrogateSpec
from ax.metrics.branin import BraninMetric
from ax.runners.synthetic import SyntheticRunner
from ax.storage.json_store.decoder import (
    generation_node_from_json,
    transition_criterion_from_json,
)
from ax.storage.json_store.registry import (
    CORE_CLASS_DECODER_REGISTRY,
    CORE_DECODER_REGISTRY,
)
from ax.storage.metric_registry import CORE_METRIC_REGISTRY, register_metrics
from ax.storage.registry_bundle import RegistryBundle
from ax.storage.runner_registry import CORE_RUNNER_REGISTRY, register_runner
from ax.storage.sqa_store.db import (
    create_all_tables,
    create_test_engine,
    get_engine,
    get_session,
    init_engine_and_session_factory,
    init_test_engine_and_session_factory,
    session_context,
    session_scope,
)
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.delete import delete_experiment, delete_generation_strategy
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.load import (
    _get_experiment_immutable_opt_config_and_search_space,
    _get_experiment_sqa_immutable_opt_config_and_search_space,
    _get_generation_strategy_sqa_immutable_opt_config_and_search_space,
    _query_historical_experiments_given_parameters,
    identify_transferable_experiments,
    load_analysis_cards_by_experiment_name,
    load_candidate_source_auxiliary_experiments,
    load_experiment,
    load_generation_strategy_by_experiment_name,
    load_generation_strategy_by_id,
)
from ax.storage.sqa_store.reduced_state import GR_LARGE_MODEL_ATTRS, SQA_COL_TO_GR_ATTR
from ax.storage.sqa_store.save import (
    save_analysis_card,
    save_experiment,
    save_generation_strategy,
    save_or_update_trial,
    save_or_update_trials,
    update_generation_strategy,
    update_properties_on_experiment,
    update_properties_on_trial,
    update_runner_on_experiment,
    update_trial_status,
)
from ax.storage.sqa_store.sqa_classes import (
    SQAAbandonedArm,
    SQAArm,
    SQAExperiment,
    SQAGeneratorRun,
    SQAMetric,
    SQAParameter,
    SQAParameterConstraint,
    SQARunner,
    SQATrial,
)
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.storage.sqa_store.tests.utils import TEST_CASES
from ax.storage.utils import DomainType, MetricIntent, ParameterConstraintType
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.serialization import serialize_init_args
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    CustomTestMetric,
    CustomTestRunner,
    get_arm,
    get_branin_data,
    get_branin_experiment,
    get_branin_experiment_with_timestamp_map_metric,
    get_branin_metric,
    get_choice_parameter,
    get_data,
    get_experiment,
    get_experiment_with_batch_trial,
    get_experiment_with_custom_runner_and_metric,
    get_experiment_with_map_data_type,
    get_experiment_with_multi_objective,
    get_experiment_with_scalarized_objective_and_outcome_constraint,
    get_fixed_parameter,
    get_generator_run,
    get_model_predictions_per_arm,
    get_multi_objective_optimization_config,
    get_multi_type_experiment,
    get_objective,
    get_objective_threshold,
    get_optimization_config,
    get_outcome_constraint,
    get_range_parameter,
    get_range_parameter2,
    get_scalarized_outcome_constraint,
    get_search_space,
    get_sum_constraint2,
    get_synthetic_runner,
)
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import (
    get_generation_strategy,
    sobol_gpei_generation_node_gs,
)
from plotly import graph_objects as go, io as pio
from pyre_extensions import assert_is_instance, none_throws

logger: Logger = get_logger(__name__)

GET_GS_SQA_IMM_FUNC = _get_generation_strategy_sqa_immutable_opt_config_and_search_space
T = TypeVar("T")


class MockExperimentTypeEnum(Enum):
    """Mock enum for testing experiment types (not a pytest test class)."""

    TEST = 0


class SQAStoreTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        init_test_engine_and_session_factory(force_init=True)
        self.config = SQAConfig()
        self.encoder = Encoder(config=self.config)
        self.decoder = Decoder(config=self.config)
        self.experiment = get_experiment_with_batch_trial()
        self.dummy_parameters = [
            get_range_parameter(),  # w
            get_range_parameter2(),  # x
        ]

    def test_creation_of_test_db(self) -> None:
        init_test_engine_and_session_factory(tier_or_path=":memory:", force_init=True)
        engine = get_engine()
        self.assertIsNotNone(engine)

    def test_db_connection_without_force_init(self) -> None:
        init_test_engine_and_session_factory(tier_or_path=":memory:")

    def test_connection_to_db_with_url(self) -> None:
        init_engine_and_session_factory(url="sqlite://", force_init=True)

    def MockDBAPI(self) -> MagicMock:
        connection = Mock()

        # pyre-fixme[53]: Captured variable `connection` is not annotated.
        def connect(*args: Any, **kwargs: Any) -> Mock:
            return connection

        return MagicMock(connect=Mock(side_effect=connect))

    def test_connection_to_db_with_creator(self) -> None:
        mocked_dbapi = self.MockDBAPI()
        init_engine_and_session_factory(
            creator=lambda: mocked_dbapi.connect(),
            force_init=True,
            module=mocked_dbapi,
            echo=True,
            pool_size=2,
            _initialize=False,
        )
        with session_scope() as session:
            engine = session.bind
            engine.connect()
            self.assertEqual(mocked_dbapi.connect.call_count, 1)
            self.assertTrue(engine.echo)
            self.assertEqual(engine.pool.size(), 2)

    def test_connection_to_db_with_session_context(self) -> None:
        mocked_dbapi: MagicMock = self.MockDBAPI()

        def creator() -> Mock:
            return mocked_dbapi.connect()

        init_engine_and_session_factory(
            creator=creator,
            force_init=True,
            module=mocked_dbapi,
        )
        session_before = get_session()

        with session_context(
            creator=creator,
            module=mocked_dbapi,
        ):
            in_context_session = get_session()
            # Inside context we should have a new session
            self.assertNotEqual(session_before, in_context_session)

        # After context we should have the same session as before
        session_after = get_session()
        self.assertEqual(session_after, session_before)

    def test_generator_run_type_validation(self) -> None:
        experiment = get_experiment_with_batch_trial()
        generator_run = experiment.trials[0].generator_runs[0]
        generator_run._generator_run_type = "foobar"
        with self.assertRaises(SQAEncodeError):
            self.encoder.generator_run_to_sqa(generator_run)

        generator_run._generator_run_type = "STATUS_QUO"
        generator_run_sqa = self.encoder.generator_run_to_sqa(generator_run)
        generator_run_sqa.generator_run_type = 2
        with self.assertRaises(SQADecodeError):
            self.decoder.generator_run_from_sqa(generator_run_sqa, False, False)

        generator_run_sqa.generator_run_type = 0
        self.decoder.generator_run_from_sqa(generator_run_sqa, False, False)

    def test_generator_run_best_arm(self) -> None:
        experiment = self.experiment

        generator_run = experiment.trials[0].generator_runs[0]
        generator_run._generator_run_type = "STATUS_QUO"

        arm = get_arm()
        arm_predictions = get_model_predictions_per_arm()
        arm._name = arm.signature

        generator_run._best_arm_predictions = (arm, arm_predictions[arm.signature])

        generator_run_sqa = self.encoder.generator_run_to_sqa(generator_run)

        self.assertIsNotNone(generator_run_sqa.best_arm_name)
        self.assertIsNotNone(generator_run_sqa.best_arm_predictions)

    def test_generator_run_no_best_arm(self) -> None:
        experiment = self.experiment

        generator_run = experiment.trials[0].generator_runs[0]
        generator_run._generator_run_type = "STATUS_QUO"
        generator_run._best_arm_predictions = None

        generator_run_sqa = self.encoder.generator_run_to_sqa(generator_run)

        self.assertIsNone(generator_run_sqa.best_arm_name)
        self.assertIsNone(generator_run_sqa.best_arm_predictions)

    def test_generator_run_no_best_arm_predictions(self) -> None:
        experiment = self.experiment

        generator_run = experiment.trials[0].generator_runs[0]
        generator_run._generator_run_type = "STATUS_QUO"

        arm = get_arm()
        arm._name = "best_arm"

        generator_run._best_arm_predictions = (arm, None)

        with self.assertLogs("ax", level=logging.WARNING) as logs:
            generator_run_sqa = self.encoder.generator_run_to_sqa(generator_run)

        self.assertEqual(
            [
                "WARNING:ax.storage.sqa_store.encoder:"
                "No model predictions found with best arm 'best_arm'."
                " Setting best_arm_predictions=None in storage"
            ],
            logs.output,
        )

        self.assertIsNotNone(generator_run_sqa.best_arm_name)
        self.assertIsNone(generator_run_sqa.best_arm_predictions)

    @mock_botorch_optimize
    def test_save_experiment_with_surrogate_as_model_kwarg(self) -> None:
        experiment = get_branin_experiment(
            with_batch=True, num_batch_trial=1, with_completed_batch=True
        )
        model = Generators.BOTORCH_MODULAR(
            experiment=experiment,
            data=experiment.lookup_data(),
            surrogate=Surrogate(surrogate_spec=SurrogateSpec()),
        )
        experiment.new_batch_trial(generator_run=model.gen(1))
        # ensure we can save the experiment
        save_experiment(experiment)

    def test_experiment_save_load(self) -> None:
        for exp in [
            self.experiment,
            get_experiment_with_map_data_type(),
            get_experiment_with_multi_objective(),
            get_experiment_with_scalarized_objective_and_outcome_constraint(),
        ]:
            self.assertIsNone(exp.db_id)
            save_experiment(exp)
            self.assertIsNotNone(exp.db_id)
            loaded_experiment = load_experiment(exp.name)
            self.assertEqual(loaded_experiment, exp)

    def test_saving_and_loading_experiment_with_aux_exp(self) -> None:
        aux_experiment = Experiment(
            name="test_aux_exp_in_SQAStoreTest",
            search_space=get_search_space(),
            optimization_config=get_optimization_config(),
            description="test description",
            tracking_metrics=[Metric(name="tracking")],
            is_test=True,
        )
        save_experiment(aux_experiment, config=self.config)

        experiment_w_aux_exp = Experiment(
            name="test_experiment_w_aux_exp_in_SQAStoreTest",
            search_space=get_search_space(),
            optimization_config=get_optimization_config(),
            description="test description",
            tracking_metrics=[Metric(name="tracking")],
            is_test=True,
            auxiliary_experiments_by_purpose={
                # pyre-ignore[16]: `AuxiliaryExperimentPurpose` has no attribute
                self.config.auxiliary_experiment_purpose_enum.PE_EXPERIMENT: [
                    AuxiliaryExperiment(experiment=aux_experiment)
                ]
            },
        )
        self.assertIsNone(experiment_w_aux_exp.db_id)
        save_experiment(experiment_w_aux_exp, config=self.config)
        self.assertIsNotNone(experiment_w_aux_exp.db_id)
        loaded_experiment = load_experiment(
            experiment_w_aux_exp.name, config=self.config
        )
        self.assertEqual(experiment_w_aux_exp, loaded_experiment)
        self.assertEqual(len(loaded_experiment.auxiliary_experiments_by_purpose), 1)

    def test_saving_and_loading_experiment_with_aux_exp_reduced_state(self) -> None:
        aux_exp = Experiment(
            name="test_aux_exp_in_SQAStoreTest_reduced_state",
            search_space=get_search_space(),
            optimization_config=get_optimization_config(),
            description="test description",
            tracking_metrics=[Metric(name="tracking")],
            is_test=True,
        )
        aux_exp_gs = get_generation_strategy()
        aux_exp.new_trial(aux_exp_gs.gen_single_trial(experiment=aux_exp))
        save_experiment(aux_exp, config=self.config)
        # pyre-ignore[16]: `AuxiliaryExperimentPurpose` has no attribute
        purpose = self.config.auxiliary_experiment_purpose_enum.PE_EXPERIMENT

        target_exp = Experiment(
            name="test_experiment_w_aux_exp_in_SQAStoreTest_reduced_state",
            search_space=get_search_space(),
            optimization_config=get_optimization_config(),
            description="test description",
            tracking_metrics=[Metric(name="tracking")],
            is_test=True,
            auxiliary_experiments_by_purpose={
                purpose: [AuxiliaryExperiment(experiment=aux_exp)]
            },
        )
        target_exp_gs = get_generation_strategy()
        target_exp.new_trial(target_exp_gs.gen_single_trial(experiment=target_exp))
        self.assertIsNone(target_exp.db_id)
        save_experiment(target_exp, config=self.config)
        self.assertIsNotNone(target_exp.db_id)
        loaded_target_exp = load_experiment(
            target_exp.name, config=self.config, reduced_state=True
        )
        self.assertNotEqual(target_exp, loaded_target_exp)
        self.assertIsNotNone(  # State of the original aux experiment is not reduced.
            none_throws(
                assert_is_instance(aux_exp.trials[0], Trial).generator_run
            ).gen_metadata
        )
        self.assertIsNotNone(  # State of the original target experiment is not reduced.
            none_throws(
                assert_is_instance(target_exp.trials[0], Trial).generator_run
            ).gen_metadata
        )
        self.assertIsNone(  # State of the loaded target experiment *is reduced*.
            none_throws(
                assert_is_instance(loaded_target_exp.trials[0], Trial).generator_run
            ).gen_metadata
        )
        loaded_aux_exp = loaded_target_exp.auxiliary_experiments_by_purpose[purpose][0]
        self.assertIsNone(  # State of the loaded target experiment *is reduced*.
            none_throws(
                assert_is_instance(
                    loaded_aux_exp.experiment.trials[0], Trial
                ).generator_run
            ).gen_metadata
        )
        self.assertEqual(len(loaded_target_exp.auxiliary_experiments_by_purpose), 1)

    def test_saving_with_aux_exp_not_in_db(self) -> None:
        aux_experiment = Experiment(
            name="aux_experiment_not_in_db", search_space=get_search_space()
        )
        experiment_w_aux_exp = Experiment(
            name="test_experiment_w_aux_exp",
            search_space=get_search_space(),
            is_test=True,
            auxiliary_experiments_by_purpose={
                # pyre-ignore[16]: `AuxiliaryExperimentPurpose` has no attribute
                self.config.auxiliary_experiment_purpose_enum.PE_EXPERIMENT: [
                    AuxiliaryExperiment(experiment=aux_experiment)
                ]
            },
        )
        with self.assertRaisesRegex(SQAEncodeError, "that does not exist in"):
            save_experiment(experiment_w_aux_exp, config=self.config)

    def test_saving_and_loading_experiment_with_cross_referencing_aux_exp(
        self,
    ) -> None:
        exp1_name = "test_aux_exp_in_SQAStoreTest1"
        exp2_name = "test_aux_exp_in_SQAStoreTest2"
        # pyre-ignore[16]: `AuxiliaryExperimentPurpose` has no attribute
        exp_purpose = self.config.auxiliary_experiment_purpose_enum.PE_EXPERIMENT

        exp1 = Experiment(
            name=exp1_name,
            search_space=get_search_space(),
            optimization_config=get_optimization_config(),
            description="test description",
            tracking_metrics=[Metric(name="tracking")],
            is_test=True,
        )
        exp2 = Experiment(
            name=exp2_name,
            search_space=get_search_space(),
            optimization_config=get_optimization_config(),
            description="test description",
            tracking_metrics=[Metric(name="tracking")],
            is_test=True,
        )
        # Save both experiments first
        save_experiment(exp1, config=self.config)
        save_experiment(exp2, config=self.config)

        exp1.auxiliary_experiments_by_purpose = {
            exp_purpose: [AuxiliaryExperiment(experiment=exp2)]
        }
        exp2.auxiliary_experiments_by_purpose = {
            exp_purpose: [AuxiliaryExperiment(experiment=exp1)]
        }

        # Saving both experiments with cross referencing experiments should be fine
        save_experiment(exp1, config=self.config)
        save_experiment(exp2, config=self.config)

        reloaded_exp1 = load_experiment(exp1_name, config=self.config)

        # The reloaded experiment should still have the aux experiment
        rel_exp1_aux_exps = reloaded_exp1.auxiliary_experiments_by_purpose[exp_purpose]
        self.assertEqual(len(rel_exp1_aux_exps), 1)

        exp1_aux_exp = exp1.auxiliary_experiments_by_purpose[exp_purpose][0].experiment
        rel_exp1_aux_exp = rel_exp1_aux_exps[0].experiment

        # The directly reloaded experiment won't be equal since the original exp1 has
        # recursive aux experiments
        self.assertNotEqual(exp1_aux_exp, rel_exp1_aux_exp)

        # Manually set exp1's aux's aux experiment to be empty.
        # Then they will be equal as we don't load aux experiment recursively
        exp1_aux_exp.auxiliary_experiments_by_purpose = {}
        self.assertEqual(exp1_aux_exp, rel_exp1_aux_exp)

    def test_saving_an_experiment_with_type_requires_an_enum(self) -> None:
        self.experiment.experiment_type = "TEST"
        with self.assertRaises(SQAEncodeError):
            save_experiment(self.experiment)

    def test_saving_an_experiment_with_type_works_with_an_enum(self) -> None:
        self.experiment.experiment_type = "TEST"
        save_experiment(
            self.experiment,
            config=SQAConfig(experiment_type_enum=MockExperimentTypeEnum),
        )
        self.assertIsNotNone(self.experiment.db_id)

    def test_saving_an_experiment_with_type_errors_with_missing_enum_value(
        self,
    ) -> None:
        self.experiment.experiment_type = "MISSING_TEST"
        with self.assertRaises(SQAEncodeError):
            save_experiment(
                self.experiment,
                config=SQAConfig(experiment_type_enum=MockExperimentTypeEnum),
            )

    def test_load_experiment_trials_in_batches(self) -> None:
        for _ in range(4):
            self.experiment.new_trial()
        self.assertEqual(len(self.experiment.trials), 5)
        save_experiment(self.experiment)
        loaded_experiment = load_experiment(
            self.experiment.name, load_trials_in_batches_of_size=2
        )
        self.assertEqual(self.experiment, loaded_experiment)

    # The goal of this test is to confirm that when skip_runners_and_metrics
    # is set to True, we do not attempt to load runners, and load
    # metrics minimally (converted to a base metric). This enables us to
    # load experiments with custom runners and metrics without a decoder.
    def test_load_experiment_skip_metrics_and_runners(self) -> None:
        # Create a test experiment with a custom metric and runner.
        experiment = get_experiment_with_custom_runner_and_metric(
            constrain_search_space=False
        )

        # Note that the experiment is created outside of the test code.
        # Confirm that it uses the custom runner and metric
        self.assertEqual(experiment.runner.__class__, CustomTestRunner)
        self.assertTrue("custom_test_metric" in experiment.metrics)
        self.assertEqual(
            experiment.metrics["custom_test_metric"].__class__, CustomTestMetric
        )

        # Create an SQAConfig with the updated registries with the
        # custom runner and metric.
        (
            metric_registry,
            partial_encoder_registry,
            partial_decoder_registry,
        ) = register_metrics(metric_clss={CustomTestMetric: None, Metric: 0})

        runner_registry, encoder_registry, decoder_registry = register_runner(
            runner_cls=CustomTestRunner,
            encoder_registry=partial_encoder_registry,
            decoder_registry=partial_decoder_registry,
        )

        sqa_config = SQAConfig(
            json_encoder_registry=encoder_registry,
            json_decoder_registry=decoder_registry,
            metric_registry=metric_registry,
            runner_registry=runner_registry,
        )

        for immutable in [True, False]:
            for composite_type in ["none", "multi_objective", "scalarized"]:
                custom_metric_names = ["custom_test_metric"]

                # Create appropriate experiment based on composite type
                if composite_type == "multi_objective":
                    experiment = get_experiment_with_custom_runner_and_metric(
                        constrain_search_space=False,
                        immutable=immutable,
                        multi_objective=True,
                        num_trials=1,
                    )
                    custom_metric_names.extend(["m1", "m3"])
                elif composite_type == "scalarized":
                    experiment = get_experiment_with_custom_runner_and_metric(
                        constrain_search_space=False,
                        immutable=immutable,
                        scalarized_objective=True,
                        has_outcome_constraint=True,
                        num_trials=1,
                    )
                    custom_metric_names.extend(["m1", "m3", "oc_m3", "oc_m4"])
                else:  # "none" - regular single objective
                    experiment = get_experiment_with_custom_runner_and_metric(
                        constrain_search_space=False,
                        immutable=immutable,
                        multi_objective=False,
                        num_trials=1,
                    )

                # Verify custom metrics are being used
                for metric_name in custom_metric_names:
                    self.assertEqual(
                        experiment.metrics[metric_name].__class__, CustomTestMetric
                    )

                # Save the experiment to db using the updated registries.
                save_experiment(experiment, config=sqa_config)

                # At this point try to load the experiment back without specifying
                # updated registries. Confirm that this attempt fails.
                with self.assertRaises(SQADecodeError):
                    loaded_experiment = load_experiment(experiment.name)

                # Now load it with the skip_runners_and_metrics argument set.
                # The experiment should load (i.e. no exceptions raised)
                loaded_experiment = load_experiment(
                    experiment.name, skip_runners_and_metrics=True
                )

                # Validate that:
                #   - the runner is not loaded
                #   - the metric is loaded as a base Metric class, not CustomTestMetric
                self.assertIs(loaded_experiment.runner, None)

                for metric_name in custom_metric_names:
                    if metric_name in loaded_experiment.metrics:
                        self.assertEqual(
                            loaded_experiment.metrics[metric_name].__class__,
                            Metric,
                        )
                self.assertEqual(len(loaded_experiment.trials), 1)
                trial = loaded_experiment.trials[0]
                self.assertIs(trial.runner, None)
                delete_experiment(exp_name=experiment.name)

                # Check generator runs
                gr = trial.generator_runs[0]
                if composite_type == "multi_objective" and not immutable:
                    objectives = assert_is_instance(
                        none_throws(gr.optimization_config).objective, MultiObjective
                    ).objectives
                    for i, objective in enumerate(objectives):
                        metric = objective.metric
                        self.assertEqual(metric.name, f"m{1 + 2 * i}")
                        self.assertEqual(metric.signature, f"m{1 + 2 * i}")
                        self.assertEqual(metric.__class__, Metric)
                elif composite_type == "scalarized" and not immutable:
                    # Check scalarized objective children
                    scalarized_objective = none_throws(gr.optimization_config).objective
                    if isinstance(scalarized_objective, ScalarizedObjective):
                        for metric in scalarized_objective.metrics:
                            self.assertEqual(metric.__class__, Metric)

                    # Check scalarized outcome constraint children
                    for constraint in none_throws(
                        gr.optimization_config
                    ).outcome_constraints:
                        if isinstance(constraint, ScalarizedOutcomeConstraint):
                            for metric in constraint.metrics:
                                self.assertEqual(metric.__class__, Metric)

    @patch(
        f"{Decoder.__module__}.Decoder.generator_run_from_sqa",
        side_effect=Decoder(SQAConfig()).generator_run_from_sqa,
    )
    @patch(
        f"{Decoder.__module__}.Decoder.trial_from_sqa",
        side_effect=Decoder(SQAConfig()).trial_from_sqa,
    )
    # pyre-fixme[56]: Pyre was not able to infer the type of argument `ax.storage.sqa...
    @patch(
        f"{Decoder.__module__}.Decoder.experiment_from_sqa",
        side_effect=Decoder(SQAConfig()).experiment_from_sqa,
    )
    def test_experiment_save_and_load_reduced_state(
        self, _mock_exp_from_sqa, _mock_trial_from_sqa, _mock_gr_from_sqa
    ) -> None:
        for skip_runners_and_metrics in [False, True]:
            # 1. No abandoned arms + no trials case, reduced state should be the
            # same as non-reduced state.
            exp = get_experiment_with_batch_trial(constrain_search_space=False)
            save_experiment(exp)
            loaded_experiment = load_experiment(
                exp.name,
                reduced_state=True,
                skip_runners_and_metrics=skip_runners_and_metrics,
            )
            loaded_experiment.runner = exp.runner
            self.assertEqual(loaded_experiment, exp)
            # Make sure decoder function was called with `reduced_state=True`.
            self.assertTrue(_mock_exp_from_sqa.call_args[1].get("reduced_state"))
            self.assertTrue(_mock_trial_from_sqa.call_args[1].get("reduced_state"))
            self.assertTrue(_mock_gr_from_sqa.call_args[1].get("reduced_state"))
            _mock_exp_from_sqa.reset_mock()

            # 3. Try case with model state and search space + opt.config on a
            # generator run in the experiment.
            gr = Generators.SOBOL(experiment=exp).gen(1)
            # Expecting model kwargs to have 7 fields (seed, deduplicate, init_position,
            # scramble, generated_points, fallback_to_sample_polytope,
            # polytope_sampler_kwargs)
            # and the rest of model-state info on generator run to have values too.
            generator_kwargs = gr._generator_kwargs
            self.assertIsNotNone(generator_kwargs)
            self.assertEqual(len(generator_kwargs), 7)
            adapter_kwargs = gr._adapter_kwargs
            self.assertIsNotNone(adapter_kwargs)
            self.assertEqual(len(adapter_kwargs), 6)
            # This has seed and init position.
            ms = gr._generator_state_after_gen
            self.assertIsNotNone(ms)
            self.assertEqual(len(ms), 2)
            gm = gr._gen_metadata
            self.assertIsNotNone(gm)
            self.assertEqual(len(gm), 0)
            self.assertIsNotNone(gr._search_space, gr.optimization_config)
            exp.new_trial(generator_run=gr)
            save_experiment(exp)
            # Make sure that all relevant decoding functions were called with
            # `reduced_state=True` and correct number of times.
            loaded_experiment = load_experiment(
                exp.name,
                reduced_state=True,
                skip_runners_and_metrics=skip_runners_and_metrics,
            )
            loaded_experiment.runner = exp.runner
            self.assertTrue(_mock_exp_from_sqa.call_args[1].get("reduced_state"))
            self.assertTrue(_mock_trial_from_sqa.call_args[1].get("reduced_state"))
            # 2 generator runs from trial #0 + 1 from trial #1.
            self.assertTrue(_mock_gr_from_sqa.call_args[1].get("reduced_state"))
            self.assertNotEqual(loaded_experiment, exp)
            # Remove all fields that are not part of the reduced state and
            # check that everything else is equal as expected.
            exp.trials.get(1).generator_run._generator_kwargs = None
            exp.trials.get(1).generator_run._adapter_kwargs = None
            exp.trials.get(1).generator_run._gen_metadata = None
            exp.trials.get(1).generator_run._generator_state_after_gen = None
            exp.trials.get(1).generator_run._search_space = None
            exp.trials.get(1).generator_run._optimization_config = None
            self.assertEqual(loaded_experiment, exp)
            delete_experiment(exp_name=exp.name)

    def test_load_and_save_reduced_state_does_not_lose_abandoned_arms(self) -> None:
        exp = get_experiment_with_batch_trial(constrain_search_space=False)
        self.assertEqual(len(exp.trials[0].abandoned_arms), 1)
        save_experiment(exp)
        loaded_experiment = load_experiment(
            exp.name, reduced_state=True, skip_runners_and_metrics=True
        )
        save_experiment(loaded_experiment)
        reloaded_experiment = load_experiment(exp.name)
        self.assertEqual(
            reloaded_experiment.trials[0].abandoned_arms,
            exp.trials[0].abandoned_arms,
        )
        self.assertEqual(len(reloaded_experiment.trials[0].abandoned_arms), 1)

    def test_experiment_save_and_load_gr_with_opt_config(self) -> None:
        exp = get_experiment_with_batch_trial(constrain_search_space=False)
        gr = Generators.SOBOL(experiment=exp).gen(
            n=1, optimization_config=exp.optimization_config
        )
        exp.new_trial(generator_run=gr)
        save_experiment(exp)
        loaded_experiment = load_experiment(
            exp.name,
            reduced_state=False,
            skip_runners_and_metrics=True,
        )
        self.assertEqual(loaded_experiment.trials[1], exp.trials[1])

    def test_load_gr_with_non_decodable_metadata_and_reduced_state(self) -> None:
        def spy(original_method: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args: Any, **kwargs: Any) -> T:
                # Check if a specific argument is set to a certain value
                if "reduced_state" in kwargs and not kwargs["reduced_state"]:
                    raise JSONDecodeError("Can't decode gen_metadata")
                return original_method(*args, **kwargs)

            return wrapper

        gs = get_generation_strategy(
            with_experiment=True,
        )
        gs.gen_single_trial(experiment=gs.experiment)
        gs.gen_single_trial(experiment=gs.experiment)

        save_experiment(gs.experiment)
        save_generation_strategy(gs)

        with self.assertLogs("ax", level=logging.ERROR):
            with patch.object(
                Decoder, "generator_run_from_sqa", spy(Decoder.generator_run_from_sqa)
            ):
                load_generation_strategy_by_id(
                    gs_id=none_throws(gs.db_id),
                    experiment=gs.experiment,
                    reduced_state=True,
                )

    def test_mt_experiment_save_and_load(self) -> None:
        experiment = get_multi_type_experiment(add_trials=True)
        save_experiment(experiment)
        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(loaded_experiment.default_trial_type, "type1")
        self.assertEqual(len(loaded_experiment._trial_type_to_runner), 2)
        # pyre-fixme[16]: `Experiment` has no attribute `metric_to_trial_type`.
        self.assertEqual(loaded_experiment.metric_to_trial_type["m1"], "type1")
        self.assertEqual(loaded_experiment.metric_to_trial_type["m2"], "type2")
        # pyre-fixme[16]: `Experiment` has no attribute `_metric_to_canonical_name`.
        self.assertEqual(loaded_experiment._metric_to_canonical_name["m2"], "m1")
        self.assertEqual(len(loaded_experiment.trials), 2)

    def test_mt_experiment_save_and_load_skip_runners_and_metrics(self) -> None:
        experiment = get_multi_type_experiment(add_trials=True)
        save_experiment(experiment)
        loaded_experiment = load_experiment(
            experiment.name, skip_runners_and_metrics=True
        )
        self.assertEqual(loaded_experiment.default_trial_type, "type1")
        self.assertIsNone(loaded_experiment._trial_type_to_runner["type1"])
        self.assertIsNone(loaded_experiment._trial_type_to_runner["type2"])
        # pyre-fixme[16]: `Experiment` has no attribute `metric_to_trial_type`.
        self.assertEqual(loaded_experiment.metric_to_trial_type["m1"], "type1")
        self.assertEqual(loaded_experiment.metric_to_trial_type["m2"], "type2")
        # pyre-fixme[16]: `Experiment` has no attribute `_metric_to_canonical_name`.
        self.assertEqual(loaded_experiment._metric_to_canonical_name["m2"], "m1")
        self.assertEqual(len(loaded_experiment.trials), 2)

    def test_experiment_new_trial(self) -> None:
        # Create a new trial without data
        save_experiment(self.experiment)
        trial = self.experiment.new_batch_trial()
        save_or_update_trial(experiment=self.experiment, trial=trial)

        loaded_experiment = load_experiment(self.experiment.name)
        self.assertEqual(self.experiment, loaded_experiment)

        # Create a new trial with data
        trial = self.experiment.new_batch_trial(generator_run=get_generator_run())
        self.experiment.attach_data(get_data(trial_index=trial.index))
        save_or_update_trial(experiment=self.experiment, trial=trial)

        loaded_experiment = load_experiment(self.experiment.name)
        self.assertEqual(self.experiment, loaded_experiment)

    def test_experiment_new_trial_validation(self) -> None:
        trial = self.experiment.new_batch_trial()

        with self.assertRaises(ValueError):
            # must save experiment first
            save_or_update_trial(experiment=self.experiment, trial=trial)

    def test_experiment_update_trial(self) -> None:
        save_experiment(self.experiment)

        trial = self.experiment.trials[0]
        trial.mark_staged()
        save_or_update_trial(experiment=self.experiment, trial=trial)

        # Update a trial by changing metadata
        trial._run_metadata = {"foo": "bar"}
        save_or_update_trial(experiment=self.experiment, trial=trial)

        loaded_experiment = load_experiment(self.experiment.name)
        self.assertEqual(self.experiment, loaded_experiment)

        # Update a trial by attaching data
        data = get_data(trial_index=trial.index)
        self.experiment.attach_data(data)
        save_or_update_trial(experiment=self.experiment, trial=trial)
        # make sure loaded experiment has data
        loaded_experiment = load_experiment(self.experiment.name)
        self.assertEqual(self.experiment, loaded_experiment)
        # verify that saving is idempotent, by saving the original experiment again
        # and reloading
        save_or_update_trial(experiment=self.experiment, trial=trial)
        loaded_experiment = load_experiment(self.experiment.name)
        self.assertEqual(self.experiment, loaded_experiment)

        # Update a trial by attaching data again
        self.experiment.attach_data(get_data(trial_index=trial.index))
        save_or_update_trial(experiment=self.experiment, trial=trial)

        loaded_experiment = load_experiment(self.experiment.name)
        self.assertEqual(self.experiment, loaded_experiment)

    def test_experiment_save_and_update_trials(self) -> None:
        save_experiment(self.experiment)

        existing_trial = self.experiment.trials[0]
        existing_trial.mark_staged()
        new_trial = self.experiment.new_batch_trial(generator_run=get_generator_run())
        save_or_update_trials(
            experiment=self.experiment, trials=[existing_trial, new_trial]
        )

        loaded_experiment = load_experiment(self.experiment.name)
        self.assertEqual(self.experiment, loaded_experiment)

        self.experiment.attach_data(get_data(trial_index=new_trial.index))
        new_trial_2 = self.experiment.new_batch_trial(generator_run=get_generator_run())
        save_or_update_trials(
            experiment=self.experiment,
            trials=[existing_trial, new_trial, new_trial_2],
            batch_size=2,
        )
        loaded_experiment = load_experiment(self.experiment.name)
        self.assertEqual(self.experiment, loaded_experiment)

        exp = get_branin_experiment_with_timestamp_map_metric()
        save_experiment(exp)
        generator_run = Generators.SOBOL(experiment=exp).gen(n=1)
        trial = exp.new_trial(generator_run=generator_run)
        exp.attach_data(trial.run().fetch_data())
        save_or_update_trials(
            experiment=exp,
            trials=[trial],
            batch_size=2,
        )
        loaded_experiment = load_experiment(exp.name)
        self.assertEqual(exp, loaded_experiment)

    def test_save_validation(self) -> None:
        with self.assertRaises(ValueError):
            save_experiment(self.experiment.trials[0])

        experiment = get_experiment_with_batch_trial()
        # pyre-fixme[8]: Attribute has type `str`; used as `None`.
        experiment.name = None
        with self.assertRaises(ValueError):
            save_experiment(experiment)

    def test_encode_decode(self) -> None:
        for class_, fake_func, unbound_encode_func, unbound_decode_func in TEST_CASES:
            # Can't load trials from SQL, because a trial needs an experiment
            # in order to be initialized
            if class_ == "BatchTrial" or class_ == "Trial":
                continue
            original_object = fake_func()
            encode_func = unbound_encode_func.__get__(self.encoder)
            decode_func = unbound_decode_func.__get__(self.decoder)
            sqa_object = encode_func(original_object)

            if class_ in ["OrderConstraint", "ParameterConstraint", "SumConstraint"]:
                converted_object = decode_func(sqa_object, self.dummy_parameters)
            elif class_ == "GeneratorRun" or class_ == "GeneratorRunReducedState":
                # Need to pass in reduced_state and immutable_oc_and_ss
                converted_object = decode_func(sqa_object, False, False)
            elif isinstance(sqa_object, tuple):
                converted_object = decode_func(*sqa_object)
            else:
                converted_object = decode_func(sqa_object)

            if class_ == "SimpleExperiment":
                # Evaluation functions will be different, so need to do
                # this so equality test passes
                with self.assertRaises(RuntimeError):
                    converted_object.evaluation_function(parameterization={})

                original_object.evaluation_function = None
                converted_object.evaluation_function = None
                # Experiment to SQA encoder stores the experiment subclass
                # among its properties; we then remove the subclass when
                # decoding. Removing subclass from original object here
                # for parity with the expected decoded (converted) object.
                original_object._properties.pop(Keys.SUBCLASS)

            self.assertEqual(
                original_object,
                converted_object,
                msg=f"Error encoding/decoding {class_}.",
            )

    def test_encode_generator_run_reduced_state(self) -> None:
        exp = get_branin_experiment()
        gs = get_generation_strategy()
        gr = gs.gen_single_trial(experiment=exp)

        for key in [attr.key for attr in GR_LARGE_MODEL_ATTRS]:
            python_attr = SQA_COL_TO_GR_ATTR[key]
            self.assertIsNotNone(getattr(gr, f"_{python_attr}"))

        gr_sqa_reduced_state = self.encoder.generator_run_to_sqa(
            generator_run=gr, weight=None, reduced_state=True
        )

        gr_decoded_reduced_state = self.decoder.generator_run_from_sqa(
            gr_sqa_reduced_state,
            reduced_state=False,
            immutable_search_space_and_opt_config=False,
        )

        for key in [attr.key for attr in GR_LARGE_MODEL_ATTRS]:
            python_attr = SQA_COL_TO_GR_ATTR[key]
            setattr(gr, f"_{python_attr}", None)

        self.assertEqual(gr, gr_decoded_reduced_state)

    def test_load_and_save_generator_run_reduced_state(self) -> None:
        exp = get_branin_experiment()
        gs = get_generation_strategy()
        gr = gs.gen_single_trial(experiment=exp)
        original_gen_metadata = {"foo": "bar"}
        gr._gen_metadata = original_gen_metadata
        exp.new_trial(generator_run=gr)
        save_experiment(exp)

        loaded_reduced_state = load_experiment(
            experiment_name=exp.name, reduced_state=True
        )
        save_experiment(loaded_reduced_state)
        reloaded_experiment = load_experiment(exp.name, reduced_state=False)

        self.assertEqual(
            original_gen_metadata,
            reloaded_experiment.trials[0].generator_runs[0].gen_metadata,
        )

    def test_experiment_updates(self) -> None:
        experiment = get_experiment(with_status_quo=False)
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAExperiment).count(), 1)

        # update experiment
        # (should perform update in place)
        experiment.description = "foobar"
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAExperiment).count(), 1)

        experiment.status_quo = Arm(
            parameters={"w": 0.0, "x": 1, "y": "y", "z": True, "d": 1.0}
        )
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAExperiment).count(), 1)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_experiment_parameter_updates(self) -> None:
        experiment = get_experiment(with_status_quo=False)
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAParameter).count(),
            len(experiment.search_space.parameters),
        )

        # update a parameter
        # (should perform update in place)
        search_space = get_search_space()
        parameter = get_choice_parameter()
        parameter.add_values(["foobar"])
        search_space.update_parameter(parameter)
        experiment.search_space = search_space
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAParameter).count(),
            len(experiment.search_space.parameters),
        )

        # add a parameter
        parameter = RangeParameter(
            name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
        )
        search_space.add_parameter(parameter)
        experiment.search_space = search_space
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAParameter).count(),
            len(experiment.search_space.parameters),
        )

        # remove a parameter
        # (old one should be deleted)
        del search_space._parameters["x1"]
        experiment.search_space = search_space
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAParameter).count(),
            len(experiment.search_space.parameters),
        )

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_experiment_parameter_constraint_updates(self) -> None:
        experiment = get_experiment_with_batch_trial()
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAParameterConstraint).count(),  # 3
            len(experiment.search_space.parameter_constraints),  # 3
        )

        # add a parameter constraint
        search_space = experiment.search_space
        existing_constraint = experiment.search_space.parameter_constraints[0]
        new_constraint = get_sum_constraint2()
        search_space.add_parameter_constraints([new_constraint])
        experiment.search_space = search_space
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAParameterConstraint).count(),
            len(experiment.search_space.parameter_constraints),
        )

        # update a parameter constraint
        # (since we don't have UIDs for these, we throw out the old one
        # and create a new one)
        new_constraint.bound = 5.0
        search_space.set_parameter_constraints([existing_constraint, new_constraint])
        experiment.search_space = search_space
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAParameterConstraint).count(),
            len(experiment.search_space.parameter_constraints),
        )

        # remove a parameter constraint
        # (old one should be deleted)
        search_space.set_parameter_constraints([new_constraint])
        experiment.search_space = search_space
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAParameterConstraint).count(),
            len(experiment.search_space.parameter_constraints),
        )

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_experiment_objective_updates(self) -> None:
        experiment = get_experiment_with_batch_trial()
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics)
        )

        # update objective
        # (should perform update in place)
        optimization_config = get_optimization_config()
        objective = get_objective()
        objective.minimize = True
        optimization_config.objective = objective
        experiment.optimization_config = optimization_config
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics)
        )

        # replace objective
        # (old one should become tracking metric)
        optimization_config.objective = Objective(
            metric=Metric(name="objective"), minimize=False
        )
        experiment.optimization_config = optimization_config
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics)
        )

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_experiment_outcome_constraint_updates(self) -> None:
        experiment = get_experiment_with_batch_trial()
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics)
        )

        # update outcome constraint
        # (should perform update in place)
        optimization_config = get_optimization_config()
        outcome_constraint = get_outcome_constraint()
        outcome_constraint.bound = -1.0
        optimization_config.outcome_constraints = [outcome_constraint]
        experiment.optimization_config = optimization_config
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics)
        )

        # add outcome constraint
        outcome_constraint2 = OutcomeConstraint(
            metric=Metric(name="outcome"), op=ComparisonOp.GEQ, bound=-0.5
        )
        optimization_config.outcome_constraints = [
            outcome_constraint,
            outcome_constraint2,
        ]
        experiment.optimization_config = optimization_config
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics)
        )

        # add a scalarized outcome constraint
        outcome_constraint3 = get_scalarized_outcome_constraint()
        optimization_config.outcome_constraints = [
            outcome_constraint,
            outcome_constraint3,
        ]
        experiment.optimization_config = optimization_config
        save_experiment(experiment)
        # one more for `scalarized_outcome_constraint`
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics) + 1
        )

        # remove outcome constraint
        # (old one should become tracking metric)
        optimization_config.outcome_constraints = [outcome_constraint]
        experiment.optimization_config = optimization_config
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics)
        )

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_experiment_pruning_target_arm(self) -> None:
        experiment = get_experiment_with_batch_trial()
        pruning_target_parameterization = next(
            iter(experiment.arms_by_name.values())
        ).clone()
        none_throws(
            experiment.optimization_config
        ).pruning_target_parameterization = pruning_target_parameterization
        save_experiment(experiment)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)
        self.assertEqual(
            none_throws(
                loaded_experiment.optimization_config
            ).pruning_target_parameterization,
            pruning_target_parameterization,
        )

    def test_optimization_config_pruning_target_parameterization_sqa_roundtrip(
        self,
    ) -> None:
        # Setup: create experiment with basic OptimizationConfig and
        # pruning_target_parameterization
        experiment = get_experiment_with_batch_trial()
        pruning_target_parameterization = Arm(
            parameters={"w": 1.5, "x": 2.5, "y": "choice_1", "z": False}
        )

        optimization_config = OptimizationConfig(
            objective=get_objective(),
            outcome_constraints=[get_outcome_constraint()],
            pruning_target_parameterization=pruning_target_parameterization,
        )
        experiment.optimization_config = optimization_config

        # Execute: save and load experiment through SQA store
        save_experiment(experiment)
        loaded_experiment = load_experiment(experiment.name)

        # Assert: confirm pruning_target_parameterization is preserved correctly
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
        # Verify the target arm parameters are correct
        loaded_pruning_target_parameterization = none_throws(
            none_throws(
                loaded_experiment.optimization_config
            ).pruning_target_parameterization
        )
        self.assertEqual(loaded_pruning_target_parameterization.parameters["w"], 1.5)
        self.assertEqual(loaded_pruning_target_parameterization.parameters["x"], 2.5)
        self.assertEqual(
            loaded_pruning_target_parameterization.parameters["y"], "choice_1"
        )
        self.assertEqual(loaded_pruning_target_parameterization.parameters["z"], False)

    def test_multi_objective_optimization_config_pruning_target_sqa_roundtrip(
        self,
    ) -> None:
        # Test that MultiObjectiveOptimizationConfig with
        # pruning_target_parameterization can be saved/loaded correctly

        # Setup: create experiment with MultiObjectiveOptimizationConfig and
        # pruning_target_parameterization
        experiment = get_experiment_with_batch_trial()
        pruning_target_parameterization = next(
            iter(experiment.arms_by_name.values())
        ).clone()

        multi_objective_config = MultiObjectiveOptimizationConfig(
            objective=get_multi_objective_optimization_config().objective,
            pruning_target_parameterization=pruning_target_parameterization,
        )
        # Can't use experiment.clone_with_args, so create new experiment
        experiment = Experiment(
            name=experiment.name,
            search_space=experiment.search_space,
            optimization_config=multi_objective_config,
            description=experiment.description,
            is_test=experiment.is_test,
        )
        # Copy trials from original experiment
        for trial_idx, trial in experiment.trials.items():
            experiment._trials[trial_idx] = trial

        # Execute: save and load experiment through SQA store
        save_experiment(experiment)
        loaded_experiment = load_experiment(experiment.name)

        # Assert: confirm pruning_target_parameterization is preserved correctly
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

    def test_pruning_target_update_sqa_roundtrip(self) -> None:
        # Test that pruning_target_parameterization can be updated and modifications
        # are preserved

        # Setup: create experiment with initial pruning_target_parameterization
        experiment = get_experiment_with_batch_trial()
        initial_pruning_target_parameterization = next(
            iter(experiment.arms_by_name.values())
        ).clone()
        none_throws(
            experiment.optimization_config
        ).pruning_target_parameterization = initial_pruning_target_parameterization
        save_experiment(experiment)

        # Execute: update pruning_target_parameterization with different parameters

        updated_pruning_target_parameterization = Arm(
            parameters={"x1": 999.0, "x2": 888.0}
        )
        none_throws(
            experiment.optimization_config
        ).pruning_target_parameterization = updated_pruning_target_parameterization
        save_experiment(experiment)

        loaded_experiment = load_experiment(experiment.name)

        # Assert: confirm updated pruning_target_parameterization is preserved correctly
        self.assertEqual(experiment, loaded_experiment)
        self.assertIsNotNone(
            none_throws(
                loaded_experiment.optimization_config
            ).pruning_target_parameterization
        )
        self.assertEqual(
            none_throws(
                loaded_experiment.optimization_config
            ).pruning_target_parameterization,
            updated_pruning_target_parameterization,
        )
        # Confirm it's different from initial pruning_target_parameterization
        self.assertNotEqual(
            none_throws(
                loaded_experiment.optimization_config
            ).pruning_target_parameterization,
            initial_pruning_target_parameterization,
        )

    def test_pruning_target_none_to_arm_sqa_roundtrip(self) -> None:
        # Test that pruning_target_parameterization can be set from None to a
        # valid arm

        # Setup: create experiment without pruning_target_parameterization
        experiment = get_experiment_with_batch_trial()
        none_throws(
            experiment.optimization_config
        ).pruning_target_parameterization = None
        save_experiment(experiment)

        # Execute: set pruning_target_parameterization to a valid arm
        new_pruning_target_parameterization = next(
            iter(experiment.arms_by_name.values())
        ).clone()
        none_throws(
            experiment.optimization_config
        ).pruning_target_parameterization = new_pruning_target_parameterization
        save_experiment(experiment)

        loaded_experiment = load_experiment(experiment.name)

        # Assert: confirm pruning_target_parameterization is correctly set
        self.assertEqual(experiment, loaded_experiment)
        self.assertIsNotNone(
            none_throws(
                loaded_experiment.optimization_config
            ).pruning_target_parameterization
        )
        self.assertEqual(
            none_throws(
                loaded_experiment.optimization_config
            ).pruning_target_parameterization,
            new_pruning_target_parameterization,
        )

    def test_pruning_target_to_none_sqa_roundtrip(self) -> None:
        # Test that pruning_target_parameterization can be set from a
        # valid arm to None

        # Setup: create experiment with pruning_target_parameterization
        experiment = get_experiment_with_batch_trial()
        initial_pruning_target_parameterization = next(
            iter(experiment.arms_by_name.values())
        ).clone()
        none_throws(
            experiment.optimization_config
        ).pruning_target_parameterization = initial_pruning_target_parameterization
        save_experiment(experiment)

        # Execute: set pruning_target_parameterization to None
        none_throws(
            experiment.optimization_config
        ).pruning_target_parameterization = None
        save_experiment(experiment)

        loaded_experiment = load_experiment(experiment.name)

        # Assert: confirm pruning_target_parameterization is correctly set to None
        self.assertEqual(experiment, loaded_experiment)
        self.assertIsNone(
            none_throws(
                loaded_experiment.optimization_config
            ).pruning_target_parameterization
        )

    def test_preference_optimization_config_sqa_roundtrip(self) -> None:
        # Test that PreferenceOptimizationConfig with expect_relativized_outcomes
        # can be saved/loaded correctly through SQA storage
        base_experiment = get_experiment_with_batch_trial()
        multi_objective = MultiObjective(
            objectives=[
                Objective(metric=Metric(name="m1"), minimize=False),
                Objective(metric=Metric(name="m2"), minimize=True),
            ]
        )

        test_test_profile = "test_profile_name"
        for expect_relativized_outcomes in (True, False):
            with self.subTest(f"{expect_relativized_outcomes=}"):
                pref_config = PreferenceOptimizationConfig(
                    objective=multi_objective,
                    preference_profile_name=test_test_profile,
                    expect_relativized_outcomes=expect_relativized_outcomes,
                )
                experiment = Experiment(
                    name=f"test_pref_opt_config_sqa_{expect_relativized_outcomes}",
                    search_space=base_experiment.search_space,
                    optimization_config=pref_config,
                    is_test=base_experiment.is_test,
                )

                save_experiment(experiment)
                loaded_experiment = load_experiment(experiment.name)

                loaded_config = assert_is_instance(
                    loaded_experiment.optimization_config, PreferenceOptimizationConfig
                )
                self.assertEqual(
                    loaded_config.preference_profile_name, test_test_profile
                )
                self.assertEqual(
                    loaded_config.expect_relativized_outcomes,
                    expect_relativized_outcomes,
                )
                loaded_objective = assert_is_instance(
                    loaded_config.objective, MultiObjective
                )
                self.assertEqual(
                    len(loaded_objective.objectives),
                    len(multi_objective.objectives),
                )
                loaded_metric_names = {
                    obj.metric.name for obj in loaded_objective.objectives
                }
                self.assertEqual(loaded_metric_names, {"m1", "m2"})

    def test_experiment_objective_threshold_updates(self) -> None:
        experiment = get_experiment_with_batch_trial()
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics)
        )

        # update objective threshold
        # (should perform update in place)
        optimization_config = get_multi_objective_optimization_config()
        objective_threshold = get_objective_threshold()
        objective_threshold2 = get_objective_threshold(
            "m3", bound=3.0, comparison_op=ComparisonOp.LEQ
        )
        optimization_config.objective_thresholds = [
            objective_threshold,
            objective_threshold2,
        ]
        experiment.optimization_config = optimization_config
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAMetric).count(), 7)
        self.assertIsNotNone(
            # pyre-fixme[16]: Optional type has no attribute `objective_thresholds`.
            experiment.optimization_config.objective_thresholds[0].metric.db_id
        )

        # add outcome constraint
        outcome_constraint2 = OutcomeConstraint(
            metric=Metric(name="outcome"), op=ComparisonOp.GEQ, bound=-0.5
        )
        optimization_config.outcome_constraints = [
            optimization_config.outcome_constraints[0],
            outcome_constraint2,
        ]
        experiment.optimization_config = optimization_config
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAMetric).count(), 8)

        # remove outcome constraint
        # (old one should become tracking metric)
        optimization_config.outcome_constraints = []
        experiment.optimization_config = optimization_config
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAMetric).count(), 6)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

        # Optimization config should correctly reload even with no
        # objective_thresholds
        optimization_config.objective_thresholds = []
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAMetric).count(), 4)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_failed_load(self) -> None:
        with self.assertRaises(ObjectNotFoundError):
            load_experiment("nonexistent_experiment")

    def test_experiment_tracking_metric_updates(self) -> None:
        experiment = get_experiment_with_batch_trial()
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics)
        )

        # update tracking metric
        # (should perform update in place)
        metric = Metric(name="tracking", lower_is_better=True)
        experiment.update_tracking_metric(metric)
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics)
        )

        # add tracking metric
        metric = Metric(name="tracking2")
        experiment.add_tracking_metric(metric)
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics)
        )

        # remove tracking metric
        # (old one should be deleted)
        experiment.remove_tracking_metric("tracking2")
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics)
        )

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_experiment_runner_updates(self) -> None:
        experiment = get_experiment_with_batch_trial()
        save_experiment(experiment)
        # one runner on the batch
        self.assertEqual(get_session().query(SQARunner).count(), 1)

        # add runner to experiment
        runner = get_synthetic_runner()
        experiment.runner = runner
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQARunner).count(), 1)

        # update runner
        # (should perform update in place)
        runner = get_synthetic_runner()
        # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Dict[str, str]`.
        runner.dummy_metadata = {"foo": "bar"}
        experiment.runner = runner
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQARunner).count(), 1)

        # remove runner
        # (old one should be deleted)
        experiment.runner = None
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQARunner).count(), 0)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_experiment_trial_updates(self) -> None:
        experiment = get_experiment_with_batch_trial()
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQATrial).count(), 1)
        self.assertEqual(get_session().query(SQARunner).count(), 1)

        # add trial
        trial = experiment.new_batch_trial()
        runner = get_synthetic_runner()
        trial.experiment.runner = runner
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQATrial).count(), 2)
        self.assertEqual(get_session().query(SQARunner).count(), 1)

        # update trial's runner
        runner.dummy_metadata = "dummy metadata"
        trial.experiment.runner = runner
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQATrial).count(), 2)
        self.assertEqual(get_session().query(SQARunner).count(), 1)

        trial.run()
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQATrial).count(), 2)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_experiment_abandoned_arm_updates(self) -> None:
        experiment = get_experiment_with_batch_trial()
        save_experiment(experiment)
        # one arm is already abandoned
        self.assertEqual(get_session().query(SQAAbandonedArm).count(), 1)

        trial = experiment.trials[0]
        trial.mark_arm_abandoned(trial.arms[1].name)
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAAbandonedArm).count(), 2)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_experiment_generator_run_updates(self) -> None:
        experiment = get_experiment_with_batch_trial()
        save_experiment(experiment)
        # one main generator run, one for the status quo
        self.assertEqual(get_session().query(SQAGeneratorRun).count(), 2)

        # add a arm
        # this will create one wrapper generator run
        # this will also replace the status quo generator run,
        # since the weight of the status quo will have changed
        trial = experiment.trials[0]
        trial.add_arm(get_arm())
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAGeneratorRun).count(), 3)

        generator_run = get_generator_run()
        trial.add_generator_run(generator_run=generator_run)
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAGeneratorRun).count(), 4)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_parameter_validation(self) -> None:
        sqa_parameter = SQAParameter(
            domain_type=DomainType.RANGE,
            parameter_type=ParameterType.FLOAT,
            name="foobar",
        )
        with self.assertRaises(ValueError):
            with session_scope() as session:
                session.add(sqa_parameter)

        sqa_parameter.experiment_id = 0
        with session_scope() as session:
            session.add(sqa_parameter)
        with self.assertRaises(ValueError):
            sqa_parameter.generator_run_id = 0
            with session_scope() as session:
                session.add(sqa_parameter)

        sqa_parameter = SQAParameter(
            domain_type=DomainType.RANGE,
            parameter_type=ParameterType.FLOAT,
            name="foobar",
            generator_run_id=0,
        )
        with session_scope() as session:
            session.add(sqa_parameter)
        with self.assertRaises(ValueError):
            sqa_parameter.experiment_id = 0
            with session_scope() as session:
                session.add(sqa_parameter)

    def test_parameter_decode_failure(self) -> None:
        parameter = get_fixed_parameter()
        sqa_parameter = self.encoder.parameter_to_sqa(parameter)
        # pyre-fixme[8]: Attribute has type `DomainType`; used as `int`.
        sqa_parameter.domain_type = 5
        with self.assertRaises(SQADecodeError):
            self.decoder.parameter_from_sqa(sqa_parameter)

    def test_logit_scale(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "logit-scale"):
            self.encoder.parameter_to_sqa(
                parameter=RangeParameter(
                    name="foo",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.1,
                    upper=0.99,
                    logit_scale=True,
                )
            )

    def test_bypass_cardinality_check(self) -> None:
        choice_parameter = ChoiceParameter(
            name="test_choice",
            parameter_type=ParameterType.INT,
            values=[1, 2, 3],
            bypass_cardinality_check=True,
        )
        with self.assertRaisesRegex(
            UnsupportedError,
            "`bypass_cardinality_check` should only be set to `True` "
            "when constructing parameters within the modeling layer. It is not "
            "supported for storage.",
        ):
            self.encoder.parameter_to_sqa(parameter=choice_parameter)

    def test_parameter_constraint_validation(self) -> None:
        sqa_parameter_constraint = SQAParameterConstraint(
            bound=Decimal(0),
            constraint_dict={},
            # pyre-fixme[6]: For 3rd param expected `IntEnum` but got
            #  `ParameterConstraintType`.
            type=ParameterConstraintType.LINEAR,
        )
        with self.assertRaises(ValueError):
            with session_scope() as session:
                session.add(sqa_parameter_constraint)

        sqa_parameter_constraint.experiment_id = 0
        with session_scope() as session:
            session.add(sqa_parameter_constraint)
        with self.assertRaises(ValueError):
            sqa_parameter_constraint.generator_run_id = 0
            with session_scope() as session:
                session.add(sqa_parameter_constraint)

        sqa_parameter_constraint = SQAParameterConstraint(
            bound=Decimal(0),
            constraint_dict={},
            # pyre-fixme[6]: For 3rd param expected `IntEnum` but got
            #  `ParameterConstraintType`.
            type=ParameterConstraintType.LINEAR,
            generator_run_id=0,
        )
        with session_scope() as session:
            session.add(sqa_parameter_constraint)
        with self.assertRaises(ValueError):
            sqa_parameter_constraint.experiment_id = 0
            with session_scope() as session:
                session.add(sqa_parameter_constraint)

    def test_decode_order_parameter_constraint_failure(self) -> None:
        sqa_parameter = SQAParameterConstraint(
            # pyre-fixme[6]: For 1st param expected `IntEnum` but got
            #  `ParameterConstraintType`.
            type=ParameterConstraintType.ORDER,
            constraint_dict={},
            bound=Decimal(0),
        )
        with self.assertRaises(SQADecodeError):
            self.decoder.parameter_constraint_from_sqa(
                sqa_parameter, self.dummy_parameters
            )

    def test_decode_sum_parameter_constraint_failure(self) -> None:
        sqa_parameter = SQAParameterConstraint(
            # pyre-fixme[6]: For 1st param expected `IntEnum` but got
            #  `ParameterConstraintType`.
            type=ParameterConstraintType.SUM,
            constraint_dict={},
            bound=Decimal(0),
        )
        with self.assertRaises(SQADecodeError):
            self.decoder.parameter_constraint_from_sqa(
                sqa_parameter, self.dummy_parameters
            )

    def test_metric_validation(self) -> None:
        sqa_metric = SQAMetric(
            name="foobar",
            intent=MetricIntent.OBJECTIVE,
            metric_type=CORE_METRIC_REGISTRY[BraninMetric],
            signature="foobar",
        )
        with self.assertRaises(ValueError):
            with session_scope() as session:
                session.add(sqa_metric)

        sqa_metric.experiment_id = 0
        with session_scope() as session:
            session.add(sqa_metric)
        with self.assertRaises(ValueError):
            sqa_metric.generator_run_id = 0
            with session_scope() as session:
                session.add(sqa_metric)

        sqa_metric = SQAMetric(
            name="foobar",
            signature="foobar",
            intent=MetricIntent.OBJECTIVE,
            metric_type=CORE_METRIC_REGISTRY[BraninMetric],
            generator_run_id=0,
        )
        with session_scope() as session:
            session.add(sqa_metric)
        with self.assertRaises(ValueError):
            sqa_metric.experiment_id = 0
            with session_scope() as session:
                session.add(sqa_metric)

    def test_metric_decode_with_no_signature_override(self) -> None:
        metric_name = "testMetric"
        testMetric = Metric(name=metric_name)
        sqa_metric = self.encoder.metric_to_sqa(testMetric)
        metric = cast(Metric, self.decoder.metric_from_sqa(sqa_metric))
        self.assertEqual(metric.name, metric_name)
        self.assertEqual(metric.signature, metric_name)

    def test_metric_decode_with_signature_override(self) -> None:
        metric_name = "testMetric"
        testMetric = Metric(name=metric_name, signature_override="override")
        sqa_metric = self.encoder.metric_to_sqa(testMetric)
        metric = cast(Metric, self.decoder.metric_from_sqa(sqa_metric))
        self.assertEqual(metric.name, metric_name)
        self.assertEqual(metric.signature, "override")

    def test_metric_encode_captures_signature(self) -> None:
        # with override
        metric = get_branin_metric()
        metric.signature_override = "override"
        sqa_metric = self.encoder.metric_to_sqa(metric)
        self.assertEqual(sqa_metric.signature, "override")

        # without override
        metric = get_branin_metric()
        sqa_metric = self.encoder.metric_to_sqa(metric)
        self.assertEqual(sqa_metric.signature, metric.name)

    def test_metric_encode_failure(self) -> None:
        metric = get_branin_metric()
        del metric.__dict__["param_names"]
        with self.assertRaises(AttributeError):
            self.encoder.metric_to_sqa(metric)

    def test_metric_decode_failure(self) -> None:
        metric = get_branin_metric()
        sqa_metric = self.encoder.metric_to_sqa(metric)
        # pyre-fixme[8]: Attribute has type `int`; used as `str`.
        sqa_metric.metric_type = "foobar"
        with self.assertRaises(SQADecodeError):
            self.decoder.metric_from_sqa(sqa_metric)

        sqa_metric.metric_type = CORE_METRIC_REGISTRY[BraninMetric]
        # pyre-fixme[8]: Attribute has type `MetricIntent`; used as `str`.
        sqa_metric.intent = "foobar"
        with self.assertRaises(SQADecodeError):
            self.decoder.metric_from_sqa(sqa_metric)

        sqa_metric.intent = MetricIntent.TRACKING
        sqa_metric.properties = {}
        with self.assertRaises(ValueError):
            self.decoder.metric_from_sqa(sqa_metric)

    def test_objective_threshold_from_sqa_with_relative_none(self) -> None:
        # Setup: Create a metric and metric_sqa with relative set to None
        metric = get_branin_metric()
        metric_sqa = SQAMetric(
            name="test_metric",
            intent=MetricIntent.OBJECTIVE_THRESHOLD,
            metric_type=CORE_METRIC_REGISTRY[BraninMetric],
            signature="test_metric",
            bound=Decimal(10.0),
            relative=None,  # Set relative to None
        )

        # Execute: Call _objective_threshold_from_sqa and verify it logs a warning
        with self.assertLogs("ax", level=logging.WARNING) as logs:
            objective_threshold = self.decoder._objective_threshold_from_sqa(
                metric=metric, metric_sqa=metric_sqa
            )

        # Assert: Verify the warning message appears in logs
        self.assertTrue(
            any("When decoding SQAMetric" in output for output in logs.output),
            f"Expected warning log not found. Logs: {logs.output}",
        )

        # Assert: Verify the returned ObjectiveThreshold has relative set to False
        # (not None)
        self.assertIsNotNone(objective_threshold)
        self.assertEqual(objective_threshold.relative, False)
        self.assertEqual(objective_threshold.bound, 10.0)

    def test_runner_decode_failure(self) -> None:
        runner = get_synthetic_runner()
        sqa_runner = self.encoder.runner_to_sqa(runner)
        # pyre-fixme[8]: Attribute has type `int`; used as `str`.
        sqa_runner.runner_type = "foobar"
        with self.assertRaises(SQADecodeError):
            self.decoder.runner_from_sqa(sqa_runner)

    def test_runner_validation(self) -> None:
        sqa_runner = SQARunner(runner_type=CORE_RUNNER_REGISTRY[SyntheticRunner])
        with self.assertRaises(ValueError):
            with session_scope() as session:
                session.add(sqa_runner)

        sqa_runner.experiment_id = 0
        with session_scope() as session:
            session.add(sqa_runner)
        with self.assertRaises(ValueError):
            sqa_runner.trial_id = 0
            with session_scope() as session:
                session.add(sqa_runner)

        sqa_runner = SQARunner(
            runner_type=CORE_RUNNER_REGISTRY[SyntheticRunner], trial_id=0
        )
        with session_scope() as session:
            session.add(sqa_runner)
        with self.assertRaises(ValueError):
            sqa_runner.experiment_id = 0
            with session_scope() as session:
                session.add(sqa_runner)

    def test_timestamp_update(self) -> None:
        self.experiment.trials[0]._time_staged = datetime.now()
        save_experiment(self.experiment)

        # second save should not fail
        save_experiment(self.experiment)

    def test_get_properties(self) -> None:
        # Extract default value.
        properties = serialize_init_args(obj=Metric(name="foo"))
        self.assertEqual(
            properties,
            {
                "name": "foo",
                "lower_is_better": None,
                "properties": {},
                "signature_override": None,
            },
        )

        # Extract passed value.
        properties = serialize_init_args(
            obj=Metric(
                name="foo",
                lower_is_better=True,
                properties={"foo": "bar"},
                signature_override="foo_signature",
            )
        )
        self.assertEqual(
            properties,
            {
                "name": "foo",
                "lower_is_better": True,
                "properties": {"foo": "bar"},
                "signature_override": "foo_signature",
            },
        )

    def test_registry_additions(self) -> None:
        class MyRunner(Runner):
            def run():
                pass

            def staging_required():
                return False

        class MyMetric(Metric):
            pass

        (
            metric_registry,
            partial_encoder_registry,
            partial_decoder_registry,
        ) = register_metrics(metric_clss={MyMetric: None, Metric: 0})
        self.assertEqual(metric_registry, {MyMetric: mock.ANY, Metric: 0})
        runner_registry, encoder_registry, decoder_registry = register_runner(
            runner_cls=MyRunner,
            encoder_registry=partial_encoder_registry,
            decoder_registry=partial_decoder_registry,
        )

        sqa_config = SQAConfig(
            json_encoder_registry=encoder_registry,
            json_decoder_registry=decoder_registry,
            metric_registry=metric_registry,
            runner_registry=runner_registry,
        )

        experiment = get_experiment_with_batch_trial()
        experiment.runner = MyRunner()
        experiment.add_tracking_metric(MyMetric(name="my_metric"))
        save_experiment(experiment, config=sqa_config)
        loaded_experiment = load_experiment(experiment.name, config=sqa_config)
        self.assertEqual(loaded_experiment, experiment)

    def test_registry_bundle(self) -> None:
        class MyRunner(Runner):
            def run():
                pass

            def staging_required():
                return False

        class MyMetric(Metric):
            pass

        bundle = RegistryBundle(
            metric_clss={MyMetric: 1998, Metric: 0}, runner_clss={MyRunner: None}
        )
        self.assertEqual(bundle.metric_registry, {MyMetric: 1998, Metric: 0})

        experiment = get_experiment_with_batch_trial()
        experiment.runner = MyRunner()
        experiment.add_tracking_metric(MyMetric(name="my_metric"))
        save_experiment(experiment, config=bundle.sqa_config)
        loaded_experiment = load_experiment(experiment.name, config=bundle.sqa_config)
        self.assertEqual(loaded_experiment, experiment)

    def test_encode_decode_generation_strategy_base_case(self) -> None:
        # Cannot load generation strategy before it has been saved
        with self.assertRaises(ObjectNotFoundError):
            load_generation_strategy_by_id(gs_id=0)

        # Check that we can encode and decode the generation strategy *before*
        # it has generated some trials and been updated with some data.
        generation_strategy = get_generation_strategy()
        # Check that we can save a generation strategy without an experiment
        # attached.
        save_generation_strategy(generation_strategy=generation_strategy)
        # Also try restoring this generation strategy by its ID in the DB.
        new_generation_strategy = load_generation_strategy_by_id(
            # pyre-fixme[6]: For 1st param expected `int` but got `Optional[int]`.
            gs_id=generation_strategy._db_id
        )
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(generation_strategy, new_generation_strategy)
        self.assertIsNone(generation_strategy._experiment)

        # Cannot load generation strategy before it has been saved
        experiment = get_branin_experiment()
        save_experiment(experiment)
        with self.assertRaises(ObjectNotFoundError):
            load_generation_strategy_by_experiment_name(experiment_name=experiment.name)

        # Check that we can encode and decode the generation strategy *after*
        # it has generated some trials and been updated with some data.
        # Since we now need to `gen`, we remove the fake callable kwarg we added,
        # since model does not expect it.
        generation_strategy = get_generation_strategy()
        experiment.new_trial(
            generation_strategy.gen_single_trial(experiment=experiment)
        )
        generation_strategy.gen_single_trial(experiment, data=get_branin_data())
        save_experiment(experiment)
        save_generation_strategy(generation_strategy=generation_strategy)
        # Try restoring the generation strategy using the experiment its
        # attached to.
        new_generation_strategy = load_generation_strategy_by_experiment_name(
            experiment_name=experiment.name
        )
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(generation_strategy, new_generation_strategy)
        self.assertEqual(
            new_generation_strategy._nodes[0].generator_specs[0].generator_enum,
            Generators.SOBOL,
        )
        self.assertEqual(len(new_generation_strategy._generator_runs), 2)
        self.assertEqual(
            none_throws(new_generation_strategy._experiment)._name, experiment._name
        )

    def test_encode_decode_generation_node_gs_with_advanced_settings(self) -> None:
        """Test to ensure that GenerationNode based GenerationStrategies are
        able to be encoded/decoded correctly. This test adds transition criteria
        and input constructors to the nodes in the generation strategy.
        """
        generation_strategy = sobol_gpei_generation_node_gs(
            with_input_constructors_all_n=True
        )

        # Try restoring this generation strategy by its ID in the DB.
        save_generation_strategy(generation_strategy=generation_strategy)
        new_generation_strategy = load_generation_strategy_by_id(
            # pyre-fixme[6]: For 1st param expected `int` but got `Optional[int]`.
            gs_id=generation_strategy._db_id
        )

        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(generation_strategy, new_generation_strategy)
        self.assertIsNone(generation_strategy._experiment)
        experiment = get_branin_experiment()
        save_experiment(experiment)

        # Check that we can encode and decode the generation strategy *after*
        # it has generated some trials and been updated with some data.
        # Since we now need to `gen`, we remove the fake callable kwarg we added,
        # since model does not expect it.
        generation_strategy = sobol_gpei_generation_node_gs(
            with_input_constructors_all_n=True
        )
        experiment.new_batch_trial(
            generator_runs=generation_strategy.gen(experiment=experiment)[0]
        )
        generation_strategy.gen(experiment, data=get_branin_data())
        save_experiment(experiment)
        save_generation_strategy(generation_strategy=generation_strategy)

        # Try restoring the generation strategy using the experiment its
        # attached to.
        new_generation_strategy = load_generation_strategy_by_experiment_name(
            experiment_name=experiment.name
        )
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(generation_strategy, new_generation_strategy)
        self.assertIsInstance(
            new_generation_strategy._nodes[0].generator_spec_to_gen_from.generator_enum,
            Generators,
        )
        self.assertEqual(len(new_generation_strategy._generator_runs), 2)
        self.assertEqual(
            none_throws(new_generation_strategy._experiment)._name, experiment._name
        )

    def test_encode_decode_generation_node_based_generation_strategy(self) -> None:
        """Test to ensure that GenerationNode based GenerationStrategies are
        able to be encoded/decoded correctly.
        """
        # we don't support callable models for GenNode based strategies
        generation_strategy = get_generation_strategy()
        # Check that we can save a generation strategy without an experiment
        # attached.
        save_generation_strategy(generation_strategy=generation_strategy)
        # Also try restoring this generation strategy by its ID in the DB.
        new_generation_strategy = load_generation_strategy_by_id(
            gs_id=none_throws(generation_strategy._db_id)
        )
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(generation_strategy, new_generation_strategy)
        self.assertIsNone(generation_strategy._experiment)

        # Cannot load generation strategy before it has been saved
        experiment = get_branin_experiment()
        save_experiment(experiment)
        with self.assertRaises(ObjectNotFoundError):
            load_generation_strategy_by_experiment_name(experiment_name=experiment.name)

        # Check that we can encode and decode the generation strategy *after*
        # it has generated some trials and been updated with some data.
        generation_strategy = get_generation_strategy()
        experiment.new_trial(
            generation_strategy.gen_single_trial(experiment=experiment)
        )
        generation_strategy.gen_single_trial(experiment, data=get_branin_data())
        save_experiment(experiment)

        save_generation_strategy(generation_strategy=generation_strategy)
        # Try restoring the generation strategy using the experiment its
        # attached to.
        new_generation_strategy = load_generation_strategy_by_experiment_name(
            experiment_name=experiment.name
        )
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(generation_strategy, new_generation_strategy)
        self.assertIsInstance(
            new_generation_strategy._nodes[0].generator_spec_to_gen_from.generator_enum,
            Generators,
        )
        self.assertEqual(len(new_generation_strategy._generator_runs), 2)
        self.assertEqual(
            none_throws(new_generation_strategy._experiment)._name, experiment._name
        )

    def test_encode_decode_generation_strategy_reduced_state(self) -> None:
        """Try restoring the generation strategy using the experiment its attached to,
        passing the experiment object.
        """
        generation_strategy = get_generation_strategy()
        experiment = get_branin_experiment()
        experiment.new_trial(
            generation_strategy.gen_single_trial(experiment=experiment)
        )
        generation_strategy.gen_single_trial(experiment, data=get_branin_data())
        self.assertEqual(len(generation_strategy._generator_runs), 2)
        save_experiment(experiment)
        save_generation_strategy(generation_strategy=generation_strategy)

        new_generation_strategy = load_generation_strategy_by_experiment_name(
            experiment_name=experiment.name,
            reduced_state=True,
            experiment=experiment,
        )
        self.assertEqual(len(new_generation_strategy._generator_runs), 2)
        # Experiment should be the exact same object passed into `load_...`
        self.assertTrue(new_generation_strategy.experiment is experiment)
        # Generation strategies should not be equal, since its generator run #0
        # should be missing model state (and #1 should have it).
        self.assertNotEqual(new_generation_strategy, generation_strategy)
        generation_strategy._generator_runs[0]._generator_kwargs = None
        generation_strategy._generator_runs[0]._adapter_kwargs = None
        generation_strategy._generator_runs[0]._gen_metadata = None
        generation_strategy._generator_runs[0]._generator_state_after_gen = None
        generation_strategy._generator_runs[0]._search_space = None
        generation_strategy._generator_runs[0]._optimization_config = None
        generation_strategy._generator_runs[1]._search_space = None
        generation_strategy._generator_runs[1]._optimization_config = None
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        # Now the generation strategies should be equal.
        self.assertEqual(new_generation_strategy, generation_strategy)
        # Model should be successfully restored in generation strategy even with
        # the reduced state.
        self.assertEqual(
            new_generation_strategy._nodes[0].generator_specs[0].generator_enum,
            Generators.SOBOL,
        )
        self.assertEqual(len(new_generation_strategy._generator_runs), 2)
        self.assertEqual(
            none_throws(new_generation_strategy._experiment)._name, experiment._name
        )
        experiment.new_trial(
            new_generation_strategy.gen_single_trial(experiment=experiment)
        )

    def test_encode_decode_generation_strategy_reduced_state_load_experiment(
        self,
    ) -> None:
        """Try restoring the generation strategy using the experiment its
        attached to, not passing the experiment object (it should then be loaded
        as part of generation strategy loading).
        """
        generation_strategy = get_generation_strategy()
        experiment = get_branin_experiment()
        experiment.new_trial(
            generation_strategy.gen_single_trial(experiment=experiment)
        )
        generation_strategy.gen_single_trial(experiment, data=get_branin_data())
        self.assertEqual(len(generation_strategy._generator_runs), 2)
        save_experiment(experiment)
        save_generation_strategy(generation_strategy=generation_strategy)

        new_generation_strategy = load_generation_strategy_by_experiment_name(
            experiment_name=experiment.name,
            reduced_state=True,
        )
        # Experiment should not be exactly the same object, since it was reloaded
        # from DB and decoded.
        self.assertFalse(new_generation_strategy.experiment is experiment)
        # Generation strategies should not be equal, since only the original one
        # has model state on generator run #0, not the reloaded one.
        self.assertNotEqual(generation_strategy, new_generation_strategy)
        self.assertNotEqual(
            generation_strategy._generator_runs[0],
            new_generation_strategy._generator_runs[0],
        )
        # Experiment should not be equal, since it would be loaded with reduced
        # state along with the generation strategy.
        self.assertNotEqual(new_generation_strategy.experiment, experiment)
        # Adjust experiment and GS to reduced state.
        experiment.trials.get(0).generator_run._generator_kwargs = None
        experiment.trials.get(0).generator_run._adapter_kwargs = None
        experiment.trials.get(0).generator_run._gen_metadata = None
        experiment.trials.get(0).generator_run._generator_state_after_gen = None
        experiment.trials.get(0).generator_run._search_space = None
        experiment.trials.get(0).generator_run._optimization_config = None
        generation_strategy._generator_runs[0]._generator_kwargs = None
        generation_strategy._generator_runs[0]._adapter_kwargs = None
        generation_strategy._generator_runs[0]._gen_metadata = None
        generation_strategy._generator_runs[0]._generator_state_after_gen = None
        generation_strategy._generator_runs[0]._search_space = None
        generation_strategy._generator_runs[0]._optimization_config = None
        generation_strategy._generator_runs[1]._search_space = None
        generation_strategy._generator_runs[1]._optimization_config = None
        # Now experiment on generation strategy should be equal to the original
        # experiment with reduced state.
        self.assertEqual(new_generation_strategy.experiment, experiment)
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(new_generation_strategy, generation_strategy)
        # Model should be successfully restored in generation strategy even with
        # the reduced state.
        self.assertEqual(
            new_generation_strategy._nodes[0].generator_specs[0].generator_enum,
            Generators.SOBOL,
        )
        self.assertEqual(len(new_generation_strategy._generator_runs), 2)
        self.assertEqual(
            none_throws(new_generation_strategy._experiment)._name, experiment._name
        )
        experiment.new_trial(
            new_generation_strategy.gen_single_trial(experiment=experiment)
        )

    def test_update_generation_strategy(self) -> None:
        generation_strategy = get_generation_strategy()
        save_generation_strategy(generation_strategy=generation_strategy)

        experiment = get_branin_experiment()
        save_experiment(experiment)

        # add generator run, save, reload
        experiment.new_trial(
            generator_run=generation_strategy.gen_single_trial(experiment)
        )
        save_generation_strategy(generation_strategy=generation_strategy)
        loaded_generation_strategy = load_generation_strategy_by_experiment_name(
            experiment_name=experiment.name
        )
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(generation_strategy, loaded_generation_strategy)

        # add another generator run, save, reload
        experiment.new_trial(
            generator_run=generation_strategy.gen_single_trial(
                experiment, data=get_branin_data()
            )
        )
        save_generation_strategy(generation_strategy=generation_strategy)
        save_experiment(experiment)
        loaded_generation_strategy = load_generation_strategy_by_experiment_name(
            experiment_name=experiment.name
        )
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(generation_strategy, loaded_generation_strategy)

        # make sure that we can update the experiment too
        experiment.description = "foobar"
        save_experiment(experiment)
        loaded_generation_strategy = load_generation_strategy_by_experiment_name(
            experiment_name=experiment.name
        )
        self.assertEqual(generation_strategy, loaded_generation_strategy)
        self.assertIsNotNone(loaded_generation_strategy._experiment)
        self.assertEqual(
            none_throws(generation_strategy._experiment).description,
            experiment.description,
        )
        self.assertEqual(
            none_throws(generation_strategy._experiment).description,
            none_throws(loaded_generation_strategy._experiment).description,
        )

    def test_generator_run_gen_metadata(self) -> None:
        gen_metadata = {"hello": "world"}
        gr = GeneratorRun(arms=[], gen_metadata=gen_metadata)
        generator_run_sqa = self.encoder.generator_run_to_sqa(gr)
        decoded_gr = self.decoder.generator_run_from_sqa(
            generator_run_sqa, False, False
        )
        self.assertEqual(decoded_gr.gen_metadata, gen_metadata)

    def test_update_generation_strategy_incrementally(self) -> None:
        experiment = get_branin_experiment()
        generation_strategy = choose_generation_strategy(
            GenerationStrategyDispatchStruct()
        )
        save_experiment(experiment=experiment)
        save_generation_strategy(generation_strategy=generation_strategy)

        # add generator runs, save, reload
        generator_runs = []
        for _ in range(7):
            gr = generation_strategy.gen_single_trial(experiment)
            generator_runs.append(gr)
            trial = experiment.new_trial(generator_run=gr).mark_running(
                no_runner_required=True
            )
            experiment.fetch_data()  # Fetch `branin` metric data and attach it.
            trial.mark_completed()

        experiment.fetch_data()
        save_experiment(experiment=experiment)
        update_generation_strategy(
            generation_strategy=generation_strategy,
            generator_runs=generator_runs,
        )
        loaded_generation_strategy = load_generation_strategy_by_experiment_name(
            experiment_name=experiment.name
        )
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(generation_strategy, loaded_generation_strategy)

        # add even more generator runs, save using batch_size, reload
        generator_runs = []
        for _ in range(7):
            gr = generation_strategy.gen_single_trial(experiment)
            generator_runs.append(gr)
            trial = experiment.new_trial(generator_run=gr).mark_running(
                no_runner_required=True
            )
            experiment.fetch_data()  # Fetch `branin` metric data and attach it.
            trial.mark_completed()

        save_experiment(experiment=experiment)
        update_generation_strategy(
            generation_strategy=generation_strategy,
            generator_runs=generator_runs,
            batch_size=3,
        )
        loaded_generation_strategy = load_generation_strategy_by_experiment_name(
            experiment_name=experiment.name
        )
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(generation_strategy, loaded_generation_strategy)

    def test_update_generation_strategy_steps(self) -> None:
        experiment = get_branin_experiment()
        generation_strategy = choose_generation_strategy_legacy(experiment.search_space)
        save_experiment(experiment=experiment)
        save_generation_strategy(generation_strategy=generation_strategy)

        # add generator runs, save, reload
        generator_runs = []
        for i in range(7):
            data = get_branin_data() if i > 0 else None
            gr = generation_strategy.gen_single_trial(experiment, data=data)
            generator_runs.append(gr)
            trial = experiment.new_trial(generator_run=gr).mark_running(
                no_runner_required=True
            )
            trial.mark_completed()

        experiment.fetch_data()
        save_experiment(experiment=experiment)
        update_generation_strategy(
            generation_strategy=generation_strategy,
            generator_runs=generator_runs,
        )
        loaded_generation_strategy = load_generation_strategy_by_experiment_name(
            experiment_name=experiment.name
        )
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(generation_strategy, loaded_generation_strategy)

        # add even more generator runs, save using batch_size, reload
        generator_runs = []
        for i in range(7):
            data = get_branin_data() if i > 0 else None
            gr = generation_strategy.gen_single_trial(experiment, data=data)
            generator_runs.append(gr)
            trial = experiment.new_trial(generator_run=gr).mark_running(
                no_runner_required=True
            )
            trial.mark_completed()

        save_experiment(experiment=experiment)
        update_generation_strategy(
            generation_strategy=generation_strategy,
            generator_runs=generator_runs,
            batch_size=3,
        )
        loaded_generation_strategy = load_generation_strategy_by_experiment_name(
            experiment_name=experiment.name
        )
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(generation_strategy, loaded_generation_strategy)

    def test_update_runner(self) -> None:
        experiment = get_branin_experiment()
        with self.assertRaisesRegex(ValueError, ".* must be saved before"):
            update_runner_on_experiment(
                experiment=experiment,
                # pyre-fixme[6]: For 2nd param expected `Runner` but got `None`.
                runner=None,  # This doesn't matter in this case
                encoder=self.encoder,
                decoder=self.decoder,
            )
        # pyre-fixme[16]: Optional type has no attribute `db_id`.
        self.assertIsNone(experiment.runner.db_id)
        self.assertIsNotNone(experiment.runner)
        # pyre-fixme[16]: `Runner` has no attribute `dummy_metadata`.
        self.assertIsNone(experiment.runner.dummy_metadata)
        save_experiment(experiment=experiment)
        old_runner_db_id = experiment.runner.db_id
        self.assertIsNotNone(old_runner_db_id)
        new_runner = get_synthetic_runner()
        # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Dict[str, str]`.
        new_runner.dummy_metadata = {"foo": "bar"}
        self.assertIsNone(new_runner.db_id)
        experiment.runner = new_runner
        update_runner_on_experiment(
            experiment=experiment,
            runner=new_runner,
            encoder=self.encoder,
            decoder=self.decoder,
        )
        self.assertIsNotNone(new_runner.db_id)  # New runner should be added to DB.
        self.assertEqual(experiment.runner.db_id, new_runner.db_id)
        loaded_experiment = load_experiment(experiment_name=experiment.name)
        self.assertEqual(loaded_experiment.runner.db_id, new_runner.db_id)

    def test_experiment_validation(self) -> None:
        exp = get_experiment()
        exp.name = "test1"
        save_experiment(exp)

        exp2 = get_experiment()
        exp2.name = "test2"
        save_experiment(exp2)

        # changing the name of an experiment is not allowed
        exp.name = "new name"
        with self.assertRaisesRegex(ValueError, ".* Changing the name .*"):
            save_experiment(exp)

        # changing the name to an experiment that already exists
        # is also not allowed
        exp.name = "test2"
        with self.assertRaisesRegex(ValueError, ".* database with the name .*"):
            save_experiment(exp)

        # can't use a name that's already been used
        exp3 = get_experiment()
        exp3.name = "test1"
        with self.assertRaisesRegex(ValueError, ".* experiment already exists .*"):
            save_experiment(exp3)

    def test_experiment_save_and_delete(self) -> None:
        for exp in [
            self.experiment,
            get_experiment_with_map_data_type(),
            get_experiment_with_multi_objective(),
            get_experiment_with_scalarized_objective_and_outcome_constraint(),
        ]:
            exp_name = exp.name
            self.assertIsNone(exp.db_id)
            save_experiment(exp)
            log_msg = (
                f"You are deleting {exp_name} and all its associated"
                + " data from the database."
            )
            with self.assertLogs(delete_experiment.__module__, logging.INFO) as logger:
                delete_experiment(exp_name)
                self.assertTrue(
                    any(log_msg in output for output in logger.output),
                    logger.output,
                )
            with self.assertRaises(ObjectNotFoundError):
                load_experiment(exp_name)

    def test_get_immutable_search_space_and_opt_config(self) -> None:
        save_experiment(self.experiment)
        immutable = _get_experiment_immutable_opt_config_and_search_space(
            experiment_name=self.experiment.name, exp_sqa_class=SQAExperiment
        )
        self.assertFalse(immutable)

        self.experiment._properties = {Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF: True}
        save_experiment(self.experiment)

        immutable = _get_experiment_immutable_opt_config_and_search_space(
            experiment_name=self.experiment.name, exp_sqa_class=SQAExperiment
        )
        self.assertTrue(immutable)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument `ax.storage.sqa...
    @patch(
        f"{Decoder.__module__}.Decoder.generator_run_from_sqa",
        side_effect=Decoder(SQAConfig()).generator_run_from_sqa,
    )
    @patch(
        (
            f"{GET_GS_SQA_IMM_FUNC.__module__}."
            "_get_generation_strategy_sqa_immutable_opt_config_and_search_space"
        ),
        side_effect=_get_generation_strategy_sqa_immutable_opt_config_and_search_space,
    )
    @patch(
        (
            f"{_get_experiment_sqa_immutable_opt_config_and_search_space.__module__}."
            "_get_experiment_sqa_immutable_opt_config_and_search_space"
        ),
        side_effect=_get_experiment_sqa_immutable_opt_config_and_search_space,
    )
    def test_immutable_search_space_and_opt_config_loading(
        self,
        _mock_get_exp_sqa_imm_oc_ss,
        _mock_get_gs_sqa_imm_oc_ss,
        _mock_gr_from_sqa,
    ) -> None:
        experiment = get_experiment_with_batch_trial(constrain_search_space=False)
        experiment._properties = {Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF: True}
        save_experiment(experiment)

        loaded_experiment = load_experiment(experiment.name)
        self.assertTrue(loaded_experiment.immutable_search_space_and_opt_config)

        _mock_get_exp_sqa_imm_oc_ss.assert_called_once()
        self.assertTrue(
            _mock_gr_from_sqa.call_args[1].get("immutable_search_space_and_opt_config")
        )
        _mock_gr_from_sqa.reset_mock()

        generation_strategy = get_generation_strategy()
        experiment.new_trial(
            generation_strategy.gen_single_trial(experiment=experiment)
        )

        save_generation_strategy(generation_strategy=generation_strategy)
        load_generation_strategy_by_experiment_name(
            experiment_name=experiment.name, reduced_state=True
        )
        self.assertTrue(
            _mock_gr_from_sqa.call_args.kwargs.get(
                "immutable_search_space_and_opt_config"
            )
        )

    def test_set_immutable_search_space_and_opt_config(self) -> None:
        experiment = get_experiment_with_batch_trial()
        self.assertFalse(experiment.immutable_search_space_and_opt_config)
        save_experiment(experiment)

        experiment._properties[Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF] = True
        update_properties_on_experiment(
            experiment_with_updated_properties=experiment,
        )

        loaded_experiment = load_experiment(experiment.name)
        self.assertTrue(loaded_experiment.immutable_search_space_and_opt_config)

    def test_update_properties_on_trial(self) -> None:
        experiment = get_experiment_with_batch_trial()
        self.assertNotIn("foo", experiment.trials[0]._properties)
        save_experiment(experiment)

        # Add a property to the trial
        experiment.trials[0]._properties["foo"] = "bar"
        update_properties_on_trial(
            trial_with_updated_properties=experiment.trials[0],
        )
        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(loaded_experiment.trials[0]._properties["foo"], "bar")

    def test_update_properties_on_trial_not_saved(self) -> None:
        experiment = get_experiment_with_batch_trial()
        experiment.trials[0]._properties["foo"] = "bar"
        with self.assertRaisesRegex(
            TrialMutationError, "Trial must be saved before being updated."
        ):
            update_properties_on_trial(
                trial_with_updated_properties=experiment.trials[0],
            )

    def test_update_trial_status(self) -> None:
        experiment = get_experiment_with_batch_trial()
        self.assertEqual(experiment.trials[0].status, TrialStatus.CANDIDATE)
        save_experiment(experiment)
        experiment.trials[0].mark_running(no_runner_required=False)

        update_trial_status(
            trial_with_updated_status=experiment.trials[0],
        )
        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(loaded_experiment.trials[0].status, TrialStatus.RUNNING)

    def test_update_trial_status_not_saved(self) -> None:
        experiment = get_experiment_with_batch_trial()
        with self.assertRaisesRegex(
            TrialMutationError, "Trial must be saved before being updated."
        ):
            update_trial_status(
                trial_with_updated_status=experiment.trials[0],
            )

    def test_repeated_arm_storage(self) -> None:
        experiment = get_experiment_with_batch_trial(with_status_quo=True)
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAArm).count(), 4)

        # add repeated arms to new trial, ensuring
        # we create completely new arms in DB for the
        # new trials
        experiment.new_batch_trial(
            generator_run=GeneratorRun(arms=experiment.trials[0].arms),
            should_add_status_quo_arm=True,
        )
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAArm).count(), 8)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_generator_run_validated_fields(self) -> None:
        # Set up an experiment with a generator run that will have modeling-related
        # fields that are not loaded on most generator runs during reduced-stat
        # experiment loading.
        exp = get_branin_experiment()
        gs = get_generation_strategy()
        trial = exp.new_trial(gs.gen_single_trial(experiment=exp))
        for instrumented_attr in GR_LARGE_MODEL_ATTRS:
            python_attr = SQA_COL_TO_GR_ATTR[instrumented_attr.key]
            self.assertIsNotNone(getattr(trial.generator_run, f"_{python_attr}"))

        # Save and reload the experiment, ensure the modeling-related fields were
        # loaded are non-null.
        save_experiment(exp)
        loaded_exp = load_experiment(exp.name)
        # pyre-fixme[16]: Optional type has no attribute `generator_run`.
        loaded_gr = loaded_exp.trials.get(0).generator_run
        for instrumented_attr in GR_LARGE_MODEL_ATTRS:
            python_attr = SQA_COL_TO_GR_ATTR[instrumented_attr.key]
            self.assertIsNotNone(getattr(loaded_gr, f"_{python_attr}"))

        # Set modeling-related fields to `None`.
        for instrumented_attr in GR_LARGE_MODEL_ATTRS:
            python_attr = SQA_COL_TO_GR_ATTR[instrumented_attr.key]
            setattr(loaded_gr, f"_{python_attr}", None)
            self.assertIsNone(getattr(loaded_gr, f"_{python_attr}"))

        # Save and reload the experiment, ensuring that setting the fields to `None`
        # was not propagated to the DB.
        save_experiment(loaded_exp)
        newly_loaded_exp = load_experiment(exp.name)
        newly_loaded_gr = newly_loaded_exp.trials.get(0).generator_run
        for instrumented_attr in GR_LARGE_MODEL_ATTRS:
            python_attr = SQA_COL_TO_GR_ATTR[instrumented_attr.key]
            self.assertIsNotNone(getattr(newly_loaded_gr, f"_{python_attr}"))

    @patch("ax.storage.sqa_store.db.SESSION_FACTORY", None)
    def test_missing_session_factory(self) -> None:
        with self.assertRaises(ValueError):
            get_session()
        with self.assertRaises(ValueError):
            get_engine()

    def test_create_all_tables_exception(self) -> None:
        engine = create_test_engine()
        engine.dialect.name = "mysql"
        engine.dialect.default_schema_name = "ax"
        with self.assertRaises(ValueError):
            create_all_tables(engine)

    def test_analysis_card(self) -> None:
        test_df = pd.DataFrame(
            columns=["a", "b"],
            data=[
                [1, 2],
                [3, 4],
            ],
        )

        base_analysis_card = AnalysisCard(
            name="test_base_analysis_card",
            title="test_title",
            subtitle="test_subtitle",
            df=test_df,
            blob="test blob",
        )
        markdown_analysis_card = MarkdownAnalysisCard(
            name="test_markdown_analysis_card",
            title="test_title",
            subtitle="test_subtitle",
            df=test_df,
            blob="This is some **really cool** markdown",
        )
        plotly_analysis_card = PlotlyAnalysisCard(
            name="test_plotly_analysis_card",
            title="test_title",
            subtitle="test_subtitle",
            df=test_df,
            blob=pio.to_json(go.Figure()),
        )

        # Create two groups which hold the leaf cards
        # Add the same analysis card multiple times to test _unique_id logic
        small_group = AnalysisCardGroup(
            name="small_group",
            title="Small Group",
            subtitle="This is a small group with just a few cards",
            children=[base_analysis_card, markdown_analysis_card, plotly_analysis_card],
        )
        big_group = AnalysisCardGroup(
            name="big_group",
            title="Big Group",
            subtitle="This is a big group with a lot of cards",
            children=[plotly_analysis_card, small_group],
        )

        with self.subTest("test_save_analysis_cards"):
            save_experiment(self.experiment)

            save_analysis_card(
                big_group,
                self.experiment,
            )

        with self.subTest("test_load_analysis_cards"):
            loaded_analysis_cards = load_analysis_cards_by_experiment_name(
                self.experiment.name
            )

            # This should only load the top level group
            self.assertEqual(len(loaded_analysis_cards), 1)
            loaded_big_group = assert_is_instance(
                loaded_analysis_cards[0], AnalysisCardGroup
            )
            self.assertEqual(loaded_big_group.name, big_group.name)
            self.assertEqual(loaded_big_group.title, big_group.title)
            self.assertEqual(loaded_big_group.subtitle, big_group.subtitle)

            loaded_big_group_plotly = assert_is_instance(
                loaded_big_group.children[0], PlotlyAnalysisCard
            )
            self.assertEqual(loaded_big_group_plotly.name, plotly_analysis_card.name)
            self.assertEqual(loaded_big_group_plotly.blob, plotly_analysis_card.blob)

            loaded_small_group = assert_is_instance(
                loaded_big_group.children[1], AnalysisCardGroup
            )
            self.assertEqual(loaded_small_group.name, small_group.name)
            self.assertEqual(loaded_small_group.title, small_group.title)
            self.assertEqual(loaded_small_group.subtitle, small_group.subtitle)

            loaded_base = assert_is_instance(
                loaded_small_group.children[0], AnalysisCard
            )
            self.assertEqual(loaded_base.name, base_analysis_card.name)
            self.assertEqual(loaded_base.blob, base_analysis_card.blob)

            loaded_markdown = assert_is_instance(
                loaded_small_group.children[1], MarkdownAnalysisCard
            )
            self.assertEqual(loaded_markdown.name, markdown_analysis_card.name)
            self.assertEqual(loaded_markdown.blob, markdown_analysis_card.blob)

            loaded_small_group_plotly = assert_is_instance(
                loaded_small_group.children[2], PlotlyAnalysisCard
            )
            self.assertEqual(loaded_small_group_plotly.name, plotly_analysis_card.name)
            self.assertEqual(loaded_small_group_plotly.blob, plotly_analysis_card.blob)

    def test_delete_generation_strategy(self) -> None:
        # GIVEN an experiment with a generation strategy
        experiment = get_branin_experiment()
        generation_strategy = choose_generation_strategy_legacy(experiment.search_space)
        generation_strategy.experiment = experiment
        save_experiment(experiment)
        save_generation_strategy(generation_strategy=generation_strategy)

        # AND GIVEN another experiment with a generation strategy
        experiment2 = get_branin_experiment()
        experiment2.name = "experiment2"
        generation_strategy2 = choose_generation_strategy_legacy(
            experiment2.search_space
        )
        generation_strategy2.experiment = experiment2
        save_experiment(experiment2)
        save_generation_strategy(generation_strategy=generation_strategy2)

        # WHEN I delete the generation strategy
        delete_generation_strategy(exp_name=experiment.name, max_gs_to_delete=2)

        # THEN the generation strategy is deleted
        with self.assertRaises(ObjectNotFoundError):
            load_generation_strategy_by_experiment_name(experiment.name)

        # AND the other generation strategy is still there
        loaded_generation_strategy = load_generation_strategy_by_experiment_name(
            experiment2.name
        )
        # Full GS fails the equality check
        self.assertEqual(str(generation_strategy2), str(loaded_generation_strategy))

    def test_delete_generation_strategy_max_gs_to_delete(self) -> None:
        # GIVEN an experiment with a generation strategy
        experiment = get_branin_experiment()
        generation_strategy = choose_generation_strategy_legacy(experiment.search_space)
        generation_strategy.experiment = experiment
        save_experiment(experiment)
        save_generation_strategy(generation_strategy=generation_strategy)

        # WHEN I delete the generation strategy with max_gs_to_delete=0
        with self.assertRaisesRegex(
            ValueError,
            "Found 1 generation strategies",
        ):
            delete_generation_strategy(exp_name=experiment.name, max_gs_to_delete=0)

        # THEN the generation strategy is not deleted
        loaded_generation_strategy = load_generation_strategy_by_experiment_name(
            experiment.name
        )
        # Full GS fails the equality check
        self.assertEqual(str(generation_strategy), str(loaded_generation_strategy))

    def test_load_latest_generation_strategy_when_multiple_exist(self) -> None:
        experiment = get_branin_experiment()
        gs1 = choose_generation_strategy_legacy(experiment.search_space)
        gs1.experiment = experiment
        save_experiment(experiment)
        save_generation_strategy(generation_strategy=gs1)
        self.assertEqual(
            gs1.db_id,
            load_generation_strategy_by_experiment_name(experiment.name).db_id,
        )

        # create a second generation strategy for the experiment
        gs2 = choose_generation_strategy_legacy(experiment.search_space)
        gs2._name = "second_gs"
        gs2.experiment = experiment
        save_generation_strategy(generation_strategy=gs2)

        # check that the latest generation stragey is loaded
        with self.assertLogs(
            "ax.storage.sqa_store.load", level=logging.WARNING
        ) as logs:
            loaded_gs = load_generation_strategy_by_experiment_name(experiment.name)
            self.assertEqual(loaded_gs.db_id, gs2.db_id)
            self.assertEqual(loaded_gs.name, gs2.name)
            self.assertTrue(
                any("Found 2 generation strategies" in log for log in logs.output)
            )

    def test_query_historical_experiments_given_parameters(self) -> None:
        # This test validates the query behavior for historical experiments.
        config = SQAConfig(experiment_type_enum=MockExperimentTypeEnum)

        with self.subTest("returns_empty_when_no_matching_experiments"):
            # Query with empty experiment_types list should return empty
            result = _query_historical_experiments_given_parameters(
                parameter_names=["w", "x"],
                experiment_types=[],
                config=config,
            )
            self.assertEqual(result, {})

        with self.subTest("returns_empty_with_empty_parameter_names"):
            # Query with empty parameter names should return empty dict
            result = _query_historical_experiments_given_parameters(
                parameter_names=[],
                experiment_types=["TEST"],
                config=config,
            )
            self.assertEqual(result, {})

        # Integration test: save experiment with data, then query for it
        with self.subTest("returns_experiments_with_matching_parameters"):
            # Create and save an experiment with data (required by query)
            experiment = get_experiment_with_batch_trial()
            experiment.name = "exp_for_historical_query"
            experiment.experiment_type = "TEST"
            experiment.is_test = False  # Query filters out is_test=True
            # Attach data to the experiment (required for the query join)
            trial = experiment.trials[0]
            experiment.attach_data(get_data(trial_index=trial.index))
            save_experiment(experiment, config=config)

            # Verify that the experiment was saved correctly by loading it
            loaded_exp = load_experiment(experiment.name, config=config)
            self.assertEqual(loaded_exp.experiment_type, "TEST")
            self.assertFalse(loaded_exp.is_test)

            # Execute: Query for experiments with matching parameters
            # The experiment has parameters: w, x, y, z (from get_search_space)
            result = _query_historical_experiments_given_parameters(
                parameter_names=["w", "x"],
                experiment_types=["TEST"],
                config=config,
            )

            # Assert: Should find the experiment with the matching parameters
            self.assertIn(experiment.name, result)
            returned_ss = result[experiment.name]
            self.assertIsNotNone(returned_ss)
            # The returned search space should contain w and x
            self.assertIn("w", none_throws(returned_ss).parameters)
            self.assertIn("x", none_throws(returned_ss).parameters)

    def test_identify_transferable_experiments(
        self,
    ) -> None:
        with self.subTest("returns_empty_when_no_experiments"):
            config = SQAConfig(experiment_type_enum=MockExperimentTypeEnum)

            # No experiments are saved, so should return empty
            search_space = get_search_space()
            result = identify_transferable_experiments(
                search_space=search_space,
                experiment_types=["TEST"],
                overlap_threshold=0.0,
                max_num_exps=10,
                config=config,
            )
            self.assertEqual(result, {})

        with self.subTest("returns_transferable_experiments_with_overlap"):
            config = SQAConfig(experiment_type_enum=MockExperimentTypeEnum)

            # Create and save a source experiment with data
            # Use get_experiment_with_batch_trial which has search space with w, x, y, z
            source_experiment = get_experiment_with_batch_trial()
            source_experiment.name = "source_exp_for_transfer"
            source_experiment.experiment_type = "TEST"
            source_experiment.is_test = False
            trial = source_experiment.trials[0]
            source_experiment.attach_data(get_data(trial_index=trial.index))
            save_experiment(source_experiment, config=config)

            # Execute: Find transferable experiments for a target search space
            # that only has RangeParameters w and x (overlapping with source)
            # Note: We use only RangeParameters to avoid incompatibility issues
            # with Choice/Fixed parameters that have different values
            from ax.core.search_space import SearchSpace

            target_search_space = SearchSpace(
                parameters=[
                    get_range_parameter(),  # w
                    get_range_parameter2(),  # x
                ]
            )
            result = identify_transferable_experiments(
                search_space=target_search_space,
                experiment_types=["TEST"],
                overlap_threshold=0.0,
                max_num_exps=10,
                config=config,
            )

            # Assert: Should find the source experiment
            self.assertIn("source_exp_for_transfer", result)
            metadata = result["source_exp_for_transfer"]
            # Should have overlap_parameters populated
            self.assertIsNotNone(metadata.overlap_parameters)
            # Both range parameters should overlap (w, x)
            self.assertEqual(len(none_throws(metadata.overlap_parameters)), 2)

        with self.subTest("filters_by_overlap_threshold"):
            config = SQAConfig(experiment_type_enum=MockExperimentTypeEnum)

            # Create and save a source experiment with parameters w, x, y, z
            source_experiment = get_experiment_with_batch_trial()
            source_experiment.name = "exp_with_partial_overlap"
            source_experiment.experiment_type = "TEST"
            source_experiment.is_test = False
            trial = source_experiment.trials[0]
            source_experiment.attach_data(get_data(trial_index=trial.index))
            save_experiment(source_experiment, config=config)

            # Target has 6 range parameters: w, x overlap with source (33% overlap)
            from ax.core.search_space import SearchSpace

            target_search_space = SearchSpace(
                parameters=[
                    get_range_parameter(),  # w - overlaps
                    get_range_parameter2(),  # x - overlaps
                    RangeParameter(
                        name="extra1",
                        parameter_type=ParameterType.FLOAT,
                        lower=0,
                        upper=10,
                    ),
                    RangeParameter(
                        name="extra2",
                        parameter_type=ParameterType.FLOAT,
                        lower=0,
                        upper=10,
                    ),
                    RangeParameter(
                        name="extra3",
                        parameter_type=ParameterType.FLOAT,
                        lower=0,
                        upper=10,
                    ),
                    RangeParameter(
                        name="extra4",
                        parameter_type=ParameterType.FLOAT,
                        lower=0,
                        upper=10,
                    ),
                ]
            )

            # Execute with threshold of 0.2 (20%) - should include experiment
            # since overlap is ~33% which exceeds 20%
            result_above = identify_transferable_experiments(
                search_space=target_search_space,
                experiment_types=["TEST"],
                overlap_threshold=0.2,  # Threshold below the ~33% overlap
                max_num_exps=10,
                config=config,
            )

            # Assert: Should find experiment
            self.assertIn("exp_with_partial_overlap", result_above)
            # Verify overlap metadata is correct
            metadata = result_above["exp_with_partial_overlap"]
            self.assertIsNotNone(metadata.overlap_parameters)
            # Should have 2 overlapping parameters (w and x)
            self.assertEqual(len(none_throws(metadata.overlap_parameters)), 2)

            # Execute with threshold of 0.5 (50%) - should exclude experiment
            # since overlap is only ~33%
            result_below = identify_transferable_experiments(
                search_space=target_search_space,
                experiment_types=["TEST"],
                overlap_threshold=0.5,  # Threshold above the ~33% overlap
                max_num_exps=10,
                config=config,
            )

            # Assert: Should NOT find experiment due to threshold
            self.assertNotIn("exp_with_partial_overlap", result_below)

        with self.subTest("respects_max_num_exps"):
            config = SQAConfig(experiment_type_enum=MockExperimentTypeEnum)

            # Create multiple experiments with the same type and matching params
            for i in range(3):
                exp = get_experiment_with_batch_trial()
                exp.name = f"max_exp_{i}"
                exp.experiment_type = "TEST"
                exp.is_test = False
                trial = exp.trials[0]
                exp.attach_data(get_data(trial_index=trial.index))
                save_experiment(exp, config=config)

            # Use a target search space with only RangeParameters
            from ax.core.search_space import SearchSpace

            target_search_space = SearchSpace(
                parameters=[
                    get_range_parameter(),  # w
                    get_range_parameter2(),  # x
                ]
            )

            # Execute with max_num_exps=1
            result = identify_transferable_experiments(
                search_space=target_search_space,
                experiment_types=["TEST"],
                overlap_threshold=0.0,
                max_num_exps=1,
                config=config,
            )

            # Assert: Should only return 1 experiment
            self.assertEqual(len(result), 1)

    def test_load_candidate_source_auxiliary_experiments(self) -> None:
        with self.subTest("raises_when_no_experiment_type"):
            experiment = get_branin_experiment()
            experiment._experiment_type = None

            with self.assertRaisesRegex(
                MisconfiguredExperiment,
                "Cannot identify transferable experiments because the target "
                "experiment does not have an experiment type",
            ):
                load_candidate_source_auxiliary_experiments(
                    target_experiment=experiment,
                    purpose=AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT,
                )

        with self.subTest("returns_empty_for_unsupported_purpose"):
            experiment = get_branin_experiment()
            experiment._experiment_type = "TEST"

            with self.assertRaisesRegex(
                NotImplementedError,
                "Loading candidate source auxiliary experiments for purpose "
                f"{AuxiliaryExperimentPurpose.PE_EXPERIMENT} is not yet implemented.",
            ):
                result = load_candidate_source_auxiliary_experiments(
                    target_experiment=experiment,
                    purpose=AuxiliaryExperimentPurpose.PE_EXPERIMENT,
                )

        with self.subTest("returns_candidate_experiments_for_transfer_learning"):
            config = SQAConfig(experiment_type_enum=MockExperimentTypeEnum)

            # Create and save source experiments with data attached
            # (required by the query which joins on SQAData)
            for i in range(2):
                source_exp = get_experiment_with_batch_trial()
                source_exp.name = f"candidate_source_exp_{i}"
                source_exp.experiment_type = "TEST"
                source_exp.is_test = False
                trial = source_exp.trials[0]
                source_exp.attach_data(get_data(trial_index=trial.index))
                save_experiment(source_exp, config=config)

            # Create target experiment with same experiment type
            # Use only RangeParameters to ensure overlap with source experiments
            from ax.core.search_space import SearchSpace

            target_search_space = SearchSpace(
                parameters=[
                    get_range_parameter(),  # w
                    get_range_parameter2(),  # x
                ]
            )
            target_experiment = Experiment(
                name="target_exp_for_candidates",
                search_space=target_search_space,
                experiment_type="TEST",
            )

            # Execute: Load candidate source auxiliary experiments
            result = load_candidate_source_auxiliary_experiments(
                target_experiment=target_experiment,
                purpose=AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT,
                config=config,
            )

            # Assert: Should find the source experiments
            self.assertIn("candidate_source_exp_0", result)
            self.assertIn("candidate_source_exp_1", result)
            # Verify the metadata contains overlap information
            metadata_0 = result["candidate_source_exp_0"]
            self.assertIsInstance(metadata_0, TransferLearningMetadata)
            # Cast to TransferLearningMetadata to access overlap_parameters
            tl_metadata_0 = cast(TransferLearningMetadata, metadata_0)
            self.assertIsNotNone(tl_metadata_0.overlap_parameters)
            # Should have 2 overlapping parameters (w and x)
            self.assertEqual(len(none_throws(tl_metadata_0.overlap_parameters)), 2)

    def test_transition_criterion_deserialize_with_extra_fields(self) -> None:
        """Test that deserialization gracefully handles extra/unknown fields
        ie this validates that backwards compatibility is maintained"""
        # Simulate old serialized format with extra fields that no longer exist
        old_format_json = {
            "threshold": 5,
            "only_in_statuses": [{"__type": "TrialStatus", "name": "RUNNING"}],
            "not_in_statuses": None,
            "transition_to": "test_node",
            "block_gen_if_met": True,
            "block_transition_if_unmet": False,
            "use_all_trials_in_exp": False,
            "continue_trial_generation": False,
            "some_deprecated_field": "should_be_ignored",
        }

        # Should not raise, extra field should be ignored
        criterion = assert_is_instance(
            transition_criterion_from_json(
                transition_criterion_class=MaxGenerationParallelism,
                object_json=old_format_json,
                decoder_registry=CORE_DECODER_REGISTRY,
                class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
            ),
            MaxGenerationParallelism,
        )
        self.assertEqual(criterion.threshold, 5)
        self.assertEqual(criterion.transition_to, "test_node")

    def test_gen_node_deserialize_with_tc_transition_to_none(
        self,
    ) -> None:
        """Test backwards compatibility when loading a MaxGenerationParallelism
        that was stored with transition_to=None
        """
        old_format_node_json = {
            "__type": "GenerationNode",
            "name": "test_node",
            "generator_specs": [
                {
                    "__type": "GeneratorSpec",
                    "generator_enum": {"__type": "Generators", "name": "SOBOL"},
                    "generator_kwargs": {},
                    "generator_gen_kwargs": {},
                }
            ],
            "transition_criteria": [
                {
                    "__type": "MaxGenerationParallelism",
                    "threshold": 3,
                    "only_in_statuses": [{"__type": "TrialStatus", "name": "RUNNING"}],
                    "transition_to": None,  # Old default
                }
            ],
        }

        node = generation_node_from_json(
            generation_node_json=old_format_node_json,
            decoder_registry=CORE_DECODER_REGISTRY,
            class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
        )
        self.assertEqual(node.name, "test_node")
        self.assertEqual(len(node.transition_criteria), 1)
        criterion = assert_is_instance(
            node.transition_criteria[0],
            MaxGenerationParallelism,
        )
        self.assertEqual(criterion.threshold, 3)
        # transition_to should now be set to the node name (pointing to itself)
        self.assertEqual(criterion.transition_to, "test_node")
