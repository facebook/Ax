#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from datetime import datetime
from decimal import Decimal
from enum import Enum, unique
from logging import Logger
from typing import Any, Callable, TypeVar
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
from ax.analysis.analysis import AnalysisCard, AnalysisCardLevel
from ax.analysis.markdown.markdown_analysis import MarkdownAnalysisCard
from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
from ax.core.arm import Arm
from ax.core.auxiliary import AuxiliaryExperiment, AuxiliaryExperimentPurpose
from ax.core.base_trial import TrialStatus
from ax.core.batch_trial import LifecycleStage
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.runner import Runner
from ax.core.types import ComparisonOp
from ax.exceptions.core import ObjectNotFoundError
from ax.exceptions.storage import JSONDecodeError, SQADecodeError, SQAEncodeError
from ax.metrics.branin import BraninMetric
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate, SurrogateSpec
from ax.runners.synthetic import SyntheticRunner
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
    session_scope,
)
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.delete import delete_experiment, delete_generation_strategy
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.load import (
    _get_experiment_immutable_opt_config_and_search_space,
    _get_experiment_sqa_immutable_opt_config_and_search_space,
    _get_generation_strategy_sqa_immutable_opt_config_and_search_space,
    load_analysis_cards_by_experiment_name,
    load_experiment,
    load_generation_strategy_by_experiment_name,
    load_generation_strategy_by_id,
)
from ax.storage.sqa_store.reduced_state import GR_LARGE_MODEL_ATTRS
from ax.storage.sqa_store.save import (
    save_analysis_cards,
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


@unique
class TestAuxiliaryExperimentPurpose(AuxiliaryExperimentPurpose):
    MyAuxExpPurpose = "my_auxiliary_experiment_purpose"


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
        self.config.auxiliary_experiment_purpose_enum = TestAuxiliaryExperimentPurpose

    def test_CreationOfTestDB(self) -> None:
        init_test_engine_and_session_factory(tier_or_path=":memory:", force_init=True)
        engine = get_engine()
        self.assertIsNotNone(engine)

    def test_DBConnectionWithoutForceInit(self) -> None:
        init_test_engine_and_session_factory(tier_or_path=":memory:")

    def test_ConnectionToDBWithURL(self) -> None:
        init_engine_and_session_factory(url="sqlite://", force_init=True)

    def test_ConnectionToDBWithCreator(self) -> None:
        def MockDBAPI() -> MagicMock:
            connection = Mock()

            # pyre-fixme[53]: Captured variable `connection` is not annotated.
            def connect(*args: Any, **kwargs: Any) -> Mock:
                return connection

            return MagicMock(connect=Mock(side_effect=connect))

        mocked_dbapi = MockDBAPI()
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

    def test_GeneratorRunTypeValidation(self) -> None:
        experiment = get_experiment_with_batch_trial()
        # pyre-fixme[16]: `BaseTrial` has no attribute `generator_run_structs`.
        generator_run = experiment.trials[0].generator_run_structs[0].generator_run
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

    @mock_botorch_optimize
    def test_SaveExperimentWithSurrogateAsModelKwarg(self) -> None:
        experiment = get_branin_experiment(
            with_batch=True, num_batch_trial=1, with_completed_batch=True
        )
        model = Models.BOTORCH_MODULAR(
            experiment=experiment,
            data=experiment.lookup_data(),
            surrogate=Surrogate(surrogate_spec=SurrogateSpec()),
        )
        experiment.new_batch_trial(generator_run=model.gen(1))
        # ensure we can save the experiment
        save_experiment(experiment)

    def test_ExperimentSaveAndLoad(self) -> None:
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
                self.config.auxiliary_experiment_purpose_enum.MyAuxExpPurpose: [
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

    def test_saving_and_loading_experiment_with_cross_referencing_aux_exp(self) -> None:
        exp1_name = "test_aux_exp_in_SQAStoreTest1"
        exp2_name = "test_aux_exp_in_SQAStoreTest2"
        # pyre-ignore[16]: `AuxiliaryExperimentPurpose` has no attribute
        exp_purpose = self.config.auxiliary_experiment_purpose_enum.MyAuxExpPurpose

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
        class TestExperimentTypeEnum(Enum):
            TEST = 0

        self.experiment.experiment_type = "TEST"
        save_experiment(
            self.experiment,
            config=SQAConfig(experiment_type_enum=TestExperimentTypeEnum),
        )
        self.assertIsNotNone(self.experiment.db_id)

    def test_saving_an_experiment_with_type_errors_with_missing_enum_value(
        self,
    ) -> None:
        class TestExperimentTypeEnum(Enum):
            NOT_TEST = 0

        self.experiment.experiment_type = "TEST"
        with self.assertRaises(SQAEncodeError):
            save_experiment(
                self.experiment,
                config=SQAConfig(experiment_type_enum=TestExperimentTypeEnum),
            )

    def test_LoadExperimentTrialsInBatches(self) -> None:
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
    def test_LoadExperimentSkipMetricsAndRunners(self) -> None:
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
            for multi_objective in [True, False]:
                custom_metric_names = ["custom_test_metric"]
                experiment = get_experiment_with_custom_runner_and_metric(
                    constrain_search_space=False,
                    immutable=immutable,
                    multi_objective=multi_objective,
                    num_trials=1,
                )
                if multi_objective:
                    custom_metric_names.extend(["m1", "m3"])
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
                    self.assertTrue(metric_name in loaded_experiment.metrics)
                    self.assertEqual(
                        loaded_experiment.metrics["custom_test_metric"].__class__,
                        Metric,
                    )
                self.assertEqual(len(loaded_experiment.trials), 1)
                trial = loaded_experiment.trials[0]
                self.assertIs(trial.runner, None)
                delete_experiment(exp_name=experiment.name)
                # check generator runs
                gr = trial.generator_runs[0]
                if multi_objective and not immutable:
                    objectives = assert_is_instance(
                        none_throws(gr.optimization_config).objective, MultiObjective
                    ).objectives
                    for i, objective in enumerate(objectives):
                        metric = objective.metric
                        self.assertEqual(metric.name, f"m{1 + 2 * i}")
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
    def test_ExperimentSaveAndLoadReducedState(
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
            for i in loaded_experiment.trials.keys():
                loaded_experiment.trials[i]._runner = exp.trials[i].runner
            self.assertEqual(loaded_experiment, exp)
            # Make sure decoder function was called with `reduced_state=True`.
            self.assertTrue(_mock_exp_from_sqa.call_args[1].get("reduced_state"))
            self.assertTrue(_mock_trial_from_sqa.call_args[1].get("reduced_state"))
            self.assertTrue(_mock_gr_from_sqa.call_args[1].get("reduced_state"))
            _mock_exp_from_sqa.reset_mock()

            # 3. Try case with model state and search space + opt.config on a
            # generator run in the experiment.
            gr = Models.SOBOL(experiment=exp).gen(1)
            # Expecting model kwargs to have 6 fields (seed, deduplicate, init_position,
            # scramble, generated_points, fallback_to_sample_polytope)
            # and the rest of model-state info on generator run to have values too.
            mkw = gr._model_kwargs
            self.assertIsNotNone(mkw)
            self.assertEqual(len(mkw), 6)
            bkw = gr._bridge_kwargs
            self.assertIsNotNone(bkw)
            self.assertEqual(len(bkw), 9)
            # This has seed, generated points and init position.
            ms = gr._model_state_after_gen
            self.assertIsNotNone(ms)
            self.assertEqual(len(ms), 3)
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
            loaded_experiment._trials[0]._runner = exp._trials[0]._runner
            self.assertTrue(_mock_exp_from_sqa.call_args[1].get("reduced_state"))
            self.assertTrue(_mock_trial_from_sqa.call_args[1].get("reduced_state"))
            # 2 generator runs from trial #0 + 1 from trial #1.
            self.assertTrue(_mock_gr_from_sqa.call_args[1].get("reduced_state"))
            self.assertNotEqual(loaded_experiment, exp)
            # Remove all fields that are not part of the reduced state and
            # check that everything else is equal as expected.
            exp.trials.get(1).generator_run._model_kwargs = None
            exp.trials.get(1).generator_run._bridge_kwargs = None
            exp.trials.get(1).generator_run._gen_metadata = None
            exp.trials.get(1).generator_run._model_state_after_gen = None
            exp.trials.get(1).generator_run._search_space = None
            exp.trials.get(1).generator_run._optimization_config = None
            self.assertEqual(loaded_experiment, exp)
            delete_experiment(exp_name=exp.name)

    def test_load_and_save_reduced_state_does_not_lose_abandoned_arms(self) -> None:
        exp = get_experiment_with_batch_trial(constrain_search_space=False)
        exp.trials[0].mark_arm_abandoned(arm_name="0_0", reason="for this test")
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
        self.assertEqual(
            len(reloaded_experiment.trials[0].abandoned_arms),
            1,
        )

    def test_ExperimentSaveAndLoadGRWithOptConfig(self) -> None:
        exp = get_experiment_with_batch_trial(constrain_search_space=False)
        gr = Models.SOBOL(experiment=exp).gen(
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
            with_experiment=True, with_callable_model_kwarg=False
        )
        gs.gen(experiment=gs.experiment)
        gs.gen(experiment=gs.experiment)

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

    def test_MTExperimentSaveAndLoad(self) -> None:
        experiment = get_multi_type_experiment(add_trials=True)
        save_experiment(experiment)
        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(loaded_experiment.default_trial_type, "type1")
        # pyre-fixme[16]: `Experiment` has no attribute `_trial_type_to_runner`.
        self.assertEqual(len(loaded_experiment._trial_type_to_runner), 2)
        # pyre-fixme[16]: `Experiment` has no attribute `metric_to_trial_type`.
        self.assertEqual(loaded_experiment.metric_to_trial_type["m1"], "type1")
        self.assertEqual(loaded_experiment.metric_to_trial_type["m2"], "type2")
        # pyre-fixme[16]: `Experiment` has no attribute `_metric_to_canonical_name`.
        self.assertEqual(loaded_experiment._metric_to_canonical_name["m2"], "m1")
        self.assertEqual(len(loaded_experiment.trials), 2)

    def test_MTExperimentSaveAndLoadSkipRunnersAndMetrics(self) -> None:
        experiment = get_multi_type_experiment(add_trials=True)
        save_experiment(experiment)
        loaded_experiment = load_experiment(
            experiment.name, skip_runners_and_metrics=True
        )
        self.assertEqual(loaded_experiment.default_trial_type, "type1")
        # pyre-fixme[16]: `Experiment` has no attribute `_trial_type_to_runner`.
        self.assertIsNone(loaded_experiment._trial_type_to_runner["type1"])
        self.assertIsNone(loaded_experiment._trial_type_to_runner["type2"])
        # pyre-fixme[16]: `Experiment` has no attribute `metric_to_trial_type`.
        self.assertEqual(loaded_experiment.metric_to_trial_type["m1"], "type1")
        self.assertEqual(loaded_experiment.metric_to_trial_type["m2"], "type2")
        # pyre-fixme[16]: `Experiment` has no attribute `_metric_to_canonical_name`.
        self.assertEqual(loaded_experiment._metric_to_canonical_name["m2"], "m1")
        self.assertEqual(len(loaded_experiment.trials), 2)

    def test_ExperimentNewTrial(self) -> None:
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

    def test_ExperimentNewTrialValidation(self) -> None:
        trial = self.experiment.new_batch_trial()

        with self.assertRaises(ValueError):
            # must save experiment first
            save_or_update_trial(experiment=self.experiment, trial=trial)

    def test_ExperimentUpdateTrial(self) -> None:
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
        self.experiment.attach_data(get_data(trial_index=trial.index))
        save_or_update_trial(experiment=self.experiment, trial=trial)

        loaded_experiment = load_experiment(self.experiment.name)
        self.assertEqual(self.experiment, loaded_experiment)

        # Update a trial by attaching data again
        self.experiment.attach_data(
            get_data(trial_index=trial.index), combine_with_last_data=True
        )
        save_or_update_trial(experiment=self.experiment, trial=trial)

        loaded_experiment = load_experiment(self.experiment.name)
        self.assertEqual(self.experiment, loaded_experiment)

    def test_ExperimentSaveAndUpdateTrials(self) -> None:
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
        generator_run = Models.SOBOL(search_space=exp.search_space).gen(n=1)
        trial = exp.new_trial(generator_run=generator_run)
        exp.attach_data(trial.run().fetch_data())
        save_or_update_trials(
            experiment=exp,
            trials=[trial],
            batch_size=2,
        )
        loaded_experiment = load_experiment(exp.name)
        self.assertEqual(exp, loaded_experiment)

    def test_trial_lifecycle_stage(self) -> None:
        save_experiment(self.experiment)

        existing_trial = self.experiment.trials[0]
        existing_trial.mark_staged()
        existing_trial._lifecycle_stage = LifecycleStage.EXPLORATION_CONCURRENT
        new_trial = self.experiment.new_batch_trial(
            generator_run=get_generator_run(),
            lifecycle_stage=LifecycleStage.ITERATION,
        )
        save_or_update_trials(
            experiment=self.experiment, trials=[existing_trial, new_trial]
        )
        loaded_experiment = load_experiment(self.experiment.name)
        self.assertEqual(
            # pyre-fixme[16]: `BaseTrial` has no attribute `lifecycle_stage`.
            loaded_experiment.trials[existing_trial.index].lifecycle_stage,
            LifecycleStage.EXPLORATION_CONCURRENT,
        )
        self.assertEqual(
            loaded_experiment.trials[new_trial.index].lifecycle_stage,
            LifecycleStage.ITERATION,
        )

    def test_SaveValidation(self) -> None:
        with self.assertRaises(ValueError):
            save_experiment(self.experiment.trials[0])

        experiment = get_experiment_with_batch_trial()
        # pyre-fixme[8]: Attribute has type `str`; used as `None`.
        experiment.name = None
        with self.assertRaises(ValueError):
            save_experiment(experiment)

    def test_EncodeDecode(self) -> None:
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

    def test_EncodeGeneratorRunReducedState(self) -> None:
        exp = get_branin_experiment()
        gs = get_generation_strategy(with_callable_model_kwarg=False)
        gr = gs.gen(experiment=exp)

        for key in [attr.key for attr in GR_LARGE_MODEL_ATTRS]:
            self.assertIsNotNone(getattr(gr, f"_{key}"))

        gr_sqa_reduced_state = self.encoder.generator_run_to_sqa(
            generator_run=gr, weight=None, reduced_state=True
        )

        gr_decoded_reduced_state = self.decoder.generator_run_from_sqa(
            gr_sqa_reduced_state,
            reduced_state=False,
            immutable_search_space_and_opt_config=False,
        )

        for key in [attr.key for attr in GR_LARGE_MODEL_ATTRS]:
            setattr(gr, f"_{key}", None)

        self.assertEqual(gr, gr_decoded_reduced_state)

    def test_load_and_save_generator_run_reduced_state(self) -> None:
        exp = get_branin_experiment()
        gs = get_generation_strategy(with_callable_model_kwarg=False)
        gr = gs.gen(experiment=exp)
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

    def test_ExperimentUpdates(self) -> None:
        experiment = get_experiment_with_batch_trial()
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAExperiment).count(), 1)

        # update experiment
        # (should perform update in place)
        experiment.description = "foobar"
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAExperiment).count(), 1)

        experiment.status_quo = Arm(parameters={"w": 0.0, "x": 1, "y": "y", "z": True})
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAExperiment).count(), 1)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_ExperimentParameterUpdates(self) -> None:
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

    def test_ExperimentParameterConstraintUpdates(self) -> None:
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

    def test_ExperimentObjectiveUpdates(self) -> None:
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

    def test_ExperimentOutcomeConstraintUpdates(self) -> None:
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

    def test_ExperimentObjectiveThresholdUpdates(self) -> None:
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

    def test_FailedLoad(self) -> None:
        with self.assertRaises(ObjectNotFoundError):
            load_experiment("nonexistent_experiment")

    def test_ExperimentTrackingMetricUpdates(self) -> None:
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

    def test_ExperimentRunnerUpdates(self) -> None:
        experiment = get_experiment_with_batch_trial()
        save_experiment(experiment)
        # one runner on the batch
        self.assertEqual(get_session().query(SQARunner).count(), 1)

        # add runner to experiment
        runner = get_synthetic_runner()
        experiment.runner = runner
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQARunner).count(), 2)

        # update runner
        # (should perform update in place)
        runner = get_synthetic_runner()
        # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Dict[str, str]`.
        runner.dummy_metadata = {"foo": "bar"}
        experiment.runner = runner
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQARunner).count(), 2)

        # remove runner
        # (old one should be deleted)
        experiment.runner = None
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQARunner).count(), 1)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_ExperimentTrialUpdates(self) -> None:
        experiment = get_experiment_with_batch_trial()
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQATrial).count(), 1)
        self.assertEqual(get_session().query(SQARunner).count(), 1)

        # add trial
        trial = experiment.new_batch_trial()
        runner = get_synthetic_runner()
        trial.runner = runner
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQATrial).count(), 2)
        self.assertEqual(get_session().query(SQARunner).count(), 2)

        # update trial's runner
        runner.dummy_metadata = "dummy metadata"
        trial.runner = runner
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQATrial).count(), 2)
        self.assertEqual(get_session().query(SQARunner).count(), 2)

        trial.run()
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQATrial).count(), 2)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_ExperimentAbandonedArmUpdates(self) -> None:
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

    def test_ExperimentGeneratorRunUpdates(self) -> None:
        experiment = get_experiment_with_batch_trial()
        save_experiment(experiment)
        # one main generator run, one for the status quo
        self.assertEqual(get_session().query(SQAGeneratorRun).count(), 2)

        # add a arm
        # this will create one wrapper generator run
        # this will also replace the status quo generator run,
        # since the weight of the status quo will have changed
        trial = experiment.trials[0]
        # pyre-fixme[16]: `BaseTrial` has no attribute `add_arm`.
        trial.add_arm(get_arm())
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAGeneratorRun).count(), 3)

        generator_run = get_generator_run()
        # pyre-fixme[16]: `BaseTrial` has no attribute `add_generator_run`.
        trial.add_generator_run(generator_run=generator_run, multiplier=0.5)
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAGeneratorRun).count(), 4)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_ParameterValidation(self) -> None:
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

    def test_ParameterDecodeFailure(self) -> None:
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

    def test_ParameterConstraintValidation(self) -> None:
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

    def test_DecodeOrderParameterConstraintFailure(self) -> None:
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

    def test_DecodeSumParameterConstraintFailure(self) -> None:
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

    def test_MetricValidation(self) -> None:
        sqa_metric = SQAMetric(
            name="foobar",
            intent=MetricIntent.OBJECTIVE,
            metric_type=CORE_METRIC_REGISTRY[BraninMetric],
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

    def test_MetricEncodeFailure(self) -> None:
        metric = get_branin_metric()
        del metric.__dict__["param_names"]
        with self.assertRaises(AttributeError):
            self.encoder.metric_to_sqa(metric)

    def test_MetricDecodeFailure(self) -> None:
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

    def test_RunnerDecodeFailure(self) -> None:
        runner = get_synthetic_runner()
        sqa_runner = self.encoder.runner_to_sqa(runner)
        # pyre-fixme[8]: Attribute has type `int`; used as `str`.
        sqa_runner.runner_type = "foobar"
        with self.assertRaises(SQADecodeError):
            self.decoder.runner_from_sqa(sqa_runner)

    def test_RunnerValidation(self) -> None:
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

    def test_TimestampUpdate(self) -> None:
        self.experiment.trials[0]._time_staged = datetime.now()
        save_experiment(self.experiment)

        # second save should not fail
        save_experiment(self.experiment)

    def test_GetProperties(self) -> None:
        # Extract default value.
        properties = serialize_init_args(obj=Metric(name="foo"))
        self.assertEqual(
            properties, {"name": "foo", "lower_is_better": None, "properties": {}}
        )

        # Extract passed value.
        properties = serialize_init_args(
            obj=Metric(name="foo", lower_is_better=True, properties={"foo": "bar"})
        )
        self.assertEqual(
            properties,
            {"name": "foo", "lower_is_better": True, "properties": {"foo": "bar"}},
        )

    def test_RegistryAdditions(self) -> None:
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

    def test_RegistryBundle(self) -> None:
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

    def test_EncodeDecodeGenerationStrategy(self) -> None:
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
        generation_strategy = get_generation_strategy(with_callable_model_kwarg=False)
        experiment.new_trial(generation_strategy.gen(experiment=experiment))
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
        self.assertIsInstance(new_generation_strategy._steps[0].model, Models)
        self.assertEqual(len(new_generation_strategy._generator_runs), 2)
        self.assertEqual(
            none_throws(new_generation_strategy._experiment)._name, experiment._name
        )

    def test_EncodeDecodeGenerationNodeGSWithAdvancedSettings(self) -> None:
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
            generator_runs=generation_strategy._gen_with_multiple_nodes(
                experiment=experiment
            )
        )
        generation_strategy._gen_with_multiple_nodes(experiment, data=get_branin_data())
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
            new_generation_strategy._nodes[0].model_spec_to_gen_from.model_enum, Models
        )
        self.assertEqual(len(new_generation_strategy._generator_runs), 2)
        self.assertEqual(
            none_throws(new_generation_strategy._experiment)._name, experiment._name
        )

    def test_EncodeDecodeGenerationNodeBasedGenerationStrategy(self) -> None:
        """Test to ensure that GenerationNode based GenerationStrategies are
        able to be encoded/decoded correctly.
        """
        # we don't support callable models for GenNode based strategies
        generation_strategy = get_generation_strategy(
            with_generation_nodes=True, with_callable_model_kwarg=False
        )
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
        generation_strategy = get_generation_strategy(with_generation_nodes=True)
        experiment.new_trial(generation_strategy.gen(experiment=experiment))
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
            new_generation_strategy._nodes[0].model_spec_to_gen_from.model_enum, Models
        )
        self.assertEqual(len(new_generation_strategy._generator_runs), 2)
        self.assertEqual(
            none_throws(new_generation_strategy._experiment)._name, experiment._name
        )

    def test_EncodeDecodeGenerationStrategyReducedState(self) -> None:
        """Try restoring the generation strategy using the experiment its attached to,
        passing the experiment object.
        """
        generation_strategy = get_generation_strategy(with_callable_model_kwarg=False)
        experiment = get_branin_experiment()
        experiment.new_trial(generation_strategy.gen(experiment=experiment))
        generation_strategy.gen(experiment, data=get_branin_data())
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
        generation_strategy._generator_runs[0]._model_kwargs = None
        generation_strategy._generator_runs[0]._bridge_kwargs = None
        generation_strategy._generator_runs[0]._gen_metadata = None
        generation_strategy._generator_runs[0]._model_state_after_gen = None
        generation_strategy._generator_runs[0]._search_space = None
        generation_strategy._generator_runs[0]._optimization_config = None
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        generation_strategy._unset_non_persistent_state_fields()
        # Now the generation strategies should be equal.
        # Reloaded generation strategy will not have attributes associated with fitting
        # the model until after it's used to fit the model or generate candidates, so
        # we unset those attributes here and compare equality of the rest.
        generation_strategy._model = None
        self.assertEqual(new_generation_strategy, generation_strategy)
        # Model should be successfully restored in generation strategy even with
        # the reduced state.
        self.assertIsInstance(new_generation_strategy._steps[0].model, Models)
        self.assertEqual(len(new_generation_strategy._generator_runs), 2)
        self.assertEqual(
            none_throws(new_generation_strategy._experiment)._name, experiment._name
        )
        experiment.new_trial(new_generation_strategy.gen(experiment=experiment))

    def test_EncodeDecodeGenerationStrategyReducedStateLoadExperiment(self) -> None:
        """Try restoring the generation strategy using the experiment its
        attached to, not passing the experiment object (it should then be loaded
        as part of generation strategy loading).
        """
        generation_strategy = get_generation_strategy(with_callable_model_kwarg=False)
        experiment = get_branin_experiment()
        experiment.new_trial(generation_strategy.gen(experiment=experiment))
        generation_strategy.gen(experiment, data=get_branin_data())
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
        experiment.trials.get(0).generator_run._model_kwargs = None
        experiment.trials.get(0).generator_run._bridge_kwargs = None
        experiment.trials.get(0).generator_run._gen_metadata = None
        experiment.trials.get(0).generator_run._model_state_after_gen = None
        experiment.trials.get(0).generator_run._search_space = None
        experiment.trials.get(0).generator_run._optimization_config = None
        generation_strategy._generator_runs[0]._model_kwargs = None
        generation_strategy._generator_runs[0]._bridge_kwargs = None
        generation_strategy._generator_runs[0]._gen_metadata = None
        generation_strategy._generator_runs[0]._model_state_after_gen = None
        generation_strategy._generator_runs[0]._search_space = None
        generation_strategy._generator_runs[0]._optimization_config = None
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
        self.assertIsInstance(new_generation_strategy._steps[0].model, Models)
        self.assertEqual(len(new_generation_strategy._generator_runs), 2)
        self.assertEqual(
            none_throws(new_generation_strategy._experiment)._name, experiment._name
        )
        experiment.new_trial(new_generation_strategy.gen(experiment=experiment))

    def test_UpdateGenerationStrategy(self) -> None:
        generation_strategy = get_generation_strategy(with_callable_model_kwarg=False)
        save_generation_strategy(generation_strategy=generation_strategy)

        experiment = get_branin_experiment()
        save_experiment(experiment)

        # add generator run, save, reload
        experiment.new_trial(generator_run=generation_strategy.gen(experiment))
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
            generator_run=generation_strategy.gen(experiment, data=get_branin_data())
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
        # Reloaded generation strategy will not have attributes associated with fitting
        # the model until after it's used to fit the model or generate candidates, so
        # we unset those attributes here and compare equality of the rest.
        generation_strategy._model = None
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

    def test_GeneratorRunGenMetadata(self) -> None:
        gen_metadata = {"hello": "world"}
        gr = GeneratorRun(arms=[], gen_metadata=gen_metadata)
        generator_run_sqa = self.encoder.generator_run_to_sqa(gr)
        decoded_gr = self.decoder.generator_run_from_sqa(
            generator_run_sqa, False, False
        )
        self.assertEqual(decoded_gr.gen_metadata, gen_metadata)

    def test_UpdateGenerationStrategyIncrementally(self) -> None:
        experiment = get_branin_experiment()
        generation_strategy = choose_generation_strategy(experiment.search_space)
        save_experiment(experiment=experiment)
        save_generation_strategy(generation_strategy=generation_strategy)

        # add generator runs, save, reload
        generator_runs = []
        for i in range(7):
            data = get_branin_data() if i > 0 else None
            gr = generation_strategy.gen(experiment, data=data)
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
            gr = generation_strategy.gen(experiment, data=data)
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

    def test_UpdateRunner(self) -> None:
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

    def test_ExperimentValidation(self) -> None:
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

    def test_ExperimentSaveAndDelete(self) -> None:
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

    def test_GetImmutableSearchSpaceAndOptConfig(self) -> None:
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
    def test_ImmutableSearchSpaceAndOptConfigLoading(
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

        generation_strategy = get_generation_strategy(with_callable_model_kwarg=False)
        experiment.new_trial(generation_strategy.gen(experiment=experiment))

        save_generation_strategy(generation_strategy=generation_strategy)
        load_generation_strategy_by_experiment_name(experiment_name=experiment.name)
        _mock_get_gs_sqa_imm_oc_ss.assert_called_once()
        self.assertTrue(
            _mock_gr_from_sqa.call_args.kwargs.get(
                "immutable_search_space_and_opt_config"
            )
        )

    def test_SetImmutableSearchSpaceAndOptConfig(self) -> None:
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
            ValueError, "Trial must be saved before being updated."
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
            ValueError, "Trial must be saved before being updated."
        ):
            update_trial_status(
                trial_with_updated_status=experiment.trials[0],
            )

    def test_RepeatedArmStorage(self) -> None:
        experiment = get_experiment_with_batch_trial()
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAArm).count(), 4)

        # add repeated arms to new trial, ensuring
        # we create completely new arms in DB for the
        # new trials
        experiment.new_batch_trial(
            generator_run=GeneratorRun(arms=experiment.trials[0].arms)
        )
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAArm).count(), 7)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def test_GeneratorRunValidatedFields(self) -> None:
        # Set up an experiment with a generator run that will have modeling-related
        # fields that are not loaded on most generator runs during reduced-stat
        # experiment loading.
        exp = get_branin_experiment()
        gs = get_generation_strategy(with_callable_model_kwarg=False)
        trial = exp.new_trial(gs.gen(experiment=exp))
        for instrumented_attr in GR_LARGE_MODEL_ATTRS:
            self.assertIsNotNone(
                getattr(trial.generator_run, f"_{instrumented_attr.key}")
            )

        # Save and reload the experiment, ensure the modeling-related fields were
        # loaded are non-null.
        save_experiment(exp)
        loaded_exp = load_experiment(exp.name)
        # pyre-fixme[16]: Optional type has no attribute `generator_run`.
        loaded_gr = loaded_exp.trials.get(0).generator_run
        for instrumented_attr in GR_LARGE_MODEL_ATTRS:
            self.assertIsNotNone(getattr(loaded_gr, f"_{instrumented_attr.key}"))

        # Set modeling-related fields to `None`.
        for instrumented_attr in GR_LARGE_MODEL_ATTRS:
            setattr(loaded_gr, f"_{instrumented_attr.key}", None)
            self.assertIsNone(getattr(loaded_gr, f"_{instrumented_attr.key}"))

        # Save and reload the experiment, ensuring that setting the fields to `None`
        # was not propagated to the DB.
        save_experiment(loaded_exp)
        newly_loaded_exp = load_experiment(exp.name)
        newly_loaded_gr = newly_loaded_exp.trials.get(0).generator_run
        for instrumented_attr in GR_LARGE_MODEL_ATTRS:
            self.assertIsNotNone(getattr(newly_loaded_gr, f"_{instrumented_attr.key}"))

    @patch("ax.storage.sqa_store.db.SESSION_FACTORY", None)
    def test_MissingSessionFactory(self) -> None:
        with self.assertRaises(ValueError):
            get_session()
        with self.assertRaises(ValueError):
            get_engine()

    def test_CreateAllTablesException(self) -> None:
        engine = create_test_engine()
        engine.dialect.name = "mysql"
        engine.dialect.default_schema_name = "ax"
        with self.assertRaises(ValueError):
            create_all_tables(engine)

    def test_AnalysisCard(self) -> None:
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
            level=AnalysisCardLevel.DEBUG,
            df=test_df,
            blob="test blob",
            attributes={"foo": "bar"},
        )
        markdown_analysis_card = MarkdownAnalysisCard(
            name="test_markdown_analysis_card",
            title="test_title",
            subtitle="test_subtitle",
            level=AnalysisCardLevel.DEBUG,
            df=test_df,
            blob="This is some **really cool** markdown",
            attributes={"foo": "baz"},
        )
        plotly_analysis_card = PlotlyAnalysisCard(
            name="test_plotly_analysis_card",
            title="test_title",
            subtitle="test_subtitle",
            level=AnalysisCardLevel.DEBUG,
            df=test_df,
            blob=pio.to_json(go.Figure()),
            attributes={"foo": "bad"},
        )
        with self.subTest("test_save_analysis_cards"):
            save_experiment(self.experiment)
            save_analysis_cards(
                [base_analysis_card, markdown_analysis_card, plotly_analysis_card],
                self.experiment,
            )
        with self.subTest("test_load_analysis_cards"):
            loaded_analysis_cards = load_analysis_cards_by_experiment_name(
                self.experiment.name
            )
            self.assertEqual(len(loaded_analysis_cards), 3)
            self.assertEqual(
                loaded_analysis_cards[0].blob,
                base_analysis_card.blob,
            )
            self.assertEqual(
                loaded_analysis_cards[1].blob,
                markdown_analysis_card.blob,
            )
            self.assertEqual(
                loaded_analysis_cards[2].blob,
                plotly_analysis_card.blob,
            )

    def test_delete_generation_strategy(self) -> None:
        # GIVEN an experiment with a generation strategy
        experiment = get_branin_experiment()
        generation_strategy = choose_generation_strategy(experiment.search_space)
        generation_strategy.experiment = experiment
        save_experiment(experiment)
        save_generation_strategy(generation_strategy=generation_strategy)

        # AND GIVEN another experiment with a generation strategy
        experiment2 = get_branin_experiment()
        experiment2.name = "experiment2"
        generation_strategy2 = choose_generation_strategy(experiment2.search_space)
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
        generation_strategy = choose_generation_strategy(experiment.search_space)
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
