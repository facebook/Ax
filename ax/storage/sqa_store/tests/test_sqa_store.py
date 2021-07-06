#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.runner import Runner
from ax.core.types import ComparisonOp
from ax.exceptions.storage import SQADecodeError, SQAEncodeError
from ax.metrics.branin import BraninMetric
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from ax.storage.metric_registry import METRIC_REGISTRY, register_metric
from ax.storage.runner_registry import RUNNER_REGISTRY, register_runner
from ax.storage.sqa_store.db import (
    get_engine,
    get_session,
    init_engine_and_session_factory,
    init_test_engine_and_session_factory,
    session_scope,
)
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.delete import delete_experiment
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.load import (
    load_experiment,
    load_generation_strategy_by_experiment_name,
    load_generation_strategy_by_id,
    _get_experiment_immutable_opt_config_and_search_space,
    _get_experiment_sqa_immutable_opt_config_and_search_space,
    _get_generation_strategy_sqa_immutable_opt_config_and_search_space,
)
from ax.storage.sqa_store.save import (
    save_experiment,
    save_generation_strategy,
    save_or_update_trial,
    save_or_update_trials,
    update_generation_strategy,
    update_runner_on_experiment,
    update_properties_on_experiment,
)
from ax.storage.sqa_store.sqa_classes import (
    SQAAbandonedArm,
    SQAExperiment,
    SQAGeneratorRun,
    SQAMetric,
    SQAParameter,
    SQAParameterConstraint,
    SQARunner,
    SQATrial,
    SQAArm,
)
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.storage.sqa_store.tests.utils import TEST_CASES
from ax.storage.utils import (
    DomainType,
    MetricIntent,
    ParameterConstraintType,
)
from ax.utils.common.constants import Keys
from ax.utils.common.serialization import serialize_init_args
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_arm,
    get_branin_data,
    get_branin_experiment,
    get_branin_metric,
    get_choice_parameter,
    get_data,
    get_experiment,
    get_experiment_with_batch_trial,
    get_experiment_with_map_data_type,
    get_experiment_with_multi_objective,
    get_experiment_with_scalarized_objective_and_outcome_constraint,
    get_scalarized_outcome_constraint,
    get_fixed_parameter,
    get_generator_run,
    get_map_data,
    get_multi_objective_optimization_config,
    get_multi_type_experiment,
    get_objective,
    get_objective_threshold,
    get_optimization_config,
    get_outcome_constraint,
    get_range_parameter,
    get_range_parameter2,
    get_search_space,
    get_sum_constraint2,
    get_synthetic_runner,
)
from ax.utils.testing.modeling_stubs import get_generation_strategy

GET_GS_SQA_IMM_FUNC = _get_generation_strategy_sqa_immutable_opt_config_and_search_space


class SQAStoreTest(TestCase):
    def setUp(self):
        init_test_engine_and_session_factory(force_init=True)
        self.config = SQAConfig()
        self.encoder = Encoder(config=self.config)
        self.decoder = Decoder(config=self.config)
        self.experiment = get_experiment_with_batch_trial()
        self.dummy_parameters = [
            get_range_parameter(),  # w
            get_range_parameter2(),  # x
        ]

    def testCreationOfTestDB(self):
        init_test_engine_and_session_factory(tier_or_path=":memory:", force_init=True)
        engine = get_engine()
        self.assertIsNotNone(engine)

    def testDBConnectionWithoutForceInit(self):
        init_test_engine_and_session_factory(tier_or_path=":memory:")

    def testConnectionToDBWithURL(self):
        init_engine_and_session_factory(url="sqlite://", force_init=True)

    def testConnectionToDBWithCreator(self):
        def MockDBAPI():
            connection = Mock()

            def connect(*args, **kwargs):
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

    def testGeneratorRunTypeValidation(self):
        experiment = get_experiment_with_batch_trial()
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

    def testExperimentSaveAndLoad(self):
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

    @patch(
        f"{Decoder.__module__}.Decoder.generator_run_from_sqa",
        side_effect=Decoder(SQAConfig()).generator_run_from_sqa,
    )
    @patch(
        f"{Decoder.__module__}.Decoder.trial_from_sqa",
        side_effect=Decoder(SQAConfig()).trial_from_sqa,
    )
    @patch(
        f"{Decoder.__module__}.Decoder.experiment_from_sqa",
        side_effect=Decoder(SQAConfig()).experiment_from_sqa,
    )
    def testExperimentSaveAndLoadReducedState(
        self, _mock_exp_from_sqa, _mock_trial_from_sqa, _mock_gr_from_sqa
    ):
        # 1. No abandoned arms + no trials case, reduced state should be the
        # same as non-reduced state.
        exp = get_experiment_with_multi_objective()
        save_experiment(exp)
        loaded_experiment = load_experiment(exp.name, reduced_state=True)
        self.assertEqual(loaded_experiment, exp)
        # Make sure decoder function was called with `reduced_state=True`.
        self.assertTrue(_mock_exp_from_sqa.call_args[1].get("reduced_state"))
        _mock_exp_from_sqa.reset_mock()

        # 2. Try case with abandoned arms.
        exp = self.experiment
        save_experiment(exp)
        loaded_experiment = load_experiment(exp.name, reduced_state=True)
        # Experiments are not the same, because one has abandoned arms info.
        self.assertNotEqual(loaded_experiment, exp)
        # Remove all abandoned arms and check that all else is equal as expected.
        exp.trials.get(0)._abandoned_arms_metadata = {}
        self.assertEqual(loaded_experiment, exp)
        # Make sure that all relevant decoding functions were called with
        # `reduced_state=True` and correct number of times.
        self.assertTrue(_mock_exp_from_sqa.call_args[1].get("reduced_state"))
        self.assertTrue(_mock_trial_from_sqa.call_args[1].get("reduced_state"))
        # 2 generator runs + regular and status quo.
        self.assertTrue(_mock_gr_from_sqa.call_args[1].get("reduced_state"))
        _mock_exp_from_sqa.reset_mock()
        _mock_trial_from_sqa.reset_mock()
        _mock_gr_from_sqa.reset_mock()

        # 3. Try case with model state and search space + opt.config on a
        # generator run in the experiment.
        gr = Models.SOBOL(experiment=exp).gen(1)
        # Expecting model kwargs to have 5 fields (deduplicate, init_position, etc.)
        # and the rest of model-state info on generator run to have values too.
        self.assertEqual(len(gr._model_kwargs), 5)
        self.assertEqual(len(gr._bridge_kwargs), 7)
        self.assertEqual(len(gr._model_state_after_gen), 1)
        self.assertEqual(len(gr._gen_metadata), 0)
        self.assertIsNotNone(gr._search_space, gr.optimization_config)
        exp.new_trial(generator_run=gr)
        save_experiment(exp)
        # Make sure that all relevant decoding functions were called with
        # `reduced_state=True` and correct number of times.
        loaded_experiment = load_experiment(exp.name, reduced_state=True)
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

    def testMTExperimentSaveAndLoad(self):
        experiment = get_multi_type_experiment(add_trials=True)
        save_experiment(experiment)
        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(loaded_experiment.default_trial_type, "type1")
        self.assertEqual(len(loaded_experiment._trial_type_to_runner), 2)
        self.assertEqual(loaded_experiment.metric_to_trial_type["m1"], "type1")
        self.assertEqual(loaded_experiment.metric_to_trial_type["m2"], "type2")
        self.assertEqual(loaded_experiment._metric_to_canonical_name["m2"], "m1")
        self.assertEqual(len(loaded_experiment.trials), 2)

    def testExperimentNewTrial(self):
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

    def testExperimentNewTrialValidation(self):
        trial = self.experiment.new_batch_trial()

        with self.assertRaises(ValueError):
            # must save experiment first
            save_or_update_trial(experiment=self.experiment, trial=trial)

    def testExperimentUpdateTrial(self):
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
        self.experiment.attach_data(get_data(trial_index=trial.index))
        save_or_update_trial(experiment=self.experiment, trial=trial)

        loaded_experiment = load_experiment(self.experiment.name)
        self.assertEqual(self.experiment, loaded_experiment)

    def testExperimentSaveAndUpdateTrials(self):
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

        exp = get_experiment_with_map_data_type()
        save_experiment(exp)
        new_trial = exp.new_batch_trial(generator_run=get_generator_run())
        exp.attach_data(get_map_data(trial_index=new_trial.index))
        save_or_update_trials(
            experiment=exp,
            trials=[new_trial],
            batch_size=2,
        )
        loaded_experiment = load_experiment(exp.name)
        self.assertEqual(exp, loaded_experiment)

    def testSaveValidation(self):
        with self.assertRaises(ValueError):
            save_experiment(self.experiment.trials[0])

        experiment = get_experiment_with_batch_trial()
        experiment.name = None
        with self.assertRaises(ValueError):
            save_experiment(experiment)

    def testEncodeDecode(self):
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
            elif class_ == "GeneratorRun":
                # Need to pass in reduced_state and immutable_oc_and_ss
                converted_object = decode_func(sqa_object, False, False)
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

    def testExperimentUpdates(self):
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

    def testExperimentParameterUpdates(self):
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

    def testExperimentParameterConstraintUpdates(self):
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

    def testExperimentObjectiveUpdates(self):
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
        optimization_config.objective = Objective(metric=Metric(name="objective"))
        experiment.optimization_config = optimization_config
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics)
        )

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def testExperimentOutcomeConstraintUpdates(self):
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

    def testExperimentObjectiveThresholdUpdates(self):
        experiment = get_experiment_with_batch_trial()
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics)
        )

        # update objective threshold
        # (should perform update in place)
        optimization_config = get_multi_objective_optimization_config()
        objective_threshold = get_objective_threshold()
        optimization_config.objective_thresholds = [objective_threshold]
        experiment.optimization_config = optimization_config
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAMetric).count(), 6)

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
        self.assertEqual(get_session().query(SQAMetric).count(), 7)

        # remove outcome constraint
        # (old one should become tracking metric)
        optimization_config.outcome_constraints = []
        experiment.optimization_config = optimization_config
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAMetric).count(), 5)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

        # Optimization config should correctly reload even with no
        # objective_thresholds
        optimization_config.objective_thresholds = []
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAMetric).count(), 4)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def testFailedLoad(self):
        with self.assertRaises(ValueError):
            load_experiment("nonexistent_experiment")

    def testExperimentTrackingMetricUpdates(self):
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

    def testExperimentRunnerUpdates(self):
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

    def testExperimentTrialUpdates(self):
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

    def testExperimentAbandonedArmUpdates(self):
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

    def testExperimentGeneratorRunUpdates(self):
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
        trial.add_generator_run(generator_run=generator_run, multiplier=0.5)
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAGeneratorRun).count(), 4)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def testParameterValidation(self):
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

    def testParameterDecodeFailure(self):
        parameter = get_fixed_parameter()
        sqa_parameter = self.encoder.parameter_to_sqa(parameter)
        sqa_parameter.domain_type = 5
        with self.assertRaises(SQADecodeError):
            self.decoder.parameter_from_sqa(sqa_parameter)

    def testParameterConstraintValidation(self):
        sqa_parameter_constraint = SQAParameterConstraint(
            bound=0, constraint_dict={}, type=ParameterConstraintType.LINEAR
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
            bound=0,
            constraint_dict={},
            type=ParameterConstraintType.LINEAR,
            generator_run_id=0,
        )
        with session_scope() as session:
            session.add(sqa_parameter_constraint)
        with self.assertRaises(ValueError):
            sqa_parameter_constraint.experiment_id = 0
            with session_scope() as session:
                session.add(sqa_parameter_constraint)

    def testDecodeOrderParameterConstraintFailure(self):
        sqa_parameter = SQAParameterConstraint(
            type=ParameterConstraintType.ORDER, constraint_dict={}, bound=0
        )
        with self.assertRaises(SQADecodeError):
            self.decoder.parameter_constraint_from_sqa(
                sqa_parameter, self.dummy_parameters
            )

    def testDecodeSumParameterConstraintFailure(self):
        sqa_parameter = SQAParameterConstraint(
            type=ParameterConstraintType.SUM, constraint_dict={}, bound=0
        )
        with self.assertRaises(SQADecodeError):
            self.decoder.parameter_constraint_from_sqa(
                sqa_parameter, self.dummy_parameters
            )

    def testMetricValidation(self):
        sqa_metric = SQAMetric(
            name="foobar",
            intent=MetricIntent.OBJECTIVE,
            metric_type=METRIC_REGISTRY[BraninMetric],
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
            metric_type=METRIC_REGISTRY[BraninMetric],
            generator_run_id=0,
        )
        with session_scope() as session:
            session.add(sqa_metric)
        with self.assertRaises(ValueError):
            sqa_metric.experiment_id = 0
            with session_scope() as session:
                session.add(sqa_metric)

    def testMetricEncodeFailure(self):
        metric = get_branin_metric()
        del metric.__dict__["param_names"]
        with self.assertRaises(AttributeError):
            self.encoder.metric_to_sqa(metric)

    def testMetricDecodeFailure(self):
        metric = get_branin_metric()
        sqa_metric = self.encoder.metric_to_sqa(metric)
        sqa_metric.metric_type = "foobar"
        with self.assertRaises(SQADecodeError):
            self.decoder.metric_from_sqa(sqa_metric)

        sqa_metric.metric_type = METRIC_REGISTRY[BraninMetric]
        sqa_metric.intent = "foobar"
        with self.assertRaises(SQADecodeError):
            self.decoder.metric_from_sqa(sqa_metric)

        sqa_metric.intent = MetricIntent.TRACKING
        sqa_metric.properties = {}
        with self.assertRaises(ValueError):
            self.decoder.metric_from_sqa(sqa_metric)

    def testRunnerDecodeFailure(self):
        runner = get_synthetic_runner()
        sqa_runner = self.encoder.runner_to_sqa(runner)
        sqa_runner.runner_type = "foobar"
        with self.assertRaises(SQADecodeError):
            self.decoder.runner_from_sqa(sqa_runner)

    def testRunnerValidation(self):
        sqa_runner = SQARunner(runner_type=RUNNER_REGISTRY[SyntheticRunner])
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

        sqa_runner = SQARunner(runner_type=RUNNER_REGISTRY[SyntheticRunner], trial_id=0)
        with session_scope() as session:
            session.add(sqa_runner)
        with self.assertRaises(ValueError):
            sqa_runner.experiment_id = 0
            with session_scope() as session:
                session.add(sqa_runner)

    def testTimestampUpdate(self):
        self.experiment.trials[0]._time_staged = datetime.now()
        save_experiment(self.experiment)

        # second save should not fail
        save_experiment(self.experiment)

    def testGetProperties(self):
        # Extract default value.
        properties = serialize_init_args(Metric(name="foo"))
        self.assertEqual(
            properties, {"name": "foo", "lower_is_better": None, "properties": {}}
        )

        # Extract passed value.
        properties = serialize_init_args(
            Metric(name="foo", lower_is_better=True, properties={"foo": "bar"})
        )
        self.assertEqual(
            properties,
            {"name": "foo", "lower_is_better": True, "properties": {"foo": "bar"}},
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

        experiment = get_experiment_with_batch_trial()
        experiment.runner = MyRunner()
        experiment.add_tracking_metric(MyMetric(name="my_metric"))
        save_experiment(experiment)
        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(loaded_experiment, experiment)

    def testEncodeDecodeGenerationStrategy(self):
        # Cannot load generation strategy before it has been saved
        with self.assertRaises(ValueError):
            load_generation_strategy_by_id(gs_id=0)

        # Check that we can encode and decode the generation strategy *before*
        # it has generated some trials and been updated with some data.
        generation_strategy = get_generation_strategy()
        # Check that we can save a generation strategy without an experiment
        # attached.
        save_generation_strategy(generation_strategy=generation_strategy)
        # Also try restoring this generation strategy by its ID in the DB.
        new_generation_strategy = load_generation_strategy_by_id(
            gs_id=generation_strategy._db_id
        )
        self.assertEqual(generation_strategy, new_generation_strategy)
        self.assertIsNone(generation_strategy._experiment)

        # Cannot load generation strategy before it has been saved
        experiment = get_branin_experiment()
        save_experiment(experiment)
        with self.assertRaises(ValueError):
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
        self.assertEqual(generation_strategy, new_generation_strategy)
        self.assertIsInstance(new_generation_strategy._steps[0].model, Models)
        self.assertIsInstance(new_generation_strategy.model, ModelBridge)
        self.assertEqual(len(new_generation_strategy._generator_runs), 2)
        self.assertEqual(new_generation_strategy._experiment._name, experiment._name)

    def testEncodeDecodeGenerationStrategyReducedState(self):
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
        # Now the generation strategies should be equal.
        self.assertEqual(new_generation_strategy, generation_strategy)
        # Model should be successfully restored in generation strategy even with
        # the reduced state.
        self.assertIsInstance(new_generation_strategy._steps[0].model, Models)
        self.assertIsInstance(new_generation_strategy.model, ModelBridge)
        self.assertEqual(len(new_generation_strategy._generator_runs), 2)
        self.assertEqual(new_generation_strategy._experiment._name, experiment._name)

    def testEncodeDecodeGenerationStrategyReducedStateLoadExperiment(self):
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
        self.assertEqual(new_generation_strategy, generation_strategy)
        # Model should be successfully restored in generation strategy even with
        # the reduced state.
        self.assertIsInstance(new_generation_strategy._steps[0].model, Models)
        self.assertIsInstance(new_generation_strategy.model, ModelBridge)
        self.assertEqual(len(new_generation_strategy._generator_runs), 2)
        self.assertEqual(new_generation_strategy._experiment._name, experiment._name)

    def testUpdateGenerationStrategy(self):
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
        # During restoration of generation strategy's model from its last generator
        # run, we set `_seen_trial_indices_by_status` to that of the experiment,
        # from which we are grabbing the data to restore the model with. When the
        # experiment was updated more recently than the last `gen` from generation
        # strategy, the generation strategy prior to save might not have 'seen'
        # some recently added trials, so we update the mappings to match and check
        # that the generation strategies are equal otherwise.
        generation_strategy._seen_trial_indices_by_status[TrialStatus.CANDIDATE].add(1)
        self.assertEqual(generation_strategy, loaded_generation_strategy)

        # make sure that we can update the experiment too
        experiment.description = "foobar"
        save_experiment(experiment)
        loaded_generation_strategy = load_generation_strategy_by_experiment_name(
            experiment_name=experiment.name
        )
        self.assertEqual(generation_strategy, loaded_generation_strategy)
        self.assertEqual(
            generation_strategy._experiment.description, experiment.description
        )
        self.assertEqual(
            generation_strategy._experiment.description,
            loaded_generation_strategy._experiment.description,
        )

    def testGeneratorRunGenMetadata(self):
        gen_metadata = {"hello": "world"}
        gr = GeneratorRun(arms=[], gen_metadata=gen_metadata)
        generator_run_sqa = self.encoder.generator_run_to_sqa(gr)
        decoded_gr = self.decoder.generator_run_from_sqa(
            generator_run_sqa, False, False
        )
        self.assertEqual(decoded_gr.gen_metadata, gen_metadata)

    def testUpdateGenerationStrategyIncrementally(self):
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
        generation_strategy._save_seen_trial_indices()
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
        generation_strategy._save_seen_trial_indices()
        self.assertEqual(generation_strategy, loaded_generation_strategy)

    def testUpdateRunner(self):
        experiment = get_branin_experiment()
        with self.assertRaisesRegex(ValueError, ".* must be saved before"):
            update_runner_on_experiment(
                experiment=experiment,
                runner=None,  # This doesn't matter in this case
                encoder=self.encoder,
                decoder=self.decoder,
            )
        self.assertIsNone(experiment.runner.db_id)
        self.assertIsNotNone(experiment.runner)
        self.assertIsNone(experiment.runner.dummy_metadata)
        save_experiment(experiment=experiment)
        old_runner_db_id = experiment.runner.db_id
        self.assertIsNotNone(old_runner_db_id)
        new_runner = get_synthetic_runner()
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

    def testExperimentValidation(self):
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

    def testExperimentSaveAndDelete(self):
        for exp in [
            self.experiment,
            get_experiment_with_map_data_type(),
            get_experiment_with_multi_objective(),
            get_experiment_with_scalarized_objective_and_outcome_constraint(),
        ]:
            exp_name = exp.name
            self.assertIsNone(exp.db_id)
            save_experiment(exp)
            delete_experiment(exp_name)
            with self.assertRaises(ValueError):
                load_experiment(exp_name)

    def testGetImmutableSearchSpaceAndOptConfig(self):
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
    def testImmutableSearchSpaceAndOptConfigLoading(
        self,
        _mock_get_exp_sqa_imm_oc_ss,
        _mock_get_gs_sqa_imm_oc_ss,
        _mock_gr_from_sqa,
    ):
        experiment = get_experiment_with_batch_trial()
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

    def testSetImmutableSearchSpaceAndOptConfig(self):
        experiment = get_experiment_with_batch_trial()
        self.assertFalse(experiment.immutable_search_space_and_opt_config)
        save_experiment(experiment)

        experiment._properties[Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF] = True
        update_properties_on_experiment(
            experiment_with_updated_properties=experiment,
        )

        loaded_experiment = load_experiment(experiment.name)
        self.assertTrue(loaded_experiment.immutable_search_space_and_opt_config)

    def testRepeatedArmStorage(self):
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
