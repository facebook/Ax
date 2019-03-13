#!/usr/bin/env python3

from collections import OrderedDict
from datetime import datetime
from unittest.mock import MagicMock, Mock

from ae.lazarus.ae.core.arm import Arm
from ae.lazarus.ae.core.batch_trial import AbandonedArm
from ae.lazarus.ae.core.metric import Metric
from ae.lazarus.ae.core.objective import Objective
from ae.lazarus.ae.core.outcome_constraint import OutcomeConstraint
from ae.lazarus.ae.core.parameter import ParameterType, RangeParameter
from ae.lazarus.ae.core.types.types import ComparisonOp
from ae.lazarus.ae.exceptions.storage import (
    ImmutabilityError,
    SQADecodeError,
    SQAEncodeError,
)
from ae.lazarus.ae.metrics.branin import BraninMetric
from ae.lazarus.ae.runners.synthetic import SyntheticRunner
from ae.lazarus.ae.storage.sqa_store.base_decoder import Decoder
from ae.lazarus.ae.storage.sqa_store.base_encoder import Encoder
from ae.lazarus.ae.storage.sqa_store.db import (
    SQABase,
    get_engine,
    get_session,
    init_engine_and_session_factory,
    init_test_engine_and_session_factory,
    session_scope,
)
from ae.lazarus.ae.storage.sqa_store.load import load_experiment
from ae.lazarus.ae.storage.sqa_store.save import save_experiment
from ae.lazarus.ae.storage.sqa_store.sqa_classes import (
    SQAAbandonedArm,
    SQAExperiment,
    SQAGeneratorRun,
    SQAMetric,
    SQAParameter,
    SQAParameterConstraint,
    SQARunner,
    SQATrial,
)
from ae.lazarus.ae.storage.sqa_store.tests.utils import (
    ENCODE_DECODE_FIELD_MAPS,
    TEST_CASES,
)
from ae.lazarus.ae.storage.sqa_store.utils import is_foreign_key_field
from ae.lazarus.ae.storage.utils import (
    DomainType,
    MetricIntent,
    ParameterConstraintType,
    remove_prefix,
)
from ae.lazarus.ae.tests.fake import (
    get_arm,
    get_batch_trial,
    get_branin_metric,
    get_choice_parameter,
    get_experiment_with_batch_trial,
    get_fixed_parameter,
    get_generator_run,
    get_objective,
    get_optimization_config,
    get_outcome_constraint,
    get_range_parameter,
    get_search_space,
    get_sum_constraint1,
    get_synthetic_runner,
)
from ae.lazarus.ae.utils.common.testutils import TestCase


class SQAStoreTest(TestCase):
    def setUp(self):
        init_test_engine_and_session_factory(force_init=True)
        self.experiment = get_experiment_with_batch_trial()

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

    def testEquals(self):
        trial_sqa = Encoder.trial_to_sqa(get_batch_trial())
        self.assertTrue(trial_sqa.equals(trial_sqa))

    def testListEquals(self):
        self.assertTrue(SQABase.list_equals([1, 2, 3], [1, 2, 3]))
        self.assertFalse(SQABase.list_equals([4], ["foo"]))

        with self.assertRaises(ValueError):
            SQABase.list_equals([[4]], [[4]])

    def testListUpdate(self):
        self.assertEqual(SQABase.list_update([1, 2, 3], [1, 2, 3]), [1, 2, 3])
        self.assertEqual(SQABase.list_update([4], ["foo"]), ["foo"])

        with self.assertRaises(ValueError):
            SQABase.list_update([[4]], [[4]])

    def testValidateUpdate(self):
        parameter = get_choice_parameter()
        parameter_sqa = Encoder.parameter_to_sqa(parameter)
        parameter2 = get_choice_parameter()
        parameter2._name = 5
        parameter_sqa_2 = Encoder.parameter_to_sqa(parameter2)
        with self.assertRaises(ImmutabilityError):
            parameter_sqa.validate_update(parameter_sqa_2)

    def testGeneratorRunTypeValidation(self):
        experiment = get_experiment_with_batch_trial()
        generator_run = experiment.trials[0].generator_run_structs[0].generator_run
        generator_run._generator_run_type = "foobar"
        with self.assertRaises(SQAEncodeError):
            Encoder.generator_run_to_sqa(generator_run)

        generator_run._generator_run_type = "STATUS_QUO"
        generator_run_sqa = Encoder.generator_run_to_sqa(generator_run)
        generator_run_sqa.generator_run_type = 1
        with self.assertRaises(SQADecodeError):
            Decoder.generator_run_from_sqa(generator_run_sqa)

        generator_run_sqa.generator_run_type = 0
        Decoder.generator_run_from_sqa(generator_run_sqa)

    def testExperimentSaveAndLoad(self):
        save_experiment(self.experiment)
        loaded_experiment = load_experiment(self.experiment.name)
        self.assertEqual(loaded_experiment, self.experiment)

    def testEncodeDecode(self):
        for class_, fake_func, encode_func, decode_func in TEST_CASES:
            # Can't load trials from SQL, because a trial needs an experiment
            # in order to be initialized
            if class_ == "BatchTrial" or class_ == "Trial":
                continue
            original_object = fake_func()
            sqa_object = encode_func(original_object)
            converted_object = decode_func(sqa_object)
            self.assertEqual(
                original_object,
                converted_object,
                msg=f"Error encoding/decoding {class_}.",
            )

    def testEncoders(self):
        for class_, fake_func, encode_func, _ in TEST_CASES:
            original_object = fake_func()
            sqa_object = encode_func(original_object)
            if isinstance(
                original_object, AbandonedArm
            ):  # handle NamedTuple differently
                object_keys = original_object._asdict().keys()
            else:
                object_keys = original_object.__dict__.keys()
            object_keys = {remove_prefix(key, "_") for key in object_keys}
            sqa_keys = {
                remove_prefix(key, "_")
                for key in sqa_object.attributes
                if key not in ["id", "_sa_instance_state"]
                and not is_foreign_key_field(key)
            }

            # Account for fields that appear in the Python object but not the SQA
            # the SQA but not the Python, and for fields that appear in both places
            # but with different names
            if class_ in ENCODE_DECODE_FIELD_MAPS:
                map = ENCODE_DECODE_FIELD_MAPS[class_]
                for field in map.python_only:
                    sqa_keys.add(field)
                for field in map.encoded_only:
                    object_keys.add(field)
                for python, encoded in map.python_to_encoded.items():
                    sqa_keys.remove(encoded)
                    sqa_keys.add(python)

            self.assertEqual(
                object_keys,
                sqa_keys,
                msg=f"Mismatch between Python and SQA representation in {class_}.",
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

        experiment.status_quo = Arm(
            params={"w": 0.0, "x": 1, "y": "y", "z": True}, name="new_status_quo"
        )
        save_experiment(experiment)
        self.assertEqual(get_session().query(SQAExperiment).count(), 1)

        loaded_experiment = load_experiment(experiment.name)
        self.assertEqual(experiment, loaded_experiment)

    def testExperimentParameterUpdates(self):
        experiment = get_experiment_with_batch_trial()
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
            get_session().query(SQAParameterConstraint).count(),  # 1
            len(experiment.search_space.parameter_constraints),  # 1
        )

        # add a parameter constraint
        search_space = experiment.search_space
        existing_constraint = experiment.search_space.parameter_constraints[0]
        new_constraint = get_sum_constraint1()
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
        experiment.update_metric(metric)
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics)
        )

        # add tracking metric
        metric = Metric(name="tracking2")
        experiment.add_metric(metric)
        save_experiment(experiment)
        self.assertEqual(
            get_session().query(SQAMetric).count(), len(experiment.metrics)
        )

        # remove tracking metric
        # (old one should be deleted)
        del experiment._metrics["tracking2"]
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
        trial.mark_arm_abandoned(trial.arms[1])
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

        # try to edit a generator run's search space
        generator_run_struct = trial._generator_run_structs[0]
        old_search_space = generator_run_struct.generator_run._search_space
        search_space = get_search_space()
        parameter = RangeParameter(
            name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
        )
        search_space.add_parameter(parameter)
        generator_run_struct.generator_run._search_space = search_space
        with self.assertRaises(ImmutabilityError):
            save_experiment(experiment)

        # undo change so that equality test below passes
        generator_run_struct.generator_run._search_space = old_search_space

        # try to edit a generator run's optimization config (not allowed)
        old_optimization_config = (
            generator_run_struct.generator_run._optimization_config
        )
        optimization_config = get_optimization_config()
        objective = get_objective()
        objective.minimize = True
        optimization_config.objective = objective
        generator_run_struct.generator_run._optimization_config = optimization_config
        with self.assertRaises(ImmutabilityError):
            save_experiment(experiment)

        # undo change so that equality test below passes
        generator_run_struct.generator_run._optimization_config = (
            old_optimization_config
        )

        # try to edit a generator run's arms (not allowed)
        old_arm_weight_table = generator_run_struct.generator_run._arm_weight_table
        generator_run_struct.generator_run._arm_weight_table = OrderedDict()
        with self.assertRaises(ImmutabilityError):
            save_experiment(experiment)

        # undo change so that equality test below passes
        generator_run_struct.generator_run._arm_weight_table = old_arm_weight_table

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
        sqa_parameter = Encoder.parameter_to_sqa(parameter)
        sqa_parameter.domain_type = 5
        with self.assertRaises(SQADecodeError):
            Decoder.parameter_from_sqa(sqa_parameter)

    def testParameterUpdateFailure(self):
        parameter = get_range_parameter()
        sqa_parameter = Encoder.parameter_to_sqa(parameter)
        parameter._name = "new"
        sqa_parameter_2 = Encoder.parameter_to_sqa(parameter)
        with self.assertRaises(ImmutabilityError):
            sqa_parameter.update(sqa_parameter_2)

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
            Decoder.parameter_constraint_from_sqa(sqa_parameter)

    def testDecodeSumParameterConstraintFailure(self):
        sqa_parameter = SQAParameterConstraint(
            type=ParameterConstraintType.SUM, constraint_dict={}, bound=0
        )
        with self.assertRaises(SQADecodeError):
            Decoder.parameter_constraint_from_sqa(sqa_parameter)

    def testMetricValidation(self):
        sqa_metric = SQAMetric(
            name="foobar",
            intent=MetricIntent.OBJECTIVE,
            metric_type=Encoder.metric_registry.CLASS_TO_TYPE[BraninMetric],
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
            metric_type=Encoder.metric_registry.CLASS_TO_TYPE[BraninMetric],
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
        with self.assertRaises(SQAEncodeError):
            Encoder.metric_to_sqa(metric)

    def testMetricDecodeFailure(self):
        metric = get_branin_metric()
        sqa_metric = Encoder.metric_to_sqa(metric)
        sqa_metric.metric_type = "foobar"
        with self.assertRaises(SQADecodeError):
            Decoder.metric_from_sqa(sqa_metric)

        sqa_metric.metric_type = Decoder.metric_registry.CLASS_TO_TYPE[BraninMetric]
        sqa_metric.intent = "foobar"
        with self.assertRaises(SQADecodeError):
            Decoder.metric_from_sqa(sqa_metric)

        sqa_metric.intent = MetricIntent.TRACKING
        sqa_metric.properties = {}
        with self.assertRaises(AttributeError):
            Decoder.metric_from_sqa(sqa_metric)

    def testRunnerDecodeFailure(self):
        runner = get_synthetic_runner()
        sqa_runner = Encoder.runner_to_sqa(runner)
        sqa_runner.runner_type = "foobar"
        with self.assertRaises(SQADecodeError):
            Decoder.runner_from_sqa(sqa_runner)

    def testRunnerValidation(self):
        sqa_runner = SQARunner(
            runner_type=Encoder.runner_registry.CLASS_TO_TYPE[SyntheticRunner]
        )
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
            runner_type=Encoder.runner_registry.CLASS_TO_TYPE[SyntheticRunner],
            trial_id=0,
        )
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
