# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
from collections import Counter
from datetime import datetime
from typing import Any, get_type_hints

from ax.adapter.registry import Generators

from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.exceptions.generation_strategy import AxGenerationException
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_node_input_constructors import (
    InputConstructorPurpose,
    NodeInputConstructors,
)
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment

# type hints are stored as str when using from __future__ import annotations
EXPECTED_INPUT_CONSTRUCTOR_PARAMETER_ANNOTATIONS = [
    inspect.Parameter(
        name="previous_node",
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation="GenerationNode | None",
    ),
    inspect.Parameter(
        name="next_node",
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation="GenerationNode",
    ),
    inspect.Parameter(
        name="gs_gen_call_kwargs",
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation="dict[str, Any]",
    ),
    inspect.Parameter(
        name="experiment",
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation="Experiment",
    ),
]


class TestGenerationNodeInputConstructors(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.sobol_generator_spec = GeneratorSpec(
            generator_enum=Generators.SOBOL,
            model_kwargs={"init_position": 3},
            model_gen_kwargs={"some_gen_kwarg": "some_value"},
        )
        self.sobol_generation_node = GenerationNode(
            node_name="test", generator_specs=[self.sobol_generator_spec]
        )
        self.experiment = get_branin_experiment()
        # construct a list of grs that will mock a list of grs that would exist during
        # a gs.gen call. This list has one single arm GR, and one 3-arm GR.
        self.grs = [
            GeneratorRun(arms=[Arm(parameters={"x1": 1, "x2": 5})]),
            GeneratorRun(arms=[Arm(parameters={"x1": 1, "x2": y}) for y in range(3)]),
        ]
        self.gs = GenerationStrategy(nodes=[self.sobol_generation_node], name="test")
        # NOTE: This mapping relies on names of the NodeInputConstructors members
        # aligning with the names of the InputConstructorPurpose members, like so:
        # InputConstructorPurpose.N matches NodeInputConstructors.*_N. We may need
        # to switch to constructing this mapping manually in the future.
        self.purposes_to_input_constructors = {
            p: [ip for ip in NodeInputConstructors if ip.name.endswith(f"{p.name}")]
            for p in InputConstructorPurpose
        }

        # type hints are stored as str when using from __future__ import annotations
        self.all_purposes_expected_signatures = {
            InputConstructorPurpose.N: inspect.Signature(
                parameters=EXPECTED_INPUT_CONSTRUCTOR_PARAMETER_ANNOTATIONS,
                return_annotation="int",
            ),
            InputConstructorPurpose.FIXED_FEATURES: inspect.Signature(
                parameters=EXPECTED_INPUT_CONSTRUCTOR_PARAMETER_ANNOTATIONS,
                return_annotation="ObservationFeatures | None",
            ),
        }

    def test_all_constructors_have_expected_signature_for_purpose(self) -> None:
        """Test that all node input constructors methods have the same signature
        and that the parameters are of the expected types."""
        untested_constructors = set(NodeInputConstructors)
        for purpose, constructors in self.purposes_to_input_constructors.items():
            # For each purpose, we check that all constructors that match it,
            # share the same expected signature.
            with self.subTest(purpose=purpose, constructors=constructors):
                self.assertIn(purpose, self.all_purposes_expected_signatures)
                for constructor in constructors:
                    self.assertEqual(
                        inspect.signature(constructor._get_function_for_value()),
                        self.all_purposes_expected_signatures[purpose],
                    )
                    untested_constructors.remove(constructor)

        # There should be no untested constructors left.
        self.assertEqual(len(untested_constructors), 0)

    def test_consume_all_n_constructor(self) -> None:
        """Test that the consume_all_n_constructor returns full n."""
        num_to_gen = NodeInputConstructors.ALL_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={"n": 5},
            experiment=self.experiment,
        )
        self.assertEqual(num_to_gen, 5)

    def test_repeat_arm_n_constructor(self) -> None:
        """Test that the repeat_arm_n_constructor returns a small percentage of n."""
        medium_n = NodeInputConstructors.REPEAT_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={"n": 8},
            experiment=self.experiment,
        )
        large_n = NodeInputConstructors.REPEAT_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={"n": 11},
            experiment=self.experiment,
        )
        self.assertEqual(medium_n, 1)
        self.assertEqual(large_n, 2)

    def test_repeat_arm_n_constructor_return_0(self) -> None:
        small_n = NodeInputConstructors.REPEAT_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={"n": 5},
            experiment=self.experiment,
        )
        self.assertEqual(small_n, 0)
        self.assertTrue(self.sobol_generation_node._should_skip)

    def test_remaining_n_constructor_expect_1(self) -> None:
        """Test that the remaining_n_constructor returns the remaining n."""
        # should return 1 because 4 arms already exist and 5 are requested
        expect_1 = NodeInputConstructors.REMAINING_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={"n": 5, "grs_this_gen": self.grs},
            experiment=self.experiment,
        )
        self.assertEqual(expect_1, 1)

    def test_remaining_n_constructor_expect_0(self) -> None:
        # should return 0 because 4 arms already exist and 4 are requested
        expect_0 = NodeInputConstructors.REMAINING_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={"n": 4, "grs_this_gen": self.grs},
            experiment=self.experiment,
        )
        self.assertEqual(expect_0, 0)

    def test_remaining_n_constructor_cap_at_zero(self) -> None:
        # should return 0 because 4 arms already exist and 3 are requested
        # this is a bad state that should never be hit, but ensuring proper
        # handling here feels like a valid edge case
        expect_0 = NodeInputConstructors.REMAINING_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={"n": 3, "grs_this_gen": self.grs},
            experiment=self.experiment,
        )
        self.assertEqual(expect_0, 0)

    def test_no_n_provided_all_n(self) -> None:
        num_to_gen = NodeInputConstructors.ALL_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(num_to_gen, 10)

    def test_no_n_provided_all_n_with_exp_prop(self) -> None:
        self.experiment._properties[Keys.EXPERIMENT_TOTAL_CONCURRENT_ARMS] = 12
        num_to_gen = NodeInputConstructors.ALL_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(num_to_gen, 12)

    def test_no_n_provided_all_n_with_exp_prop_long_run(self) -> None:
        self.experiment._properties[Keys.EXPERIMENT_TOTAL_CONCURRENT_ARMS] = 13
        self.sobol_generation_node._trial_type = Keys.LONG_RUN
        num_to_gen = NodeInputConstructors.ALL_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(num_to_gen, 7)

    def test_no_n_provided_all_n_with_exp_prop_short_run(self) -> None:
        self.experiment._properties[Keys.EXPERIMENT_TOTAL_CONCURRENT_ARMS] = 13
        self.sobol_generation_node._trial_type = Keys.SHORT_RUN
        num_to_gen = NodeInputConstructors.ALL_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(num_to_gen, 6)

    def test_no_n_provided_repeat_n(self) -> None:
        num_to_gen = NodeInputConstructors.REPEAT_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(num_to_gen, 1)

    def test_no_n_provided_repeat_n_with_exp_prop(self) -> None:
        self.experiment._properties[Keys.EXPERIMENT_TOTAL_CONCURRENT_ARMS] = 18
        num_to_gen = NodeInputConstructors.REPEAT_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(num_to_gen, 2)

    def test_no_n_provided_repeat_n_with_exp_prop_long_run(self) -> None:
        self.experiment._properties[Keys.EXPERIMENT_TOTAL_CONCURRENT_ARMS] = 18
        self.sobol_generation_node._trial_type = Keys.SHORT_RUN
        num_to_gen = NodeInputConstructors.REPEAT_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        # expect 1 arm here because total concurrent arms is 18, and we have a trial
        # type (short run), so we'll take the floor of 18/2 = 9 to be used in the
        # logic for repeat arms which says if we have less than 10 requested arms we
        # should get 1 repeat arm.
        self.assertEqual(num_to_gen, 1)

    def test_no_n_provided_remaining_n(self) -> None:
        num_to_gen = NodeInputConstructors.REMAINING_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(num_to_gen, 10)

    def test_no_n_provided_remaining_n_with_exp_prop(self) -> None:
        self.experiment._properties[Keys.EXPERIMENT_TOTAL_CONCURRENT_ARMS] = 8
        num_to_gen = NodeInputConstructors.REMAINING_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={"grs_this_gen": self.grs},
            experiment=self.experiment,
        )
        self.assertEqual(num_to_gen, 4)

    def test_set_target_trial_long_run_wins(self) -> None:
        for num_arms, trial_type in zip((1, 3), (Keys.LONG_RUN, Keys.SHORT_RUN)):
            self._add_sobol_trial(
                experiment=self.experiment,
                trial_type=trial_type,
                complete=False,
                num_arms=num_arms,
            )
        self.experiment.fetch_data()
        target_trial = NodeInputConstructors.TARGET_TRIAL_FIXED_FEATURES(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(
            target_trial,
            ObservationFeatures(
                parameters={},
                trial_index=0,
            ),
        )

    def test_set_target_trial_most_arms_long_run_wins(self) -> None:
        for num_arms in (1, 3):
            self._add_sobol_trial(
                experiment=self.experiment,
                trial_type=Keys.LONG_RUN,
                complete=False,
                num_arms=num_arms,
            )
        self.experiment.fetch_data()
        # Test most arms should win
        target_trial = NodeInputConstructors.TARGET_TRIAL_FIXED_FEATURES(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(
            target_trial,
            ObservationFeatures(
                parameters={},
                trial_index=1,
            ),
        )

    def test_set_target_trial_long_run_ties(self) -> None:
        # if all things are equal we should just pick the first one
        # in the sorted list
        for _ in range(2):
            self._add_sobol_trial(
                experiment=self.experiment,
                trial_type=Keys.LONG_RUN,
                complete=False,
                num_arms=1,
            )
        self.experiment.fetch_data()
        target_trial = NodeInputConstructors.TARGET_TRIAL_FIXED_FEATURES(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(
            target_trial,
            ObservationFeatures(
                parameters={},
                trial_index=0,
            ),
        )

    def test_set_target_trial_longest_duration_long_run_wins(self) -> None:
        for _ in range(2):
            self._add_sobol_trial(
                experiment=self.experiment,
                trial_type=Keys.LONG_RUN,
                complete=False,
                num_arms=1,
            )
        self.experiment.fetch_data()
        self.experiment.trials[0]._time_run_started = datetime(2000, 1, 2)
        self.experiment.trials[1]._time_run_started = datetime(2000, 1, 1)
        target_trial = NodeInputConstructors.TARGET_TRIAL_FIXED_FEATURES(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(
            target_trial,
            ObservationFeatures(
                parameters={},
                trial_index=1,
            ),
        )

    def test_set_target_trial_running_short_trial_wins(self) -> None:
        for trial_type, complete in zip((Keys.LONG_RUN, Keys.SHORT_RUN), (True, False)):
            self._add_sobol_trial(
                experiment=self.experiment,
                trial_type=trial_type,
                complete=complete,
                num_arms=1,
            )
        self.experiment.fetch_data()
        target_trial = NodeInputConstructors.TARGET_TRIAL_FIXED_FEATURES(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(
            target_trial,
            ObservationFeatures(
                parameters={},
                trial_index=1,
            ),
        )

    def test_set_target_trial_longest_short_wins(self) -> None:
        for _ in range(2):
            self._add_sobol_trial(
                experiment=self.experiment,
                trial_type=Keys.SHORT_RUN,
                complete=False,
                num_arms=1,
            )
        self.experiment.fetch_data()
        self.experiment.trials[0]._time_run_started = datetime(2000, 1, 2)
        self.experiment.trials[1]._time_run_started = datetime(2000, 1, 1)
        target_trial = NodeInputConstructors.TARGET_TRIAL_FIXED_FEATURES(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(
            target_trial,
            ObservationFeatures(
                parameters={},
                trial_index=1,
            ),
        )

    def test_set_target_trial_most_arms_short_running_wins(self) -> None:
        for num_arms in (1, 3):
            self._add_sobol_trial(
                experiment=self.experiment,
                trial_type=Keys.SHORT_RUN,
                complete=False,
                num_arms=num_arms,
            )
        self.experiment.fetch_data()
        target_trial = NodeInputConstructors.TARGET_TRIAL_FIXED_FEATURES(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(
            target_trial,
            ObservationFeatures(
                parameters={},
                trial_index=1,
            ),
        )

    def test_set_target_trial_most_arms_complete_short_wins(self) -> None:
        for num_arms in (1, 3):
            self._add_sobol_trial(
                experiment=self.experiment,
                trial_type=Keys.SHORT_RUN,
                complete=False,
                num_arms=num_arms,
            )
        self.experiment.fetch_data()
        target_trial = NodeInputConstructors.TARGET_TRIAL_FIXED_FEATURES(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(
            target_trial,
            ObservationFeatures(
                parameters={},
                trial_index=1,
            ),
        )

    def test_set_target_trial_longest_short_complete_wins(self) -> None:
        for _ in range(2):
            self._add_sobol_trial(
                experiment=self.experiment,
                trial_type=Keys.SHORT_RUN,
                complete=True,
                num_arms=1,
            )
        self.experiment.fetch_data()
        self.experiment.trials[0]._time_run_started = datetime(2000, 1, 2)
        self.experiment.trials[1]._time_run_started = datetime(2000, 1, 1)
        target_trial = NodeInputConstructors.TARGET_TRIAL_FIXED_FEATURES(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(
            target_trial,
            ObservationFeatures(
                parameters={},
                trial_index=1,
            ),
        )

    def test_target_trial_raises_error_if_none_found(self) -> None:
        with self.assertRaisesRegex(
            AxGenerationException,
            "Often this could be due to no trials",
        ):
            NodeInputConstructors.TARGET_TRIAL_FIXED_FEATURES(
                previous_node=None,
                next_node=self.sobol_generation_node,
                gs_gen_call_kwargs={},
                experiment=self.experiment,
            )

    def test_target_trial_latest_trial_no_data(self) -> None:
        for _ in range(2):
            self._add_sobol_trial(
                experiment=self.experiment,
                trial_type=Keys.SHORT_RUN,
                complete=True,
                num_arms=1,
            )
        self.experiment.fetch_data(trial_indices=[0])
        target_trial = NodeInputConstructors.TARGET_TRIAL_FIXED_FEATURES(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={},
            experiment=self.experiment,
        )
        self.assertEqual(
            target_trial,
            ObservationFeatures(
                parameters={},
                trial_index=0,
            ),
        )

    def _add_sobol_trial(
        self,
        experiment: Experiment,
        trial_type: str | None = None,
        complete: bool = True,
        num_arms: int = 1,
        with_status_quo: bool = True,
    ) -> BatchTrial:
        """Helper function to add a trial to an experiment, takes a trial type and
        whether or not the trial is complete, and number of arms"""
        grs = []
        for i in range(num_arms):
            grs.append(GeneratorRun(arms=[Arm(parameters={"x1": 1, "x2": i})]))
        trial = experiment.new_batch_trial(
            trial_type=trial_type,
            generator_runs=grs,
        )
        if with_status_quo:
            experiment.status_quo = Arm(parameters={"x1": 0, "x2": 0})
            trial.set_status_quo_with_weight(
                status_quo=self.experiment.status_quo,
                weight=1.0,
            )
        trial.run()
        if complete:
            trial.mark_completed()
        return trial


class TestInstantiationFromNodeInputConstructor(TestCase):
    """Class to test that all node input constructors can be instantiated and are
    being tested."""

    def setUp(self) -> None:
        super().setUp()
        self.constructor_cases = {
            "ALl_N": NodeInputConstructors.ALL_N,
            "REPEAT_N": NodeInputConstructors.REPEAT_N,
            "REMAINING_N": NodeInputConstructors.REMAINING_N,
        }
        self.purpose_cases = {
            "N": InputConstructorPurpose.N,
        }

    def test_all_constructors_have_same_signature(self) -> None:
        """Test that all node input constructors methods have the same signature
        and that the parameters are of the expected types"""
        all_constructors_tested = list(self.constructor_cases.values())
        method_signature = inspect.signature(all_constructors_tested[0])
        for constructor in all_constructors_tested[1:]:
            with self.subTest(constructor=constructor):
                func_parameters = get_type_hints(constructor.__call__)
                self.assertEqual(
                    Counter(list(func_parameters.keys())),
                    Counter(
                        [
                            "previous_node",
                            "next_node",
                            "gs_gen_call_kwargs",
                            "experiment",
                            "return",
                        ]
                    ),
                )
                self.assertEqual(
                    func_parameters["previous_node"], GenerationNode | None
                )
                self.assertEqual(func_parameters["next_node"], GenerationNode)
                self.assertEqual(func_parameters["gs_gen_call_kwargs"], dict[str, Any])
                self.assertEqual(func_parameters["experiment"], Experiment)
                self.assertEqual(method_signature, inspect.signature(constructor))
