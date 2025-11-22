#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import MagicMock, patch

import torch
from ax.adapter.factory import get_sobol
from ax.adapter.registry import Generators
from ax.core.observation import ObservationFeatures
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UserInputError
from ax.exceptions.model import ModelError
from ax.generation_strategy.best_model_selector import (
    ReductionCriterion,
    SingleDiagnosticBestModelSelector,
)
from ax.generation_strategy.generation_node import (
    GenerationNode,
    GenerationStep,
    logger,
    MISSING_MODEL_SELECTOR_MESSAGE,
)
from ax.generation_strategy.generation_node_input_constructors import (
    InputConstructorPurpose,
    NodeInputConstructors,
)
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import mock_botorch_optimize
from botorch.sampling.normal import SobolQMCNormalSampler
from pyre_extensions import none_throws


class TestGenerationNode(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.sobol_generator_spec = GeneratorSpec(
            generator_enum=Generators.SOBOL,
            model_kwargs={"init_position": 3},
            model_gen_kwargs={"some_gen_kwarg": "some_value"},
        )
        self.mbm_generator_spec = GeneratorSpec(
            generator_enum=Generators.BOTORCH_MODULAR,
            model_kwargs={},
            model_gen_kwargs={},
        )
        self.sobol_generation_node = GenerationNode(
            name="test", generator_specs=[self.sobol_generator_spec]
        )
        self.branin_experiment = get_branin_experiment(with_completed_trial=True)
        self.branin_data = self.branin_experiment.lookup_data()
        self.node_short = GenerationNode(
            name="test",
            generator_specs=[self.sobol_generator_spec],
            trial_type=Keys.SHORT_RUN,
        )

    def test_init(self) -> None:
        self.assertEqual(
            self.sobol_generation_node.generator_specs, [self.sobol_generator_spec]
        )
        with self.assertRaisesRegex(UserInputError, "Model keys must be unique"):
            GenerationNode(
                name="test",
                generator_specs=[self.sobol_generator_spec, self.sobol_generator_spec],
            )
        mbm_specs = [
            GeneratorSpec(generator_enum=Generators.BOTORCH_MODULAR),
            GeneratorSpec(
                generator_enum=Generators.BOTORCH_MODULAR, model_key_override="MBM v2"
            ),
        ]
        with self.assertRaisesRegex(UserInputError, MISSING_MODEL_SELECTOR_MESSAGE):
            GenerationNode(
                name="test",
                generator_specs=mbm_specs,
            )
        model_selector = SingleDiagnosticBestModelSelector(
            diagnostic="Fisher exact test p",
            metric_aggregation=ReductionCriterion.MEAN,
            criterion=ReductionCriterion.MIN,
        )
        node = GenerationNode(
            name="test",
            generator_specs=mbm_specs,
            best_model_selector=model_selector,
        )
        self.assertEqual(node.generator_specs, mbm_specs)
        self.assertIs(node.best_model_selector, model_selector)

    def test_input_constructor_none(self) -> None:
        self.assertEqual(self.sobol_generation_node._input_constructors, {})
        self.assertEqual(self.sobol_generation_node.input_constructors, {})

    def test_incorrect_trial_type(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "Trial type must be either"):
            GenerationNode(
                name="test",
                generator_specs=[self.sobol_generator_spec],
                trial_type="foo",
            )

    def test_init_with_trial_type(self) -> None:
        node_long = GenerationNode(
            name="test",
            generator_specs=[self.sobol_generator_spec],
            trial_type=Keys.LONG_RUN,
        )
        node_default = GenerationNode(
            name="test",
            generator_specs=[self.sobol_generator_spec],
        )
        self.assertEqual(self.node_short._trial_type, Keys.SHORT_RUN)
        self.assertEqual(node_long._trial_type, Keys.LONG_RUN)
        self.assertIsNone(node_default._trial_type)

    def test_input_constructor(self) -> None:
        node = GenerationNode(
            name="test",
            generator_specs=[self.sobol_generator_spec],
            input_constructors={InputConstructorPurpose.N: NodeInputConstructors.ALL_N},
        )
        self.assertEqual(
            node.input_constructors,
            {InputConstructorPurpose.N: NodeInputConstructors.ALL_N},
        )
        self.assertEqual(
            node._input_constructors,
            {InputConstructorPurpose.N: NodeInputConstructors.ALL_N},
        )

    def test_fit(self) -> None:
        with patch.object(
            self.sobol_generator_spec, "fit", wraps=self.sobol_generator_spec.fit
        ) as mock_generator_spec_fit:
            self.sobol_generation_node._fit(
                experiment=self.branin_experiment,
                data=self.branin_data,
            )
        mock_generator_spec_fit.assert_called_with(
            experiment=self.branin_experiment, data=self.branin_data
        )

    def test_gen(self) -> None:
        with (
            patch.object(
                self.sobol_generator_spec, "gen", wraps=self.sobol_generator_spec.gen
            ) as mock_generator_spec_gen,
            patch.object(
                self.sobol_generator_spec, "fit", wraps=self.sobol_generator_spec.fit
            ) as mock_generator_spec_fit,
        ):
            gr = self.sobol_generation_node.gen(
                experiment=self.branin_experiment,
                data=self.branin_experiment.lookup_data(),
                n=1,
                pending_observations={"branin": []},
            )
            self.assertIsNotNone(gr)
            self.assertEqual(gr._model_key, self.sobol_generator_spec.model_key)
            model_kwargs = gr._model_kwargs
            self.assertIsNotNone(model_kwargs)
            self.assertEqual(model_kwargs.get("init_position"), 3)
        mock_generator_spec_fit.assert_called_with(
            experiment=self.branin_experiment, data=self.branin_experiment.lookup_data()
        )
        mock_generator_spec_gen.assert_called_with(
            experiment=self.branin_experiment,
            data=self.branin_experiment.lookup_data(),
            n=1,
            pending_observations={"branin": []},
            fixed_features=None,
        )

    @mock_botorch_optimize
    def test_gen_with_trial_type(self) -> None:
        mbm_short = GenerationNode(
            name="test",
            generator_specs=[
                GeneratorSpec(
                    generator_enum=Generators.BOTORCH_MODULAR,
                    model_kwargs={},
                    model_gen_kwargs={
                        "n": 1,
                        "fixed_features": ObservationFeatures(
                            parameters={},
                            trial_index=0,
                        ),
                    },
                ),
            ],
            trial_type=Keys.SHORT_RUN,
        )
        gr = mbm_short.gen(
            experiment=self.branin_experiment,
            data=self.branin_experiment.lookup_data(),
            pending_observations=None,
            n=2,
        )
        self.assertIsNotNone(gr)
        gen_metadata = gr.gen_metadata
        self.assertIsNotNone(gen_metadata)
        self.assertEqual(gen_metadata["trial_type"], Keys.SHORT_RUN)
        # validate that other fields in gen_metadata are preserved
        self.assertIsNotNone(gen_metadata[Keys.EXPECTED_ACQF_VAL])

    def test_gen_with_no_trial_type(self) -> None:
        gr = self.sobol_generation_node.gen(
            experiment=self.branin_experiment,
            data=self.branin_experiment.lookup_data(),
            pending_observations=None,
            n=2,
        )
        self.assertIsNotNone(gr)
        self.assertNotIn("trial_type", none_throws(gr.gen_metadata))

    @mock_botorch_optimize
    def test_model_gen_kwargs_deepcopy(self) -> None:
        sampler = SobolQMCNormalSampler(torch.Size([1]))
        node = GenerationNode(
            name="test",
            generator_specs=[
                GeneratorSpec(
                    generator_enum=Generators.BOTORCH_MODULAR,
                    model_kwargs={},
                    model_gen_kwargs={
                        "n": 1,
                        "fixed_features": ObservationFeatures(
                            parameters={},
                            trial_index=0,
                        ),
                        "model_gen_options": {Keys.ACQF_KWARGS: {"sampler": sampler}},
                    },
                ),
            ],
        )
        dat = self.branin_experiment.lookup_data()
        node.gen(
            experiment=self.branin_experiment,
            data=dat,
            n=1,
            pending_observations={"branin": []},
        )
        # verify that sampler is not modified in-place by checking base samples
        self.assertIs(
            node.generator_spec_to_gen_from.model_gen_kwargs["model_gen_options"][
                Keys.ACQF_KWARGS
            ]["sampler"],
            sampler,
        )
        self.assertIsNone(sampler.base_samples)

    @mock_botorch_optimize
    def test_properties(self) -> None:
        node = GenerationNode(
            name="test",
            generator_specs=[
                GeneratorSpec(
                    generator_enum=Generators.BOTORCH_MODULAR,
                    model_kwargs={},
                    model_gen_kwargs={
                        "n": 1,
                        "fixed_features": ObservationFeatures(
                            parameters={},
                            trial_index=0,
                        ),
                    },
                ),
            ],
        )
        self.assertEqual(node.model_to_gen_from_name, "BoTorch")
        node._fit(
            experiment=self.branin_experiment,
            data=self.branin_data,
        )
        self.assertEqual(
            node.generator_spec_to_gen_from.generator_enum,
            node.generator_specs[0].generator_enum,
        )
        self.assertEqual(
            node.generator_spec_to_gen_from.model_kwargs,
            node.generator_specs[0].model_kwargs,
        )
        self.assertEqual(node.model_to_gen_from_name, "BoTorch")
        self.assertEqual(
            node.generator_spec_to_gen_from.model_gen_kwargs,
            node.generator_specs[0].model_gen_kwargs,
        )
        self.assertEqual(
            node.generator_spec_to_gen_from.model_cv_kwargs,
            node.generator_specs[0].model_cv_kwargs,
        )
        self.assertEqual(
            node.generator_spec_to_gen_from.fixed_features,
            node.generator_specs[0].fixed_features,
        )
        self.assertEqual(
            node.generator_spec_to_gen_from.cv_results,
            node.generator_specs[0].cv_results,
        )
        self.assertEqual(
            node.generator_spec_to_gen_from.diagnostics,
            node.generator_specs[0].diagnostics,
        )
        self.assertEqual(node.name, "test")
        self.assertEqual(node._unique_id, "test")

    def test_node_string_representation(self) -> None:
        node = GenerationNode(
            name="test",
            generator_specs=[
                self.mbm_generator_spec,
            ],
            transition_criteria=[
                MinTrials(threshold=5, only_in_statuses=[TrialStatus.RUNNING])
            ],
        )
        string_rep = str(node)

        self.assertEqual(
            string_rep,
            "GenerationNode(name='test', "
            "generator_specs=[GeneratorSpec(generator_enum=BoTorch, "
            "model_key_override=None)], "
            "transition_criteria=[MinTrials(transition_to='None')])",
        )

    def test_single_fixed_features(self) -> None:
        node = GenerationNode(
            name="test",
            generator_specs=[
                GeneratorSpec(
                    generator_enum=Generators.BOTORCH_MODULAR,
                    model_kwargs={},
                    model_gen_kwargs={
                        "n": 2,
                        "fixed_features": ObservationFeatures(parameters={"x": 0}),
                    },
                ),
            ],
        )
        self.assertEqual(
            node.generator_spec_to_gen_from.fixed_features,
            ObservationFeatures(parameters={"x": 0}),
        )

    def test_disabled_parameters(self) -> None:
        input_constructors = self.sobol_generation_node.apply_input_constructors(
            experiment=self.branin_experiment, gen_kwargs={}
        )
        self.assertIsNone(input_constructors["fixed_features"])
        # Disable parameter
        self.branin_experiment.disable_parameters_in_search_space({"x1": 1.2345})
        input_constructors = self.sobol_generation_node.apply_input_constructors(
            experiment=self.branin_experiment, gen_kwargs={}
        )
        expected_fixed_features = ObservationFeatures(parameters={"x1": 1.2345})
        self.assertEqual(input_constructors["fixed_features"], expected_fixed_features)
        # Test fixed features override
        input_constructors = self.sobol_generation_node.apply_input_constructors(
            experiment=self.branin_experiment,
            gen_kwargs={
                "fixed_features": ObservationFeatures(parameters={"x1": 0.0, "x2": 0.0})
            },
        )
        # The passed fixed feature overrides the disabled parameter default value
        expected_fixed_features = ObservationFeatures(parameters={"x1": 0.0, "x2": 0.0})
        self.assertEqual(input_constructors["fixed_features"], expected_fixed_features)


class TestGenerationStep(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.model_kwargs = {"init_position": 5}
        self.sobol_generation_step = GenerationStep(
            generator=Generators.SOBOL,
            num_trials=5,
            model_kwargs=self.model_kwargs,
        )
        self.generator_spec = GeneratorSpec(
            generator_enum=self.sobol_generation_step.generator,
            model_kwargs=self.model_kwargs,
        )

    def test_init(self) -> None:
        self.assertEqual(
            self.sobol_generation_step.generator_specs,
            [self.generator_spec],
        )
        self.assertEqual(self.sobol_generation_step.generator_name, "Sobol")
        self.assertEqual(
            self.sobol_generation_step.transition_criteria,
            [
                MinTrials(
                    threshold=5,
                    not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
                    block_gen_if_met=True,
                    block_transition_if_unmet=True,
                    use_all_trials_in_exp=False,
                ),
            ],
        )

        named_generation_step = GenerationStep(
            generator=Generators.SOBOL,
            num_trials=5,
            min_trials_observed=3,
            model_kwargs=self.model_kwargs,
            enforce_num_trials=False,
            generator_name="Custom Sobol",
            use_all_trials_in_exp=True,
        )
        self.assertEqual(named_generation_step.generator_name, "Custom Sobol")
        self.assertEqual(
            named_generation_step.transition_criteria,
            [
                MinTrials(
                    threshold=5,
                    not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
                    block_gen_if_met=False,
                    block_transition_if_unmet=True,
                    use_all_trials_in_exp=True,
                ),
                MinTrials(
                    only_in_statuses=[
                        TrialStatus.COMPLETED,
                        TrialStatus.EARLY_STOPPED,
                    ],
                    threshold=3,
                    block_gen_if_met=False,
                    block_transition_if_unmet=True,
                    use_all_trials_in_exp=True,
                ),
            ],
        )

    def test_min_trials_observed(self) -> None:
        with self.assertRaisesRegex(UserInputError, "min_trials_observed > num_trials"):
            GenerationStep(
                generator=Generators.SOBOL,
                num_trials=5,
                min_trials_observed=10,
                model_kwargs=self.model_kwargs,
            )

    def test_init_factory_function(self) -> None:
        with self.assertRaisesRegex(
            UserInputError, "must be a `GeneratorRegistryBase`"
        ):
            # pyre-ignore [6]: Testing deprecated input.
            GenerationStep(generator=get_sobol, num_trials=-1)

    def test_properties(self) -> None:
        step = self.sobol_generation_step
        self.assertEqual(step.generator_spec, self.generator_spec)
        self.assertEqual(step._unique_id, "-1")
        # Make sure that model_kwargs and model_gen_kwargs are synchronized
        # to the underlying model spec.
        spec = step.generator_spec
        spec.model_kwargs.update({"new_kwarg": 1})
        spec.model_gen_kwargs.update({"new_gen_kwarg": 1})
        self.assertEqual(step.model_kwargs, spec.model_kwargs)
        self.assertEqual(step.model_gen_kwargs, spec.model_gen_kwargs)


class TestGenerationNodeWithBestModelSelector(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.branin_experiment = get_branin_experiment(
            with_batch=True, with_completed_batch=True
        )
        self.ms_mixed = GeneratorSpec(generator_enum=Generators.BO_MIXED)
        self.ms_botorch = GeneratorSpec(generator_enum=Generators.BOTORCH_MODULAR)

        self.mock_aggregation = MagicMock(
            side_effect=ReductionCriterion.MEAN, spec=ReductionCriterion
        )
        self.model_selection_node = GenerationNode(
            name="test",
            generator_specs=[self.ms_mixed, self.ms_botorch],
            best_model_selector=SingleDiagnosticBestModelSelector(
                diagnostic="Fisher exact test p",
                metric_aggregation=self.mock_aggregation,
                criterion=ReductionCriterion.MIN,
            ),
        )

    @mock_botorch_optimize
    def test_gen(self) -> None:
        # Check that with `ModelSelectionNode` generation from a node with
        # multiple model specs does not fail.
        with patch.object(
            self.model_selection_node, "_fit", wraps=self.model_selection_node._fit
        ) as mock_fit:
            gr = self.model_selection_node.gen(
                experiment=self.branin_experiment,
                data=self.branin_experiment.lookup_data(),
                n=1,
                pending_observations={"branin": []},
            )
        # The model specs are practically identical for this example.
        # May pick either one.
        self.assertIsNotNone(gr)
        self.assertEqual(
            self.model_selection_node.model_to_gen_from_name, gr._model_key
        )
        mock_fit.assert_called_with(
            experiment=self.branin_experiment, data=self.branin_experiment.lookup_data()
        )
        # Check that the metric aggregation function is called twice, once for each
        # model spec.
        self.assertEqual(self.mock_aggregation.call_count, 2)

    @mock_botorch_optimize
    def test_pick_fitted_adapter_with_fit_errors(self) -> None:
        # Make model fitting error out for both specs. We should get an error.
        with (
            patch(
                "ax.generation_strategy.generator_spec.GeneratorSpec.fit",
                side_effect=RuntimeError,
            ),
            self.assertLogs(logger=logger, level="ERROR") as mock_logs,
        ):
            self.model_selection_node._fit(experiment=self.branin_experiment)
        self.assertEqual(len(mock_logs.records), 2)
        with self.assertRaisesRegex(ModelError, "No fitted models were found"):
            self.model_selection_node.generator_spec_to_gen_from

        # node._fitted_adapter returns None (rather than erroring out).
        self.assertIsNone(self.model_selection_node._fitted_adapter)

        # Only one spec errors out.
        with (
            patch.object(self.ms_mixed, "fit", side_effect=RuntimeError),
            self.assertLogs(logger=logger, level="ERROR") as mock_logs,
        ):
            self.model_selection_node._fit(experiment=self.branin_experiment)
        self.assertEqual(len(mock_logs.records), 1)
        # Picks the model that didn't error out.
        self.assertEqual(
            self.model_selection_node.generator_spec_to_gen_from, self.ms_botorch
        )

    @mock_botorch_optimize
    def test_best_model_selection_errors(self) -> None:
        # Testing that the errors raised within best model selector are
        # gracefully handled. In this case, we'll get an error in CV
        # due to insufficient training data.
        exp = get_branin_experiment(with_completed_trial=True)
        self.model_selection_node._fit(experiment=exp)
        # Check that it selected the first generator and logged a warning.
        with self.assertLogs(logger=logger) as logs:
            self.assertEqual(
                self.model_selection_node.generator_spec_to_gen_from, self.ms_mixed
            )
        self.assertTrue(
            any("raised an error when selecting" in str(log) for log in logs)
        )
