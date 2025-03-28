#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import MagicMock, patch

import torch
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
from ax.generation_strategy.model_spec import (
    FactoryFunctionGeneratorSpec,
    GeneratorSpec,
)
from ax.generation_strategy.transition_criterion import MinTrials
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.registry import Generators
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import mock_botorch_optimize
from botorch.sampling.normal import SobolQMCNormalSampler
from pyre_extensions import none_throws


class TestGenerationNode(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.sobol_model_spec = GeneratorSpec(
            model_enum=Generators.SOBOL,
            model_kwargs={"init_position": 3},
            model_gen_kwargs={"some_gen_kwarg": "some_value"},
        )
        self.mbm_model_spec = GeneratorSpec(
            model_enum=Generators.BOTORCH_MODULAR,
            model_kwargs={},
            model_gen_kwargs={},
        )
        self.sobol_generation_node = GenerationNode(
            node_name="test", model_specs=[self.sobol_model_spec]
        )
        self.branin_experiment = get_branin_experiment(with_completed_trial=True)
        self.branin_data = self.branin_experiment.lookup_data()
        self.node_short = GenerationNode(
            node_name="test",
            model_specs=[self.sobol_model_spec],
            trial_type=Keys.SHORT_RUN,
        )

    def test_init(self) -> None:
        self.assertEqual(
            self.sobol_generation_node.model_specs, [self.sobol_model_spec]
        )
        with self.assertRaisesRegex(UserInputError, "Model keys must be unique"):
            GenerationNode(
                node_name="test",
                model_specs=[self.sobol_model_spec, self.sobol_model_spec],
            )
        mbm_specs = [
            GeneratorSpec(model_enum=Generators.BOTORCH_MODULAR),
            GeneratorSpec(
                model_enum=Generators.BOTORCH_MODULAR, model_key_override="MBM v2"
            ),
        ]
        with self.assertRaisesRegex(UserInputError, MISSING_MODEL_SELECTOR_MESSAGE):
            GenerationNode(
                node_name="test",
                model_specs=mbm_specs,
            )
        model_selector = SingleDiagnosticBestModelSelector(
            diagnostic="Fisher exact test p",
            metric_aggregation=ReductionCriterion.MEAN,
            criterion=ReductionCriterion.MIN,
        )
        node = GenerationNode(
            node_name="test",
            model_specs=mbm_specs,
            best_model_selector=model_selector,
        )
        self.assertEqual(node.model_specs, mbm_specs)
        self.assertIs(node.best_model_selector, model_selector)

    def test_input_constructor_none(self) -> None:
        self.assertEqual(self.sobol_generation_node._input_constructors, {})
        self.assertEqual(self.sobol_generation_node.input_constructors, {})

    def test_incorrect_trial_type(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "Trial type must be either"):
            GenerationNode(
                node_name="test",
                model_specs=[self.sobol_model_spec],
                trial_type="foo",
            )

    def test_init_with_trial_type(self) -> None:
        node_long = GenerationNode(
            node_name="test",
            model_specs=[self.sobol_model_spec],
            trial_type=Keys.LONG_RUN,
        )
        node_default = GenerationNode(
            node_name="test",
            model_specs=[self.sobol_model_spec],
        )
        self.assertEqual(self.node_short._trial_type, Keys.SHORT_RUN)
        self.assertEqual(node_long._trial_type, Keys.LONG_RUN)
        self.assertIsNone(node_default._trial_type)

    def test_input_constructor(self) -> None:
        node = GenerationNode(
            node_name="test",
            model_specs=[self.sobol_model_spec],
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
            self.sobol_model_spec, "fit", wraps=self.sobol_model_spec.fit
        ) as mock_model_spec_fit:
            self.sobol_generation_node._fit(
                experiment=self.branin_experiment,
                data=self.branin_data,
            )
        mock_model_spec_fit.assert_called_with(
            experiment=self.branin_experiment, data=self.branin_data
        )

    def test_gen(self) -> None:
        with patch.object(
            self.sobol_model_spec, "gen", wraps=self.sobol_model_spec.gen
        ) as mock_model_spec_gen, patch.object(
            self.sobol_model_spec, "fit", wraps=self.sobol_model_spec.fit
        ) as mock_model_spec_fit:
            gr = self.sobol_generation_node.gen(
                experiment=self.branin_experiment,
                data=self.branin_experiment.lookup_data(),
                n=1,
                pending_observations={"branin": []},
            )
            self.assertIsNotNone(gr)
            self.assertEqual(gr._model_key, self.sobol_model_spec.model_key)
            model_kwargs = gr._model_kwargs
            self.assertIsNotNone(model_kwargs)
            self.assertEqual(model_kwargs.get("init_position"), 3)
        mock_model_spec_fit.assert_called_with(
            experiment=self.branin_experiment, data=self.branin_experiment.lookup_data()
        )
        mock_model_spec_gen.assert_called_with(
            experiment=self.branin_experiment,
            data=self.branin_experiment.lookup_data(),
            n=1,
            pending_observations={"branin": []},
            fixed_features=None,
        )

    @mock_botorch_optimize
    def test_gen_with_trial_type(self) -> None:
        mbm_short = GenerationNode(
            node_name="test",
            model_specs=[
                GeneratorSpec(
                    model_enum=Generators.BOTORCH_MODULAR,
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
            node_name="test",
            model_specs=[
                GeneratorSpec(
                    model_enum=Generators.BOTORCH_MODULAR,
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
            node.model_spec_to_gen_from.model_gen_kwargs["model_gen_options"][
                Keys.ACQF_KWARGS
            ]["sampler"],
            sampler,
        )
        self.assertIsNone(sampler.base_samples)

    @mock_botorch_optimize
    def test_properties(self) -> None:
        node = GenerationNode(
            node_name="test",
            model_specs=[
                GeneratorSpec(
                    model_enum=Generators.BOTORCH_MODULAR,
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
            node.model_spec_to_gen_from.model_enum, node.model_specs[0].model_enum
        )
        self.assertEqual(
            node.model_spec_to_gen_from.model_kwargs, node.model_specs[0].model_kwargs
        )
        self.assertEqual(node.model_to_gen_from_name, "BoTorch")
        self.assertEqual(
            node.model_spec_to_gen_from.model_gen_kwargs,
            node.model_specs[0].model_gen_kwargs,
        )
        self.assertEqual(
            node.model_spec_to_gen_from.model_cv_kwargs,
            node.model_specs[0].model_cv_kwargs,
        )
        self.assertEqual(
            node.model_spec_to_gen_from.fixed_features,
            node.model_specs[0].fixed_features,
        )
        self.assertEqual(
            node.model_spec_to_gen_from.cv_results, node.model_specs[0].cv_results
        )
        self.assertEqual(
            node.model_spec_to_gen_from.diagnostics, node.model_specs[0].diagnostics
        )
        self.assertEqual(node.node_name, "test")
        self.assertEqual(node._unique_id, "test")

    def test_node_string_representation(self) -> None:
        node = GenerationNode(
            node_name="test",
            model_specs=[
                self.mbm_model_spec,
            ],
            transition_criteria=[
                MinTrials(threshold=5, only_in_statuses=[TrialStatus.RUNNING])
            ],
        )
        string_rep = str(node)

        self.assertEqual(
            string_rep,
            "GenerationNode(node_name='test', "
            "model_specs=[GeneratorSpec(model_enum=BoTorch, "
            "model_key_override=None)], "
            "transition_criteria=[MinTrials(transition_to='None')])",
        )

    def test_single_fixed_features(self) -> None:
        node = GenerationNode(
            node_name="test",
            model_specs=[
                GeneratorSpec(
                    model_enum=Generators.BOTORCH_MODULAR,
                    model_kwargs={},
                    model_gen_kwargs={
                        "n": 2,
                        "fixed_features": ObservationFeatures(parameters={"x": 0}),
                    },
                ),
            ],
        )
        self.assertEqual(
            node.model_spec_to_gen_from.fixed_features,
            ObservationFeatures(parameters={"x": 0}),
        )


class TestGenerationStep(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.model_kwargs = {"init_position": 5}
        self.sobol_generation_step = GenerationStep(
            model=Generators.SOBOL,
            num_trials=5,
            model_kwargs=self.model_kwargs,
        )
        self.model_spec = GeneratorSpec(
            # pyre-fixme[6]: For 1st param expected `ModelRegistryBase` but got
            #  `Union[typing.Callable[..., Adapter], ModelRegistryBase]`.
            model_enum=self.sobol_generation_step.model,
            model_kwargs=self.model_kwargs,
        )

    def test_init(self) -> None:
        self.assertEqual(
            self.sobol_generation_step.model_specs,
            [self.model_spec],
        )
        self.assertEqual(self.sobol_generation_step.model_name, "Sobol")

        named_generation_step = GenerationStep(
            model=Generators.SOBOL,
            num_trials=5,
            model_kwargs=self.model_kwargs,
            model_name="Custom Sobol",
        )
        self.assertEqual(named_generation_step.model_name, "Custom Sobol")

    def test_min_trials_observed(self) -> None:
        with self.assertRaisesRegex(UserInputError, "min_trials_observed > num_trials"):
            GenerationStep(
                model=Generators.SOBOL,
                num_trials=5,
                min_trials_observed=10,
                model_kwargs=self.model_kwargs,
            )

    def test_init_factory_function(self) -> None:
        generation_step = GenerationStep(model=get_sobol, num_trials=-1)
        self.assertEqual(
            generation_step.model_specs,
            [FactoryFunctionGeneratorSpec(factory_function=get_sobol)],
        )
        generation_step = GenerationStep(
            model=get_sobol, num_trials=-1, model_name="test"
        )
        self.assertEqual(
            generation_step.model_specs,
            [
                FactoryFunctionGeneratorSpec(
                    factory_function=get_sobol, model_key_override="test"
                )
            ],
        )

    def test_properties(self) -> None:
        step = self.sobol_generation_step
        self.assertEqual(step.model_spec, self.model_spec)
        self.assertEqual(step._unique_id, "-1")
        # Make sure that model_kwargs and model_gen_kwargs are synchronized
        # to the underlying model spec.
        spec = step.model_spec
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
        self.ms_mixed = GeneratorSpec(model_enum=Generators.BO_MIXED)
        self.ms_botorch = GeneratorSpec(model_enum=Generators.BOTORCH_MODULAR)

        self.mock_aggregation = MagicMock(
            side_effect=ReductionCriterion.MEAN, spec=ReductionCriterion
        )
        self.model_selection_node = GenerationNode(
            node_name="test",
            model_specs=[self.ms_mixed, self.ms_botorch],
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
    def test_pick_fitted_model_with_fit_errors(self) -> None:
        # Make model fitting error out for both specs. We should get an error.
        with patch(
            "ax.generation_strategy.model_spec.GeneratorSpec.fit",
            side_effect=RuntimeError,
        ), self.assertLogs(logger=logger, level="ERROR") as mock_logs:
            self.model_selection_node._fit(experiment=self.branin_experiment)
        self.assertEqual(len(mock_logs.records), 2)
        with self.assertRaisesRegex(ModelError, "No fitted models were found"):
            self.model_selection_node.model_spec_to_gen_from

        # Only one spec errors out.
        with patch.object(
            self.ms_mixed, "fit", side_effect=RuntimeError
        ), self.assertLogs(logger=logger, level="ERROR") as mock_logs:
            self.model_selection_node._fit(experiment=self.branin_experiment)
        self.assertEqual(len(mock_logs.records), 1)
        # Picks the model that didn't error out.
        self.assertEqual(
            self.model_selection_node.model_spec_to_gen_from, self.ms_botorch
        )
