#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from unittest.mock import MagicMock, patch

from ax.core.base_trial import TrialStatus
from ax.core.observation import ObservationFeatures
from ax.exceptions.core import UserInputError
from ax.modelbridge.best_model_selector import (
    ReductionCriterion,
    SingleDiagnosticBestModelSelector,
)
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.generation_node import (
    GenerationNode,
    GenerationStep,
    MISSING_MODEL_SELECTOR_MESSAGE,
)
from ax.modelbridge.model_spec import FactoryFunctionModelSpec, ModelSpec
from ax.modelbridge.registry import Models
from ax.modelbridge.transition_criterion import MaxTrials
from ax.utils.common.logger import get_logger
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import fast_botorch_optimize

logger: Logger = get_logger(__name__)


class TestGenerationNode(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.sobol_model_spec = ModelSpec(
            model_enum=Models.SOBOL,
            model_kwargs={"init_position": 3},
            model_gen_kwargs={"some_gen_kwarg": "some_value"},
        )
        self.sobol_generation_node = GenerationNode(
            node_name="test", model_specs=[self.sobol_model_spec]
        )
        self.branin_experiment = get_branin_experiment(with_completed_trial=True)

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
            ModelSpec(model_enum=Models.BOTORCH_MODULAR),
            ModelSpec(model_enum=Models.BOTORCH_MODULAR, model_key_override="MBM v2"),
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

    def test_fit(self) -> None:
        dat = self.branin_experiment.lookup_data()
        with patch.object(
            self.sobol_model_spec, "fit", wraps=self.sobol_model_spec.fit
        ) as mock_model_spec_fit:
            self.sobol_generation_node.fit(
                experiment=self.branin_experiment,
                data=dat,
            )
        mock_model_spec_fit.assert_called_with(
            experiment=self.branin_experiment,
            data=dat,
            search_space=None,
            optimization_config=None,
        )

    def test_gen(self) -> None:
        dat = self.branin_experiment.lookup_data()
        self.sobol_generation_node.fit(
            experiment=self.branin_experiment,
            data=dat,
        )
        with patch.object(
            self.sobol_model_spec, "gen", wraps=self.sobol_model_spec.gen
        ) as mock_model_spec_gen:
            gr = self.sobol_generation_node.gen(
                n=1, pending_observations={"branin": []}
            )
        mock_model_spec_gen.assert_called_with(n=1, pending_observations={"branin": []})
        self.assertEqual(gr._model_key, self.sobol_model_spec.model_key)
        # pyre-fixme[16]: Optional type has no attribute `get`.
        self.assertEqual(gr._model_kwargs.get("init_position"), 3)

    @fast_botorch_optimize
    def test_properties(self) -> None:
        node = GenerationNode(
            node_name="test",
            model_specs=[
                ModelSpec(
                    model_enum=Models.BOTORCH_MODULAR,
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
        self.assertIsNone(node.model_to_gen_from_name)
        dat = self.branin_experiment.lookup_data()
        node.fit(
            experiment=self.branin_experiment,
            data=dat,
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
                ModelSpec(
                    model_enum=Models.BOTORCH_MODULAR,
                    model_kwargs={},
                    model_gen_kwargs={},
                ),
            ],
            transition_criteria=[
                MaxTrials(threshold=5, only_in_statuses=[TrialStatus.RUNNING])
            ],
        )
        string_rep = str(node)
        self.assertEqual(
            string_rep,
            (
                "GenerationNode(model_specs=[ModelSpec(model_enum=BoTorch,"
                " model_kwargs={}, model_gen_kwargs={}, model_cv_kwargs={},"
                " )], node_name=test, "
                "transition_criteria=[MaxTrials({'threshold': 5, "
                "'only_in_statuses': [<enum 'TrialStatus'>.RUNNING], "
                "'not_in_statuses': None, 'transition_to': None, "
                "'block_transition_if_unmet': True, 'block_gen_if_met': False, "
                "'use_all_trials_in_exp': False, "
                "'continue_trial_generation': False})])"
            ),
        )

    def test_single_fixed_features(self) -> None:
        node = GenerationNode(
            node_name="test",
            model_specs=[
                ModelSpec(
                    model_enum=Models.BOTORCH_MODULAR,
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
            model=Models.SOBOL,
            num_trials=5,
            model_kwargs=self.model_kwargs,
        )
        self.model_spec = ModelSpec(
            # pyre-fixme[6]: For 1st param expected `ModelRegistryBase` but got
            #  `Union[typing.Callable[..., ModelBridge], ModelRegistryBase]`.
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
            model=Models.SOBOL,
            num_trials=5,
            model_kwargs=self.model_kwargs,
            model_name="Custom Sobol",
        )
        self.assertEqual(named_generation_step.model_name, "Custom Sobol")

    def test_min_trials_observed(self) -> None:
        with self.assertRaisesRegex(UserInputError, "min_trials_observed > num_trials"):
            GenerationStep(
                model=Models.SOBOL,
                num_trials=5,
                min_trials_observed=10,
                model_kwargs=self.model_kwargs,
            )

    def test_init_factory_function(self) -> None:
        generation_step = GenerationStep(model=get_sobol, num_trials=-1)
        self.assertEqual(
            generation_step.model_specs,
            [FactoryFunctionModelSpec(factory_function=get_sobol)],
        )
        generation_step = GenerationStep(
            model=get_sobol, num_trials=-1, model_name="test"
        )
        self.assertEqual(
            generation_step.model_specs,
            [
                FactoryFunctionModelSpec(
                    factory_function=get_sobol, model_key_override="test"
                )
            ],
        )

    def test_properties(self) -> None:
        self.assertEqual(self.sobol_generation_step.model_spec, self.model_spec)
        self.assertEqual(self.sobol_generation_step._unique_id, "-1")


class TestGenerationNodeWithBestModelSelector(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.branin_experiment = get_branin_experiment(
            with_batch=True, with_completed_batch=True
        )
        ms_mixed = ModelSpec(model_enum=Models.BO_MIXED)
        ms_botorch = ModelSpec(model_enum=Models.BOTORCH_MODULAR)

        self.mock_aggregation = MagicMock(
            side_effect=ReductionCriterion.MEAN, spec=ReductionCriterion
        )
        self.model_selection_node = GenerationNode(
            node_name="test",
            model_specs=[ms_mixed, ms_botorch],
            best_model_selector=SingleDiagnosticBestModelSelector(
                diagnostic="Fisher exact test p",
                metric_aggregation=self.mock_aggregation,
                criterion=ReductionCriterion.MIN,
            ),
        )

    @fast_botorch_optimize
    def test_gen(self) -> None:
        self.model_selection_node.fit(
            experiment=self.branin_experiment, data=self.branin_experiment.lookup_data()
        )
        # Check that with `ModelSelectionNode` generation from a node with
        # multiple model specs does not fail.
        gr = self.model_selection_node.gen(n=1, pending_observations={"branin": []})
        # Check that the metric aggregation function is called twice, once for each
        # model spec.
        self.assertEqual(self.mock_aggregation.call_count, 2)
        # The model specs are practically identical for this example.
        # Should pick the first one.
        self.assertEqual(gr._model_key, "BO_MIXED")

        # test model_to_gen_from_name property
        self.assertEqual(self.model_selection_node.model_to_gen_from_name, "BO_MIXED")
