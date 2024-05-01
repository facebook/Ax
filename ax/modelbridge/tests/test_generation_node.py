#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from unittest.mock import patch, PropertyMock

from ax.core.base_trial import TrialStatus
from ax.core.observation import ObservationFeatures
from ax.exceptions.core import UserInputError
from ax.modelbridge.cross_validation import (
    DiagnosticCriterion,
    MetricAggregation,
    SingleDiagnosticBestModelSelector,
)
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.generation_node import GenerationNode, GenerationStep
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

    def test_gen_validates_one_model_spec(self) -> None:
        generation_node = GenerationNode(
            node_name="test",
            model_specs=[self.sobol_model_spec, self.sobol_model_spec],
        )
        # Base generation node can only handle one model spec at the moment
        # (this might change in the future), so it should raise a `NotImplemented
        # Error` if we attempt to generate from a generation node that has
        # more than one model spec. Note that the check is done in `gen` and
        # not in the constructor to make `GenerationNode` mode convenient to
        # subclass.
        with self.assertRaises(NotImplementedError):
            generation_node.gen()

    @fast_botorch_optimize
    def test_properties(self) -> None:
        node = GenerationNode(
            node_name="test",
            model_specs=[
                ModelSpec(
                    model_enum=Models.GPEI,
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
        self.assertEqual(node.model_to_gen_from_name, "GPEI")
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
                    model_enum=Models.GPEI,
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
                "GenerationNode(model_specs=[ModelSpec(model_enum=GPEI,"
                " model_kwargs={}, model_gen_kwargs={}, model_cv_kwargs={},"
                " )], node_name=test, "
                "transition_criteria=[MaxTrials({'threshold': 5, "
                "'only_in_statuses': [<enum 'TrialStatus'>.RUNNING], "
                "'not_in_statuses': None, 'transition_to': None, "
                "'block_transition_if_unmet': True, 'block_gen_if_met': False, "
                "'use_all_trials_in_exp': False})])"
            ),
        )

    def test_single_fixed_features(self) -> None:
        node = GenerationNode(
            node_name="test",
            model_specs=[
                ModelSpec(
                    model_enum=Models.GPEI,
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
            model_gen_kwargs=None,
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

        with patch.object(
            ModelSpec, "model_key", new=PropertyMock(side_effect=TypeError)
        ):
            unknown_generation_step = GenerationStep(
                model=Models.SOBOL,
                num_trials=5,
                model_kwargs=self.model_kwargs,
            )
        self.assertEqual(unknown_generation_step.model_name, "Unknown ModelSpec")

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
            [
                FactoryFunctionModelSpec(
                    factory_function=get_sobol,
                    model_kwargs=None,
                    model_gen_kwargs=None,
                )
            ],
        )

    def test_properties(self) -> None:
        self.assertEqual(self.sobol_generation_step.model_spec, self.model_spec)
        self.assertEqual(self.sobol_generation_step._unique_id, "-1")


class TestGenerationNodeWithBestModelSelector(TestCase):
    @fast_botorch_optimize
    def setUp(self) -> None:
        super().setUp()
        self.branin_experiment = get_branin_experiment()
        sobol = Models.SOBOL(search_space=self.branin_experiment.search_space)
        sobol_run = sobol.gen(n=20)
        self.branin_experiment.new_batch_trial().add_generator_run(
            sobol_run
        ).run().mark_completed()
        data = self.branin_experiment.fetch_data()

        ms_gpei = ModelSpec(model_enum=Models.GPEI)
        ms_gpei.fit(experiment=self.branin_experiment, data=data)

        ms_botorch = ModelSpec(model_enum=Models.BOTORCH_MODULAR)
        ms_botorch.fit(experiment=self.branin_experiment, data=data)

        self.fitted_model_specs = [ms_gpei, ms_botorch]

        self.model_selection_node = GenerationNode(
            node_name="test",
            model_specs=self.fitted_model_specs,
            best_model_selector=SingleDiagnosticBestModelSelector(
                diagnostic="Fisher exact test p",
                # pyre-fixme[6]: For 2nd param expected `DiagnosticCriterion` but
                #  got `MetricAggregation`.
                criterion=MetricAggregation.MEAN,
                # pyre-fixme[6]: For 3rd param expected `MetricAggregation` but got
                #  `DiagnosticCriterion`.
                metric_aggregation=DiagnosticCriterion.MIN,
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
        # Currently, `ModelSelectionNode` should just pick the first model
        # spec as the one to generate from.
        # TODO[adamobeng]: Test correct behavior here when implemented.
        self.assertEqual(gr._model_key, "GPEI")

        # test model_to_gen_from_name property
        self.assertEqual(self.model_selection_node.model_to_gen_from_name, "GPEI")
