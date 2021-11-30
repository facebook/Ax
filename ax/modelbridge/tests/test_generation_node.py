#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

from ax.modelbridge.factory import get_sobol
from ax.modelbridge.generation_node import (
    GenerationNode,
    ModelSelectionGenerationNode,
    GenerationStep,
)
from ax.modelbridge.model_spec import ModelSpec, FactoryFunctionModelSpec
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class TestGenerationNode(TestCase):
    def setUp(self):
        self.sobol_model_spec = ModelSpec(
            model_enum=Models.SOBOL,
            model_kwargs={"init_position": 3},
            model_gen_kwargs={"some_gen_kwarg": "some_value"},
        )
        self.sobol_generation_node = GenerationNode(model_specs=[self.sobol_model_spec])
        self.branin_experiment = get_branin_experiment(with_completed_trial=True)

    def test_init(self):
        self.assertEqual(
            self.sobol_generation_node.model_specs, [self.sobol_model_spec]
        )

    def test_fit(self):
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

    def test_gen(self):
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
        self.assertEqual(gr._model_kwargs.get("init_position"), 3)

    def test_gen_validates_one_model_spec(self):
        generation_node = GenerationNode(
            model_specs=[self.sobol_model_spec, self.sobol_model_spec]
        )
        # Base generation node can only handle one model spec at the moment
        # (this might change in the future), so it should raise a `NotImplemented
        # Error` if we attempt to generate from a generation node that has
        # more than one model spec. Note that the check is done in `gen` and
        # not in the constructor to make `GenerationNode` mode convenient to
        # subclass.
        with self.assertRaises(NotImplementedError):
            generation_node.gen()


class TestGenerationStep(TestCase):
    def setUp(self):
        self.model_kwargs = ({"init_position": 5},)
        self.sobol_generation_step = GenerationStep(
            model=Models.SOBOL,
            num_trials=5,
            model_kwargs=self.model_kwargs,
        )
        self.model_spec = ModelSpec(
            model_enum=self.sobol_generation_step.model,
            model_kwargs=self.model_kwargs,
            model_gen_kwargs=None,
        )

    def test_init(self):
        self.assertEqual(
            self.sobol_generation_step.model_specs,
            [self.model_spec],
        )

    def test_init_factory_function(self):
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

    def test_properties(self):
        self.assertEqual(self.sobol_generation_step.model_spec, self.model_spec)
        self.assertEqual(self.sobol_generation_step.model_name, "Sobol")
        self.assertEqual(self.sobol_generation_step._unique_id, "-1")


class TestModelSelectionGenerationNode(TestCase):
    def setUp(self):
        self.sobol_model_spec = ModelSpec(
            model_enum=Models.SOBOL,
            model_kwargs={"init_position": 3},
            model_gen_kwargs={"some_gen_kwarg": "some_value"},
        )
        self.uniform_model_spec = ModelSpec(
            model_enum=Models.UNIFORM,
        )
        self.model_selection_generation_node = ModelSelectionGenerationNode(
            model_specs=[self.uniform_model_spec, self.sobol_model_spec]
        )
        self.branin_experiment = get_branin_experiment(with_completed_trial=True)

    def test_gen(self):
        self.model_selection_generation_node.fit(
            experiment=self.branin_experiment, data=self.branin_experiment.lookup_data()
        )
        # Check that with `ModelSelectionGenerationNode` generation from a node with
        # multiple model specs does not fail.
        gr = self.model_selection_generation_node.gen(
            n=1, pending_observations={"branin": []}
        )
        # Currently, `ModelSelectionGenerationNode` should just pick the first model
        # spec as the one to generate from.
        # TODO[adamobeng]: Test correct behavior here when implemented.
        self.assertEqual(gr._model_key, "Uniform")
