#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
import torch
from ax.analysis.analysis import AnalysisCard
from ax.analysis.plotly.interaction import (
    generate_interaction_component,
    generate_main_effect_component,
    get_model_kwargs,
    InteractionPlot,
    TOP_K_TOO_LARGE_ERROR,
)
from ax.exceptions.core import DataRequiredError, UserInputError

from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import mock_botorch_optimize
from botorch.models.kernels.orthogonal_additive_kernel import OrthogonalAdditiveKernel
from gpytorch.kernels import RBFKernel
from plotly import graph_objects as go


class InteractionTest(TestCase):
    def test_interaction_get_model_kwargs(self) -> None:
        kwargs = get_model_kwargs(
            num_parameters=3,
            use_interaction=False,
            torch_device=torch.device("cpu"),
        )
        self.assertEqual(kwargs["covar_module_class"], OrthogonalAdditiveKernel)
        covar_module_options = kwargs["covar_module_options"]
        self.assertIsInstance(covar_module_options["base_kernel"], RBFKernel)
        self.assertEqual(covar_module_options["dim"], 3)

        # Checks that we can retrieve the modelbridge that has interaction terms
        kwargs = get_model_kwargs(
            num_parameters=5,
            use_interaction=True,
            torch_device=torch.device("cpu"),
        )
        self.assertEqual(kwargs["covar_module_class"], OrthogonalAdditiveKernel)
        self.assertIsInstance(kwargs["covar_module_options"]["base_kernel"], RBFKernel)

    @mock_botorch_optimize
    def test_interaction_analysis_without_components(self) -> None:
        exp = get_branin_experiment(with_completed_trial=True)
        analysis = InteractionPlot(
            metric_name="branin",
            fit_interactions=False,
            num_mc_samples=11,
        )
        card = analysis.compute(experiment=exp)
        self.assertIsInstance(card, AnalysisCard)
        self.assertIsInstance(card.blob, str)
        self.assertIsInstance(card.df, pd.DataFrame)
        self.assertEqual(
            card.name,
            "Interaction Analysis",
        )
        self.assertEqual(
            card.title,
            "Feature Importance Analysis for branin",
        )
        self.assertEqual(
            card.subtitle,
            "Displays the most important features for branin by order of importance.",
        )

        # with interaction terms
        analysis = InteractionPlot(
            metric_name="branin",
            fit_interactions=True,
            num_mc_samples=11,
        )
        card = analysis.compute(experiment=exp)
        self.assertIsInstance(card, AnalysisCard)
        self.assertIsInstance(card.blob, str)
        self.assertIsInstance(card.df, pd.DataFrame)
        self.assertEqual(len(card.df), 3)
        self.assertEqual(
            card.subtitle,
            "Displays the most important features for branin by order of importance.",
        )

        with self.assertRaisesRegex(UserInputError, TOP_K_TOO_LARGE_ERROR.format("7")):
            InteractionPlot(metric_name="branin", top_k=7, display_components=True)

        analysis = InteractionPlot(metric_name="branout", fit_interactions=False)
        with self.assertRaisesRegex(
            DataRequiredError, "StandardizeY` transform requires non-empty data."
        ):
            analysis.compute(experiment=exp)

    @mock_botorch_optimize
    def test_interaction_with_components(self) -> None:
        exp = get_branin_experiment(with_completed_trial=True)
        analysis = InteractionPlot(
            metric_name="branin",
            fit_interactions=True,
            display_components=True,
            num_mc_samples=11,
        )
        card = analysis.compute(experiment=exp)
        self.assertIsInstance(card, AnalysisCard)
        self.assertIsInstance(card.blob, str)
        self.assertIsInstance(card.df, pd.DataFrame)
        self.assertEqual(len(card.df), 3)

        analysis = InteractionPlot(
            metric_name="branin",
            fit_interactions=True,
            display_components=True,
            top_k=2,
            num_mc_samples=11,
        )
        card = analysis.compute(experiment=exp)
        self.assertIsInstance(card, AnalysisCard)
        self.assertEqual(len(card.df), 2)
        analysis = InteractionPlot(
            metric_name="branin",
            fit_interactions=True,
            display_components=True,
            model_fit_seed=999,
            num_mc_samples=11,
        )
        card = analysis.compute(experiment=exp)
        self.assertIsInstance(card, AnalysisCard)
        self.assertEqual(len(card.df), 3)

    @mock_botorch_optimize
    def test_generate_main_effect_component(self) -> None:
        exp = get_branin_experiment(with_completed_trial=True)
        analysis = InteractionPlot(
            metric_name="branin",
            fit_interactions=True,
            display_components=True,
            num_mc_samples=11,
        )
        density = 13
        model = analysis.get_model(experiment=exp)
        comp, _, _ = generate_main_effect_component(
            model=model,
            component="x1",
            metric="branin",
            density=density,
        )
        self.assertIsInstance(comp, go.Scatter)
        self.assertEqual(comp["x"].shape, (density,))
        self.assertEqual(comp["y"].shape, (density,))
        self.assertEqual(comp["name"], "x1")

        with self.assertRaisesRegex(KeyError, "braninandout"):
            generate_main_effect_component(
                model=model,
                component="x1",
                metric="braninandout",
                density=density,
            )

    @mock_botorch_optimize
    def test_generate_interaction_component(self) -> None:
        exp = get_branin_experiment(with_completed_trial=True)
        analysis = InteractionPlot(
            metric_name="branin",
            fit_interactions=True,
            display_components=True,
            num_mc_samples=11,
        )
        density = 3
        model = analysis.get_model(experiment=exp)
        comp, _, _ = generate_interaction_component(
            model=model,
            component_x="x1",
            component_y="x2",
            metric="branin",
            density=density,
        )
        self.assertIsInstance(comp, go.Contour)
        self.assertEqual(comp["x"].shape, (density,))
        self.assertEqual(comp["y"].shape, (density,))
        self.assertEqual(comp["z"].shape, (density, density))
        self.assertEqual(comp["name"], "x1 & x2")

        with self.assertRaisesRegex(KeyError, "braninandout"):
            generate_interaction_component(
                model=model,
                component_x="x1",
                component_y="x2",
                metric="braninandout",
                density=density,
            )
