# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections import OrderedDict
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
from ax.adapter.transfer_learning.task_selection_cv import (
    _fit_and_cv,
    _get_winsorization_test_selector,
    compute_task_selection_cv,
)
from ax.adapter.transforms.winsorize import Winsorize
from ax.core.auxiliary import AuxiliaryExperimentPurpose
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.exceptions.core import AxError
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.transition_criterion import AuxiliaryExperimentCheck


_MODULE = "ax.adapter.transfer_learning.task_selection_cv"


class ComputeTaskSelectionCVTest(TestCase):
    def _make_target_experiment(
        self, metric_names: list[str] | None = None
    ) -> MagicMock:
        if metric_names is None:
            metric_names = ["m1"]
        mock_exp = MagicMock()
        mock_exp.has_name = True
        mock_exp.name = "target"
        mock_exp.auxiliary_experiments_by_purpose = {}
        mock_data = MagicMock()
        mock_data.df.empty = False
        mock_data.df.metric_names.unique.return_value = metric_names
        mock_exp.lookup_data.return_value = mock_data
        mock_exp.metrics = {m: MagicMock() for m in metric_names}
        mock_exp.optimization_config.metrics = {m: MagicMock() for m in metric_names}
        mock_exp.optimization_config.metric_names = metric_names
        return mock_exp

    def _make_source_experiment(self, name: str) -> MagicMock:
        mock_exp = MagicMock()
        mock_exp.has_name = True
        mock_exp.name = name
        return mock_exp

    def _make_generation_strategy(self) -> MagicMock:
        """Create a mock GenerationStrategy with clone_reset/transition/fit."""
        gs = MagicMock()

        def clone_reset() -> MagicMock:
            clone = MagicMock()
            clone.current_node._fitted_adapter = MagicMock()
            clone.current_node._fitted_adapter.transforms = {}
            return clone

        gs.clone_reset.side_effect = clone_reset
        return gs

    def test_duplicate_source_names_raises(self) -> None:
        target_exp = self._make_target_experiment()
        src_a = self._make_source_experiment("same_name")
        src_b = self._make_source_experiment("same_name")

        with self.assertRaises(AxError):
            compute_task_selection_cv(
                source_experiments=[src_a, src_b],
                target_experiment=target_exp,
                generation_strategy=self._make_generation_strategy(),
            )

    def test_empty_target_data_raises(self) -> None:
        target_exp = self._make_target_experiment()
        mock_data = MagicMock()
        mock_data.df.empty = True
        target_exp.lookup_data.return_value = mock_data

        with self.assertRaises(ValueError):
            compute_task_selection_cv(
                source_experiments=[self._make_source_experiment("s")],
                target_experiment=target_exp,
                generation_strategy=self._make_generation_strategy(),
            )

    def test_no_optimization_config_uses_experiment_metrics(self) -> None:
        target_exp = self._make_target_experiment()
        target_exp.optimization_config = None

        with (
            patch(f"{_MODULE}.cross_validate"),
            patch(f"{_MODULE}.compute_diagnostics") as mock_diag,
        ):
            mock_diag.return_value = {"MSE": {"m1": 1.0}}

            result = compute_task_selection_cv(
                source_experiments=[],
                target_experiment=target_exp,
                generation_strategy=self._make_generation_strategy(),
            )

        self.assertEqual(result, [])

    def test_empty_source_list(self) -> None:
        target_exp = self._make_target_experiment()

        with (
            patch(f"{_MODULE}.cross_validate"),
            patch(f"{_MODULE}.compute_diagnostics") as mock_diag,
        ):
            mock_diag.return_value = {"MSE": {"m1": 1.0}}

            result = compute_task_selection_cv(
                source_experiments=[],
                target_experiment=target_exp,
                generation_strategy=self._make_generation_strategy(),
            )

        self.assertEqual(result, [])

    def test_selects_improving_source(self) -> None:
        target_exp: MagicMock = self._make_target_experiment()
        source_good = self._make_source_experiment("good")
        source_bad = self._make_source_experiment("bad")

        def diagnostics_side_effect(cv_results: object) -> dict[str, dict[str, float]]:
            aux = target_exp.auxiliary_experiments_by_purpose.get(
                AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT, []
            )
            names = frozenset(a.experiment.name for a in aux)
            scores: dict[frozenset[str], float] = {
                frozenset(): 1.0,
                frozenset(["good"]): 0.5,
                frozenset(["bad"]): 1.5,
                frozenset(["good", "bad"]): 0.8,
            }
            return {"MSE": {"m1": scores.get(names, 2.0)}}

        with (
            patch(f"{_MODULE}.AuxiliarySource") as mock_aux_cls,
            patch(f"{_MODULE}.cross_validate"),
            patch(f"{_MODULE}.compute_diagnostics") as mock_diag,
        ):
            mock_aux_cls.side_effect = lambda experiment: MagicMock(
                experiment=experiment
            )
            mock_diag.side_effect = diagnostics_side_effect

            result = compute_task_selection_cv(
                source_experiments=[source_good, source_bad],
                target_experiment=target_exp,
                generation_strategy=self._make_generation_strategy(),
            )

        self.assertEqual(result, ["good"])

    def test_no_sources_selected_when_none_improve(self) -> None:
        target_exp: MagicMock = self._make_target_experiment()
        src_a = self._make_source_experiment("a")
        src_b = self._make_source_experiment("b")

        def diagnostics_side_effect(cv_results: object) -> dict[str, dict[str, float]]:
            aux = target_exp.auxiliary_experiments_by_purpose.get(
                AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT, []
            )
            if not aux:
                return {"MSE": {"m1": 1.0}}
            return {"MSE": {"m1": 2.0}}

        with (
            patch(f"{_MODULE}.AuxiliarySource") as mock_aux_cls,
            patch(f"{_MODULE}.cross_validate"),
            patch(f"{_MODULE}.compute_diagnostics") as mock_diag,
        ):
            mock_aux_cls.side_effect = lambda experiment: MagicMock(
                experiment=experiment
            )
            mock_diag.side_effect = diagnostics_side_effect

            result = compute_task_selection_cv(
                source_experiments=[src_a, src_b],
                target_experiment=target_exp,
                generation_strategy=self._make_generation_strategy(),
            )

        self.assertEqual(result, [])

    def test_max_tasks_limit(self) -> None:
        target_exp = self._make_target_experiment()
        sources = [self._make_source_experiment(f"s{i}") for i in range(4)]

        call_count: list[int] = [0]

        def diagnostics_side_effect(cv_results: object) -> dict[str, dict[str, float]]:
            score = 1.0 - 0.1 * call_count[0]
            call_count[0] += 1
            return {"MSE": {"m1": score}}

        with (
            patch(f"{_MODULE}.AuxiliarySource") as mock_aux_cls,
            patch(f"{_MODULE}.cross_validate"),
            patch(f"{_MODULE}.compute_diagnostics") as mock_diag,
        ):
            mock_aux_cls.side_effect = lambda experiment: MagicMock(
                experiment=experiment
            )
            mock_diag.side_effect = diagnostics_side_effect

            result = compute_task_selection_cv(
                source_experiments=sources,
                target_experiment=target_exp,
                generation_strategy=self._make_generation_strategy(),
                max_tasks=2,
            )

        self.assertEqual(len(result), 2)

    def test_restores_auxiliary_experiments(self) -> None:
        target_exp = self._make_target_experiment()
        original_aux = [MagicMock()]
        target_exp.auxiliary_experiments_by_purpose[
            AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT
        ] = original_aux

        with (
            patch(f"{_MODULE}.AuxiliarySource") as mock_aux_cls,
            patch(f"{_MODULE}.cross_validate"),
            patch(f"{_MODULE}.compute_diagnostics") as mock_diag,
        ):
            mock_aux_cls.side_effect = lambda experiment: MagicMock(
                experiment=experiment
            )
            mock_diag.return_value = {"MSE": {"m1": 1.0}}

            compute_task_selection_cv(
                source_experiments=[self._make_source_experiment("s")],
                target_experiment=target_exp,
                generation_strategy=self._make_generation_strategy(),
            )

        self.assertIs(
            target_exp.auxiliary_experiments_by_purpose[
                AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT
            ],
            original_aux,
        )

    def test_restores_when_no_original_aux(self) -> None:
        target_exp = self._make_target_experiment()
        self.assertNotIn(
            AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT,
            target_exp.auxiliary_experiments_by_purpose,
        )

        with (
            patch(f"{_MODULE}.AuxiliarySource") as mock_aux_cls,
            patch(f"{_MODULE}.cross_validate"),
            patch(f"{_MODULE}.compute_diagnostics") as mock_diag,
        ):
            mock_aux_cls.side_effect = lambda experiment: MagicMock(
                experiment=experiment
            )
            mock_diag.return_value = {"MSE": {"m1": 0.5}}

            compute_task_selection_cv(
                source_experiments=[self._make_source_experiment("s")],
                target_experiment=target_exp,
                generation_strategy=self._make_generation_strategy(),
            )

        self.assertNotIn(
            AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT,
            target_exp.auxiliary_experiments_by_purpose,
        )

    def test_maximize_criterion(self) -> None:
        target_exp: MagicMock = self._make_target_experiment()
        source_good = self._make_source_experiment("good")
        source_bad = self._make_source_experiment("bad")

        def diagnostics_side_effect(cv_results: object) -> dict[str, dict[str, float]]:
            aux = target_exp.auxiliary_experiments_by_purpose.get(
                AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT, []
            )
            names = frozenset(a.experiment.name for a in aux)
            scores: dict[frozenset[str], float] = {
                frozenset(): 1.0,
                frozenset(["good"]): 1.5,
                frozenset(["bad"]): 0.5,
                frozenset(["good", "bad"]): 1.2,
            }
            return {"Log likelihood": {"m1": scores.get(names, 0.0)}}

        with (
            patch(f"{_MODULE}.AuxiliarySource") as mock_aux_cls,
            patch(f"{_MODULE}.cross_validate"),
            patch(f"{_MODULE}.compute_diagnostics") as mock_diag,
        ):
            mock_aux_cls.side_effect = lambda experiment: MagicMock(
                experiment=experiment
            )
            mock_diag.side_effect = diagnostics_side_effect

            result = compute_task_selection_cv(
                source_experiments=[source_good, source_bad],
                target_experiment=target_exp,
                generation_strategy=self._make_generation_strategy(),
                eval_criterion="Log likelihood",
            )

        self.assertEqual(result, ["good"])

    def test_invalid_eval_criterion_raises(self) -> None:
        target_exp = self._make_target_experiment()

        with self.assertRaises(ValueError):
            compute_task_selection_cv(
                source_experiments=[self._make_source_experiment("s")],
                target_experiment=target_exp,
                generation_strategy=self._make_generation_strategy(),
                eval_criterion="InvalidMetric",
            )

    def test_assigns_default_names(self) -> None:
        target_exp = self._make_target_experiment()
        src = MagicMock()
        src.has_name = False

        with (
            patch(f"{_MODULE}.AuxiliarySource") as mock_aux_cls,
            patch(f"{_MODULE}.cross_validate"),
            patch(f"{_MODULE}.compute_diagnostics") as mock_diag,
        ):
            mock_aux_cls.side_effect = lambda experiment: MagicMock(
                experiment=experiment
            )
            mock_diag.side_effect = [
                {"MSE": {"m1": 1.0}},
                {"MSE": {"m1": 0.5}},
            ]

            result = compute_task_selection_cv(
                source_experiments=[src],
                target_experiment=target_exp,
                generation_strategy=self._make_generation_strategy(),
            )

        self.assertEqual(src.name, "source_0")
        self.assertEqual(result, ["source_0"])

    def _make_tl_generation_strategy(self) -> GenerationStrategy:
        """Create a GS with two nodes: 'BoTorch' and 'TL'.

        Transitions from BoTorch -> TL when TRANSFERABLE_EXPERIMENT aux
        sources are present, and from TL -> BoTorch when they are absent.
        Both nodes use mock GeneratorSpecs.
        """
        mock_spec = MagicMock()
        mock_spec.generator_key = "BoTorch"
        tl_mock_spec = MagicMock()
        tl_mock_spec.generator_key = "BOTL"

        botorch_node = GenerationNode(
            name="BoTorch",
            generator_specs=[mock_spec],
            transition_criteria=[
                AuxiliaryExperimentCheck(
                    transition_to="TL",
                    auxiliary_experiment_purposes_to_include=[
                        AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT,
                    ],
                ),
            ],
        )
        tl_node = GenerationNode(
            name="TL",
            generator_specs=[tl_mock_spec],
            transition_criteria=[
                AuxiliaryExperimentCheck(
                    transition_to="BoTorch",
                    auxiliary_experiment_purposes_to_exclude=[
                        AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT,
                    ],
                ),
            ],
        )
        return GenerationStrategy(name="test_gs", nodes=[botorch_node, tl_node])

    def test_fit_and_cv_uses_botorch_node_without_aux_sources(self) -> None:
        """Without aux sources, _fit_and_cv should stay on the BoTorch node."""
        target_exp = self._make_target_experiment()
        mock_data = target_exp.lookup_data.return_value
        gs = self._make_tl_generation_strategy()

        node_names: list[str] = []

        def tracking_fit(node_self: GenerationNode, **kwargs: object) -> None:
            node_names.append(node_self._name)

        with (
            patch.object(GenerationNode, "_fit", tracking_fit),
            patch(f"{_MODULE}.cross_validate"),
            patch(f"{_MODULE}.compute_diagnostics") as mock_diag,
        ):
            mock_diag.return_value = {"MSE": {"m1": 0.5}}

            _fit_and_cv(
                generation_strategy=gs,
                experiment=target_exp,
                data=mock_data,
                eval_criterion="MSE",
                metric_names=["m1"],
            )

        self.assertEqual(node_names, ["BoTorch"])

    def test_fit_and_cv_uses_tl_node_with_aux_sources(self) -> None:
        """With TRANSFERABLE_EXPERIMENT aux sources, _fit_and_cv should
        transition to the TL node."""
        target_exp = self._make_target_experiment()
        target_exp.auxiliary_experiments_by_purpose[
            AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT
        ] = [MagicMock()]
        mock_data = target_exp.lookup_data.return_value
        gs = self._make_tl_generation_strategy()

        node_names: list[str] = []

        def tracking_fit(node_self: GenerationNode, **kwargs: object) -> None:
            node_names.append(node_self._name)

        with (
            patch.object(GenerationNode, "_fit", tracking_fit),
            patch(f"{_MODULE}.cross_validate"),
            patch(f"{_MODULE}.compute_diagnostics") as mock_diag,
        ):
            mock_diag.return_value = {"MSE": {"m1": 0.5}}

            _fit_and_cv(
                generation_strategy=gs,
                experiment=target_exp,
                data=mock_data,
                eval_criterion="MSE",
                metric_names=["m1"],
            )

        self.assertEqual(node_names, ["TL"])

    def test_baseline_uses_botorch_candidate_uses_tl(self) -> None:
        """In compute_task_selection_cv, the baseline (no aux sources)
        should use the BoTorch node, and candidate evaluation (with aux
        sources) should use the TL node."""
        target_exp: MagicMock = self._make_target_experiment()
        source = self._make_source_experiment("src")
        gs = self._make_tl_generation_strategy()

        node_names: list[str] = []

        def tracking_fit(node_self: GenerationNode, **kwargs: object) -> None:
            node_names.append(node_self._name)

        call_count: list[int] = [0]

        def diagnostics_side_effect(
            cv_results: object,
        ) -> dict[str, dict[str, float]]:
            score = 1.0 - 0.1 * call_count[0]
            call_count[0] += 1
            return {"MSE": {"m1": score}}

        with (
            patch.object(GenerationNode, "_fit", tracking_fit),
            patch(f"{_MODULE}.AuxiliarySource") as mock_aux_cls,
            patch(f"{_MODULE}.cross_validate"),
            patch(f"{_MODULE}.compute_diagnostics") as mock_diag,
        ):
            mock_aux_cls.side_effect = lambda experiment: MagicMock(
                experiment=experiment
            )
            mock_diag.side_effect = diagnostics_side_effect

            compute_task_selection_cv(
                source_experiments=[source],
                target_experiment=target_exp,
                generation_strategy=gs,
            )

        # First fit is baseline (BoTorch), second is candidate (TL).
        self.assertEqual(node_names, ["BoTorch", "TL"])


class GetWinsorizationTestSelectorTest(TestCase):
    def _make_observation(
        self, metric_signatures: list[str], means: list[float]
    ) -> Observation:
        k = len(metric_signatures)
        return Observation(
            features=ObservationFeatures(parameters={"x": 1.0}),
            data=ObservationData(
                metric_signatures=metric_signatures,
                means=np.array(means),
                covariance=np.zeros((k, k)),
            ),
        )

    def test_returns_none_when_no_winsorize_transform(self) -> None:
        adapter = MagicMock()
        adapter.transforms = OrderedDict()
        self.assertIsNone(_get_winsorization_test_selector(adapter))

    def test_returns_none_when_all_cutoffs_infinite(self) -> None:
        adapter = MagicMock()
        winsorize = MagicMock(spec=Winsorize)
        winsorize.cutoffs = {"m1": (float("-inf"), float("inf"))}
        adapter.transforms = OrderedDict({"Winsorize": winsorize})
        self.assertIsNone(_get_winsorization_test_selector(adapter))

    def test_excludes_observation_at_lower_cutoff(self) -> None:
        adapter = MagicMock()
        winsorize = MagicMock(spec=Winsorize)
        winsorize.cutoffs = {"m1": (0.5, float("inf"))}
        adapter.transforms = OrderedDict({"Winsorize": winsorize})
        selector = _get_winsorization_test_selector(adapter)
        assert selector is not None

        obs_clipped = self._make_observation(["m1"], [0.5])
        self.assertFalse(selector(obs_clipped))

        obs_below = self._make_observation(["m1"], [0.3])
        self.assertFalse(selector(obs_below))

        obs_ok = self._make_observation(["m1"], [1.0])
        self.assertTrue(selector(obs_ok))

    def test_excludes_observation_at_upper_cutoff(self) -> None:
        adapter = MagicMock()
        winsorize = MagicMock(spec=Winsorize)
        winsorize.cutoffs = {"m1": (float("-inf"), 2.0)}
        adapter.transforms = OrderedDict({"Winsorize": winsorize})
        selector = _get_winsorization_test_selector(adapter)
        assert selector is not None

        obs_clipped = self._make_observation(["m1"], [2.0])
        self.assertFalse(selector(obs_clipped))

        obs_above = self._make_observation(["m1"], [2.5])
        self.assertFalse(selector(obs_above))

        obs_ok = self._make_observation(["m1"], [1.0])
        self.assertTrue(selector(obs_ok))

    def test_includes_observation_for_unclipped_metric(self) -> None:
        adapter = MagicMock()
        winsorize = MagicMock(spec=Winsorize)
        winsorize.cutoffs = {"m1": (0.5, 2.0)}
        adapter.transforms = OrderedDict({"Winsorize": winsorize})
        selector = _get_winsorization_test_selector(adapter)
        assert selector is not None

        obs = self._make_observation(["m_other"], [100.0])
        self.assertTrue(selector(obs))

    def test_multi_metric_excludes_if_any_clipped(self) -> None:
        adapter = MagicMock()
        winsorize = MagicMock(spec=Winsorize)
        winsorize.cutoffs = {"m1": (0.0, 10.0), "m2": (float("-inf"), 5.0)}
        adapter.transforms = OrderedDict({"Winsorize": winsorize})
        selector = _get_winsorization_test_selector(adapter)
        assert selector is not None

        # m1 ok, m2 at upper cutoff -> excluded
        obs = self._make_observation(["m1", "m2"], [5.0, 5.0])
        self.assertFalse(selector(obs))

        # Both ok
        obs_ok = self._make_observation(["m1", "m2"], [5.0, 3.0])
        self.assertTrue(selector(obs_ok))
