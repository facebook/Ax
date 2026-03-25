# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Callable
from logging import Logger

from ax.adapter.base import Adapter
from ax.adapter.cross_validation import compute_diagnostics, cross_validate
from ax.adapter.transforms.winsorize import Winsorize
from ax.core.auxiliary import AuxiliaryExperimentPurpose
from ax.core.auxiliary_source import AuxiliarySource
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.observation import Observation
from ax.exceptions.core import AxError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.utils.common.logger import get_logger
from ax.utils.stats.model_fit_stats import (
    DIAGNOSTIC_FN_DIRECTIONS,
    ModelFitMetricDirection,
)
from botorch.exceptions.errors import ModelFittingError
from pyre_extensions import assert_is_instance

logger: Logger = get_logger(__name__)


def _mean_diagnostic(
    diagnostics: dict[str, dict[str, float]],
    eval_criterion: str,
    metric_names: list[str],
) -> float:
    """Compute the mean of ``eval_criterion`` across ``metric_names``."""
    criterion_values = diagnostics[eval_criterion]
    return sum(criterion_values[m] for m in metric_names) / len(metric_names)


def _get_winsorization_test_selector(
    adapter: Adapter,
) -> Callable[[Observation], bool] | None:
    """Return a test selector that excludes observations outside Winsorize cutoffs.

    When a model uses the Winsorize transform, observations whose raw values
    fall outside the learned clipping bounds are not meaningful test points
    because their observed values would be clipped during transformation.
    This selector keeps only observations where all metrics' means
    lie strictly within their cutoff ranges, so that cross-validation scores are
    computed on un-clipped data.

    Returns None if the adapter has no Winsorize transform or if all cutoffs
    are effectively unbounded (negative infinity to positive infinity).
    """
    if "Winsorize" not in adapter.transforms:
        return None

    winsorize_transform: Winsorize = assert_is_instance(
        adapter.transforms["Winsorize"], Winsorize
    )

    # Check if all cutoffs are effectively unbounded.
    all_unbounded = all(
        lo == float("-inf") and hi == float("inf")
        for lo, hi in winsorize_transform.cutoffs.values()
    )
    if all_unbounded:
        return None

    def test_selector(obs: Observation) -> bool:
        od = obs.data
        for i, metric_signature in enumerate(od.metric_signatures):
            cutoffs = winsorize_transform.cutoffs.get(metric_signature)
            if cutoffs is None:
                continue
            mean = od.means[i]
            if mean <= cutoffs[0] or mean >= cutoffs[1]:
                return False
        return True

    return test_selector


def _fit_and_cv(
    generation_strategy: GenerationStrategy,
    experiment: Experiment,
    data: Data,
    eval_criterion: str,
    metric_names: list[str],
) -> float:
    """Clone a GenerationStrategy, fit the appropriate node, and compute
    mean CV score.

    Uses ``GenerationStrategy.fit`` to let the GS select the correct
    node (e.g. TL vs non-TL) based on the current experiment state, then
    runs cross-validation on the best fitted adapter.
    """
    gs = generation_strategy.clone_reset()
    adapter = gs.fit(experiment=experiment, data=data)
    if adapter is None:
        raise AxError("No fitted adapter after fitting the generation node.")
    test_selector = _get_winsorization_test_selector(adapter)
    cv_results = cross_validate(adapter, untransform=False, test_selector=test_selector)
    return _mean_diagnostic(
        compute_diagnostics(cv_results), eval_criterion, metric_names
    )


def compute_task_selection_cv(
    source_experiments: list[Experiment],
    target_experiment: Experiment,
    generation_strategy: GenerationStrategy,
    target_data: Data | None = None,
    eval_criterion: str = "MSE",
    max_tasks: int = 2,
) -> list[str]:
    """Greedy forward task selection via cross-validation (RP_CV).

    Starting from a target-only model, greedily adds source tasks one at a
    time, keeping each addition only if it improves the leave-one-out
    cross-validation score on the target data.

    The metric names are extracted from the target experiment's
    ``optimization_config``. When the objective has multiple metrics
    (e.g. ``MultiObjective``), the mean ``eval_criterion`` across all
    objective metrics is used for selection.

    At each step the generation strategy is cloned and
    ``GenerationStrategy.fit`` is called so that the GS picks the
    appropriate node (TL or non-TL) based on whether auxiliary sources are
    attached to the experiment. The node is then fitted and
    cross-validated.

    The direction (minimize vs maximize) for ``eval_criterion`` is looked up
    automatically from ``DIAGNOSTIC_FN_DIRECTIONS``.

    Args:
        source_experiments: Candidate source experiments.
        target_experiment: Target experiment with attached data and an
            ``optimization_config`` whose objective defines the metrics.
        generation_strategy: A ``GenerationStrategy`` that will be cloned
            via ``clone_reset()`` before each fit. The GS should contain
            nodes that handle both the single-task (no auxiliary sources)
            and multi-task (with auxiliary sources) cases via transition
            criteria.
        target_data: Data to use for fitting and CV. If ``None``, uses
            ``target_experiment.lookup_data()``.
        eval_criterion: Diagnostic key from ``compute_diagnostics``.
            Must be a key in ``DIAGNOSTIC_FN_DIRECTIONS``.
            Defaults to ``"MSE"``.
        max_tasks: Maximum number of sources to select. Defaults to 2.

    Returns:
        Ordered list of selected source experiment names, in the order
        they were greedily added. Empty if no source improves CV.

    Raises:
        AxError: If source experiments have duplicate names.
        ValueError: If the target experiment has no data or if
            ``eval_criterion`` is not in ``DIAGNOSTIC_FN_DIRECTIONS``.
    """
    # Validate unique source names.
    source_names: list[str] = []
    for i, exp in enumerate(source_experiments):
        if not exp.has_name:
            exp.name = f"source_{i}"
        if exp.name in source_names:
            raise AxError("Source experiments must have unique names.")
        source_names.append(exp.name)

    if target_data is None:
        target_data = target_experiment.lookup_data()
    if target_data.df.empty:
        raise ValueError(
            "Target experiment has no data. Cannot perform CV task selection."
        )

    if eval_criterion not in DIAGNOSTIC_FN_DIRECTIONS:
        raise ValueError(
            f"Unknown eval_criterion '{eval_criterion}'. "
            f"Must be one of {list(DIAGNOSTIC_FN_DIRECTIONS.keys())}."
        )
    minimize = (
        DIAGNOSTIC_FN_DIRECTIONS[eval_criterion] == ModelFitMetricDirection.MINIMIZE
    )

    opt_config = target_experiment.optimization_config
    if opt_config is None:
        metric_names = list(
            set(target_experiment.metrics.keys()).intersection(
                target_data.df.metric_names.unique()
            )
        )
    else:
        metric_names = list(opt_config.metric_names)
    logger.info(f"Evaluating CV on metrics: {metric_names}")

    aux_srcs: list[AuxiliarySource] = [
        AuxiliarySource(experiment=exp) for exp in source_experiments
    ]

    # Fit base adapter (target only) and compute baseline CV score.
    logger.info("Fitting base adapter (target only) for CV baseline.")
    best_score = _fit_and_cv(
        generation_strategy=generation_strategy,
        experiment=target_experiment,
        data=target_data,
        eval_criterion=eval_criterion,
        metric_names=metric_names,
    )
    logger.info(f"Baseline mean CV {eval_criterion}: {best_score:.6f}")

    # Save original auxiliary experiments to restore later.
    original_aux = target_experiment.auxiliary_experiments_by_purpose.get(
        AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT
    )

    selected_names: list[str] = []
    selected_aux_srcs: list[AuxiliarySource] = []
    remaining_idcs: set[int] = set(range(len(aux_srcs)))

    try:
        for step in range(max_tasks):
            best_idx: int | None = None
            for i in remaining_idcs:
                candidate_aux = selected_aux_srcs + [aux_srcs[i]]
                target_experiment.auxiliary_experiments_by_purpose[
                    AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT
                ] = candidate_aux  # pyre-ignore[6]

                try:
                    score = _fit_and_cv(
                        generation_strategy=generation_strategy,
                        experiment=target_experiment,
                        data=target_data,
                        eval_criterion=eval_criterion,
                        metric_names=metric_names,
                    )
                except (AxError, ModelFittingError, RuntimeError) as e:
                    logger.warning(
                        f"CV failed for candidate '{source_names[i]}': {e}. Skipping.",
                        exc_info=True,
                    )
                    continue

                is_better = score < best_score if minimize else score > best_score
                if is_better:
                    best_score = score
                    best_idx = i

            if best_idx is None:
                logger.info(
                    f"No improvement at step {step + 1}. "
                    f"Stopping with {len(selected_names)} selected sources."
                )
                break

            selected_aux_srcs.append(aux_srcs[best_idx])
            remaining_idcs.remove(best_idx)
            selected_names.append(source_names[best_idx])
            logger.info(
                f"Step {step + 1}: selected '{source_names[best_idx]}' "
                f"(mean CV {eval_criterion}={best_score:.6f})"
            )
    finally:
        # Restore original auxiliary experiments.
        if original_aux is not None:
            target_experiment.auxiliary_experiments_by_purpose[
                AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT
            ] = original_aux
        else:
            target_experiment.auxiliary_experiments_by_purpose.pop(
                AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT, None
            )

    return selected_names
