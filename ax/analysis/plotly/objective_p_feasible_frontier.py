# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from typing import Mapping, Sequence

from ax.adapter.base import Adapter
from ax.adapter.torch import TorchAdapter
from ax.analysis.analysis import Analysis
from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
from ax.analysis.plotly.scatter import ScatterPlot
from ax.analysis.utils import extract_relevant_adapter
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import ScalarizedOutcomeConstraint
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import AxError, UnsupportedError, UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.generators.torch.botorch_modular.multi_acquisition import MultiAcquisition
from ax.utils.common.logger import get_logger
from botorch.acquisition.analytic import LogProbabilityOfFeasibility, PosteriorMean
from pyre_extensions import override

logger: Logger = get_logger(__name__)
OBJ_PFEAS_CARDGROUP_SUBTITLE = (
    "This plot shows <b>newly generated arms</b> with optimal trade-offs between"
    " <b>Ax model-estimated effect on the objective (x-axis)</b> and <b>Ax-model"
    " estimated probability of satisfying the constraints (y-axis)</b>. This plot"
    " is useful for understanding: 1) how tight the constraints are (sometimes the"
    " constraints can be configured too conservatively, making it difficult to find"
    " an arm that improves the objective(s) while satisfying the constraints),"
    " 2) how much headroom there is with the current optimization configuration "
    " (objective(s) and constraints). <b>If arms that are likely feasible (y-axis),"
    " do not improve your objective enough, revisiting your optimization config and"
    " relaxing the constraints may be helpful.</b> This analysis can be computed"
    " adhoc in a notebook environment, and will change with modifications to the"
    " optimization config, so you can understand the potential impact of optimization"
    " config modifications prior to running another iteration. Get in touch with the"
    " Ax developers for pointers on including these arms in a trial or running this"
    " via a notebook."
)


class ObjectivePFeasibleFrontierPlot(Analysis):
    """
    Plotly Scatter plot for the objective vs p(feasible). Each arm is represented by
    a single point with 95% confidence intervals if the data is available. Effects are
    the predicted effects using a model.
    """

    def __init__(
        self,
        relativize: bool = False,
        trial_index: int | None = None,
        trial_statuses: Sequence[TrialStatus] | None = None,
        additional_arms: Sequence[Arm] | None = None,
        labels: Mapping[str, str] | None = None,
        show_pareto_frontier: bool = False,
        num_points_to_generate: int = 100,
    ) -> None:
        """
        Args:
            relativize: Whether to relativize the effects of each arm against the status
                quo arm. If multiple status quo arms are present, relativize each arm
                against the status quo arm from the same trial.
            trial_index: If present, only use arms from the trial with the given index.
            additional_arms: If present, include these arms in the plot in addition to
                the arms in the experiment. These arms will be marked as belonging to a
                trial with index -1.
            labels: A mapping from metric names to labels to use in the plot. If a label
                is not provided for a metric, the metric name will be used.
            show_pareto_frontier: Whether to draw a line representing the Pareto
                frontier for the two metrics on the plot.
            num_points_to_generate: The number of points to generate on the frontier.
                Ideally this should be sufficiently large to provide a frontier with
                reasonably good coverage.
        """
        self.use_model_predictions = True
        self.relativize = relativize
        self.trial_index = trial_index
        self.trial_statuses = trial_statuses
        self._additional_arms: list[Arm] = (
            [] if additional_arms is None else list(additional_arms)
        )
        self.additional_arms = additional_arms
        self.labels: dict[str, str] = {**labels} if labels is not None else {}
        self.labels.setdefault("p_feasible", "% Chance of Satisfying the Constraints")
        self.show_pareto_frontier = show_pareto_frontier
        self.num_points_to_generate = num_points_to_generate

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> PlotlyAnalysisCard:
        if experiment is None:
            raise UserInputError(
                "ObjectivePFeasibleFrontierPlot requires an Experiment."
            )
        if experiment.optimization_config is None:
            raise UserInputError("Optimization_config must be set to compute frontier.")
        elif isinstance(
            experiment.optimization_config, MultiObjectiveOptimizationConfig
        ):
            raise UnsupportedError("Multi-objective optimization is not supported.")
        elif len(experiment.optimization_config.outcome_constraints) == 0:
            raise UserInputError(
                "Plotting the objective-p(feasible) frontier requires at least one "
                "outcome constraint."
            )
        elif any(
            isinstance(oc, ScalarizedOutcomeConstraint)
            for oc in experiment.optimization_config.outcome_constraints
        ):
            raise UnsupportedError(
                "Scalarized outcome constraints are not supported yet."
            )
        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        if not isinstance(relevant_adapter, TorchAdapter) or not isinstance(
            relevant_adapter.generator, BoTorchGenerator
        ):
            raise AxError(
                "The Objective vs P(feasible) plot cannot be computed using the"
                f" current Adapter ({relevant_adapter}) and generator"
                f" ({relevant_adapter.generator}). Only TorchAdapters using"
                " BoTorchGenerators are supported."
            )
        # Generate arms on the objective p_feasible frontier
        # Specify to optimize multiple acquisition functions (via
        # MultiAcquisition). This will optimize the PosteriorMean and the
        # LogProbabilityOfFeasibility using MOO.
        generator = relevant_adapter.generator
        orig_acquisition_class = generator.acquisition_class
        orig_botorch_acqf_classes_with_options = (
            generator._botorch_acqf_classes_with_options
        )
        generator.acquisition_class = MultiAcquisition
        generator._botorch_acqf_classes_with_options = [
            (PosteriorMean, {}),
            (LogProbabilityOfFeasibility, {}),
        ]
        frontier_gr = relevant_adapter.gen(n=self.num_points_to_generate)
        # in case this generator is used again, restore the original settings
        generator.acquisition_class = orig_acquisition_class
        generator._botorch_acqf_classes_with_options = (
            orig_botorch_acqf_classes_with_options
        )
        self.additional_arms = self._additional_arms + [
            Arm(name=f"frontier_{i}", parameters=a.parameters)
            for i, a in enumerate(frontier_gr.arms)
        ]
        scatter_plot = ScatterPlot(
            x_metric_name=experiment.optimization_config.objective.metric.name,
            y_metric_name="p_feasible",
            use_model_predictions=self.use_model_predictions,
            relativize=self.relativize,
            trial_index=self.trial_index,
            trial_statuses=self.trial_statuses,
            additional_arms=self.additional_arms,
            labels=self.labels,
            show_pareto_frontier=self.show_pareto_frontier,
            title=(
                f"Modeled {'Relativized ' if self.relativize else ''} Effect on the"
                " Objective vs % Chance of Satisfying the Constraints"
            ),
        )
        return scatter_plot.compute(experiment=experiment, adapter=relevant_adapter)
