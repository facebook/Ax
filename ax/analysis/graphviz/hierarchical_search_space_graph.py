# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.graphviz.graphviz_analysis import (
    create_graphviz_analysis_card,
    GraphvizAnalysisCard,
)
from ax.analysis.search_space_summary import SearchSpaceSummary
from ax.analysis.utils import validate_experiment
from ax.core.analysis_card import AnalysisCard
from ax.core.experiment import Experiment
from ax.core.parameter import Parameter
from ax.core.search_space import SearchSpace
from ax.generation_strategy.generation_strategy import GenerationStrategy
from graphviz import Digraph
from pyre_extensions import assert_is_instance, none_throws, override

SUBTITLE: str = """
Visualize the dependency relationships within the search space. If a parameter lies
within a box it is only active when the parameter pointing to the box takes on the
value on the arrow.
"""


class HierarchicalSearchSpaceGraph(Analysis):
    """
    Create a graphviz graph of the hierarchical search space, showing dependency
    relationships as subgraphs.

    Each Parameter is a node on the graph. If a Parameter has dependents, an edge is
    drawn from that Parameter to a subgraph (dotted box) containing the dependents. The
    subgraph and the edge is labeled with the value of the Parameter that the
    dependents depend on.

    The attached DataFrame is simply the SearchSpaceSummary and contains same columns.
    """

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        """
        HierarchicalSearchSpaceGraph only requires the Experiment's SearchSpace have at
        least one hierarchical Parameter.
        """
        if (
            experiment_invalid_reason := validate_experiment(
                experiment=experiment,
                require_trials=False,
                require_data=False,
            )
        ) is not None:
            return experiment_invalid_reason

        if not none_throws(experiment).search_space.is_hierarchical:
            return "Requires a SearchSpace with at least one hierarchical Parameter"

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> GraphvizAnalysisCard:
        search_space = none_throws(experiment).search_space

        dot = search_space_to_graphviz(search_space=search_space)

        df = assert_is_instance(
            SearchSpaceSummary()
            .compute_result(
                experiment=experiment,
                generation_strategy=generation_strategy,
                adapter=adapter,
            )
            .ok,
            AnalysisCard,
        ).df

        return create_graphviz_analysis_card(
            name=self.__class__.__name__,
            title="Hierarchical Search Space Graph",
            subtitle=SUBTITLE,
            df=df,
            dot=dot,
        )


def search_space_to_graphviz(search_space: SearchSpace) -> Digraph:
    """
    Helper function to create a graphviz graph of the hierarchical search space by
    calling parameter_to_graphviz on the top level parameters and performing a depth
    first search through the dependents.
    """
    dot = Digraph(name="SearchSpace")

    dot.attr(
        # Arrange the graph left-to-right (as opposed to the default top-to-bottom)
        rankdir="LR",
        # Allow edges to travel through separate subgraphs
        compound="true",
    )

    for parameter in search_space.top_level_parameters.values():
        parameter_to_graphviz(
            dot=dot,
            search_space=search_space,
            parameter=parameter,
        )

    return dot


def parameter_to_graphviz(
    dot: Digraph,
    search_space: SearchSpace,
    parameter: Parameter,
) -> None:
    """
    Recursively add nodes and edges to the graph for a given parameter. Modifies `dot`
    in place.
    """
    if parameter.is_hierarchical:
        for value, dependents in parameter.dependents.items():
            # Create a subgraph for hierarchical parameters
            with dot.subgraph(name=f"cluster_{parameter.name}_{value}") as sub:
                sub.attr(style="dashed", label=f"{parameter.name} == '{value}'")

                # Create an anchor node since edges can only connect nodes to other
                # nodes.
                sub.node(
                    f"cluster_{parameter.name}_{value}_anchor",
                    label="",
                    style="invis",
                )
                for dep_name in dependents:
                    dep_param = search_space[dep_name]
                    parameter_to_graphviz(sub, search_space, dep_param)

            # Connect parameter to subgraph anchor
            dot.edge(
                parameter.name,
                f"cluster_{parameter.name}_{value}_anchor",
                label=str(value),
            )
    else:
        dot.node(parameter.name)
