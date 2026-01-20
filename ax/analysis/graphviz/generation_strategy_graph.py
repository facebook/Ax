# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.graphviz.graphviz_analysis import (
    create_graphviz_analysis_card,
    GraphvizAnalysisCard,
)
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.transition_criterion import (
    AuxiliaryExperimentCheck,
    TransitionCriterion,
    TrialBasedCriterion,
)
from graphviz import Digraph
from pyre_extensions import none_throws, override

SUBTITLE: str = """
Visualize the structure of a GenerationStrategy as a directed graph. Each node
represents a GenerationNode in the strategy, and edges represent transitions
between nodes based on TransitionCriterion. Edge labels show the criterion
class names that trigger the transition.
"""


def _format_criterion(tc: TransitionCriterion) -> str:
    """
    Format a TransitionCriterion for display in a concise way.

    For TrialBasedCriterion (MinTrials, MaxGenerationParallelism, etc.),
    shows just the threshold: MinTrials(5)

    For AuxiliaryExperimentCheck, shows the purposes to include:
    AuxiliaryExperimentCheck([PE_EXPERIMENT])

    For other criteria, shows just the class name.
    """
    class_name = tc.criterion_class

    if isinstance(tc, TrialBasedCriterion):
        return f"{class_name}({tc.threshold})"

    if isinstance(tc, AuxiliaryExperimentCheck):
        purposes = tc.auxiliary_experiment_purposes_to_include
        if purposes:
            purpose_values = ", ".join(p.value for p in purposes)
            return f"{class_name}({purpose_values})"
        return class_name

    return class_name


class GenerationStrategyGraph(Analysis):
    """
    Create a graphviz graph of a GenerationStrategy, showing the nodes and
    transition edges between them.

    Each GenerationNode is represented as a node in the graph. Edges are drawn
    between nodes based on the TransitionCriterion defined on each node. The
    edge labels show the names of the TransitionCriterion classes that define
    the transition.

    The current node in the GenerationStrategy is filled in with light blue
    color to indicate the strategy's current position.

    The attached DataFrame contains information about each node including its
    name, generator specs, and transition criteria.
    """

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        """
        GenerationStrategyGraph requires a GenerationStrategy to be provided.
        """
        if generation_strategy is None:
            return "GenerationStrategyGraph requires a GenerationStrategy"

        if len(generation_strategy._nodes) == 0:
            return "GenerationStrategy has no nodes to visualize"

        return None

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> GraphvizAnalysisCard:
        gs = none_throws(generation_strategy)

        dot = generation_strategy_to_graphviz(generation_strategy=gs)

        df = _create_generation_strategy_df(generation_strategy=gs)

        return create_graphviz_analysis_card(
            name=self.__class__.__name__,
            title="Generation Strategy Graph",
            subtitle=f"GenerationStrategy: {gs.name}\n{SUBTITLE}",
            df=df,
            dot=dot,
        )


def generation_strategy_to_graphviz(generation_strategy: GenerationStrategy) -> Digraph:
    """
    Create a graphviz Digraph representing the GenerationStrategy's node graph.

    Args:
        generation_strategy: The GenerationStrategy to visualize.

    Returns:
        A graphviz Digraph object representing the strategy.
    """
    dot = Digraph(name="GenerationStrategy")

    dot.attr(
        rankdir="TB",  # Arrange the graph top-to-bottom
    )
    current_node_name = generation_strategy.current_node_name

    # Add all nodes & edges
    for node in generation_strategy._nodes:
        _add_node_to_graph(
            dot=dot,
            node=node,
            is_current=node.name == current_node_name,
        )
    for node in generation_strategy._nodes:
        _add_edges_for_node(dot=dot, node=node)

    return dot


def _add_node_to_graph(
    dot: Digraph,
    node: GenerationNode,
    is_current: bool,
) -> None:
    """
    Add a GenerationNode as a node in the graphviz graph.

    Args:
        dot: The Digraph to add the node to.
        node: The GenerationNode to add.
        is_current: Whether this is the current node in the strategy.
    """
    # Build the label with node name and generator info
    generator_names = [spec.generator_key for spec in node.generator_specs]
    generators_str = ", ".join(generator_names)
    # Elide the generators if they match the node name
    if node.name == generators_str:
        label = node.name
    else:
        label = f"{node.name}\\n({generators_str})"

    # Style the current node differently
    if is_current:
        dot.node(
            node.name,
            label=label,
            shape="box",
            style="filled,bold",
            fillcolor="lightblue",
            penwidth="2",
        )
    else:
        dot.node(
            node.name,
            label=label,
            shape="box",
            style="rounded",
        )


def _add_edges_for_node(dot: Digraph, node: GenerationNode) -> None:
    """
    Add edges from a GenerationNode to its transition targets.

    Args:
        dot: The Digraph to add edges to.
        node: The GenerationNode whose transitions to add.
    """
    for next_node_name, criteria in node.transition_edges.items():
        if next_node_name is None:
            # Skip self-loops for criteria like MaxGenerationParallelism
            # that don't have a transition target
            continue

        # Create edge label from criterion representations (concise format)
        criterion_strs = [_format_criterion(tc) for tc in criteria]
        edge_label = "\\n".join(criterion_strs)

        # Check if this is a "continue trial generation" edge
        # (i.e., multiple nodes contribute to the same trial)
        is_continue_trial = all(tc.continue_trial_generation for tc in criteria)

        if is_continue_trial:
            dot.edge(
                node.name,
                next_node_name,
                label=edge_label,
                style="dashed",
                color="blue",
            )
        else:
            dot.edge(
                node.name,
                next_node_name,
                label=edge_label,
            )


def _create_generation_strategy_df(
    generation_strategy: GenerationStrategy,
) -> pd.DataFrame:
    """
    Create a DataFrame summarizing the GenerationStrategy's nodes.

    Args:
        generation_strategy: The GenerationStrategy to summarize.

    Returns:
        A DataFrame with columns for node name, generators, transition criteria,
        and whether it's the current node.
    """
    rows = []
    current_node_name = generation_strategy.current_node_name

    for node in generation_strategy._nodes:
        generator_names = [spec.generator_key for spec in node.generator_specs]

        # Build transition info
        transitions = []
        for next_node_name, criteria in node.transition_edges.items():
            if next_node_name is not None:
                criterion_strs = [_format_criterion(tc) for tc in criteria]
                transitions.append(f"-> {next_node_name}: {', '.join(criterion_strs)}")

        rows.append(
            {
                "node_name": node.name,
                "generators": ", ".join(generator_names),
                "transitions": "; ".join(transitions) if transitions else "None",
                "is_current": node.name == current_node_name,
            }
        )

    return pd.DataFrame(rows)
