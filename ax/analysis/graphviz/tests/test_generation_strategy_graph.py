# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.adapter.registry import Generators
from ax.analysis.graphviz.generation_strategy_graph import (
    _add_edges_for_node,
    _add_node_to_graph,
    _create_generation_strategy_df,
    generation_strategy_to_graphviz,
    GenerationStrategyGraph,
    SUBTITLE,
)
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import (
    GenerationStep,
    GenerationStrategy,
)
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import (
    AutoTransitionAfterGen,
    MinTrials,
)
from ax.utils.common.testutils import TestCase
from graphviz import Digraph


class TestGenerationStrategyGraph(TestCase):
    def setUp(self) -> None:
        super().setUp()

        # Create a simple step-based generation strategy
        self.step_gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    generator=Generators.SOBOL,
                    num_trials=5,
                ),
                GenerationStep(
                    generator=Generators.BOTORCH_MODULAR,
                    num_trials=-1,
                ),
            ]
        )

        # Create a node-based generation strategy with multiple transitions
        self.node_gs = GenerationStrategy(
            name="Sobol+MBM",
            nodes=[
                GenerationNode(
                    name="Sobol",
                    generator_specs=[GeneratorSpec(generator_enum=Generators.SOBOL)],
                    transition_criteria=[
                        MinTrials(threshold=5, transition_to="MBM"),
                    ],
                ),
                GenerationNode(
                    name="MBM",
                    generator_specs=[
                        GeneratorSpec(generator_enum=Generators.BOTORCH_MODULAR)
                    ],
                ),
            ],
        )

        # Create a more complex node-based GS with branching transitions
        self.complex_gs = GenerationStrategy(
            name="Complex_GS",
            nodes=[
                GenerationNode(
                    name="Sobol",
                    generator_specs=[GeneratorSpec(generator_enum=Generators.SOBOL)],
                    transition_criteria=[
                        MinTrials(threshold=5, transition_to="MBM"),
                    ],
                ),
                GenerationNode(
                    name="MBM",
                    generator_specs=[
                        GeneratorSpec(generator_enum=Generators.BOTORCH_MODULAR)
                    ],
                    transition_criteria=[
                        AutoTransitionAfterGen(
                            transition_to="Refinement",
                            continue_trial_generation=True,
                        ),
                    ],
                ),
                GenerationNode(
                    name="Refinement",
                    generator_specs=[GeneratorSpec(generator_enum=Generators.SOBOL)],
                ),
            ],
        )

    def test_validate_applicable_state_no_gs(self) -> None:
        """Test that validation fails when no GenerationStrategy is provided."""
        analysis = GenerationStrategyGraph()
        result = analysis.validate_applicable_state(
            experiment=None,
            generation_strategy=None,
        )
        self.assertIsNotNone(result)
        self.assertIn("requires a GenerationStrategy", result)

    def test_validate_applicable_state_valid(self) -> None:
        """Test that validation passes with a valid GenerationStrategy."""
        analysis = GenerationStrategyGraph()
        result = analysis.validate_applicable_state(
            experiment=None,
            generation_strategy=self.node_gs,
        )
        self.assertIsNone(result)

    def test_compute_step_based_gs(self) -> None:
        """Test computing the graph for a step-based GenerationStrategy."""
        analysis = GenerationStrategyGraph()
        card = analysis.compute(generation_strategy=self.step_gs)

        # Test metadata
        self.assertEqual(card.name, "GenerationStrategyGraph")
        self.assertEqual(card.title, "Generation Strategy Graph")
        # Subtitle includes strategy name prefix followed by SUBTITLE
        self.assertIn(self.step_gs.name, card.subtitle)
        self.assertIn(SUBTITLE, card.subtitle)
        self.assertIsNotNone(card.blob)

        # Test that the graph contains the expected nodes
        dot = card.get_digraph()
        source = dot.source
        self.assertIn("GenerationStep_0", source)
        self.assertIn("GenerationStep_1", source)

    def test_compute_node_based_gs(self) -> None:
        """Test computing the graph for a node-based GenerationStrategy."""
        analysis = GenerationStrategyGraph()
        card = analysis.compute(generation_strategy=self.node_gs)

        # Test metadata
        self.assertEqual(card.name, "GenerationStrategyGraph")
        self.assertEqual(card.title, "Generation Strategy Graph")

        # Test that the graph contains the expected nodes
        dot = card.get_digraph()
        source = dot.source
        self.assertIn("Sobol", source)
        self.assertIn("MBM", source)

        # Test that the transition edge is present
        self.assertIn("Sobol -> MBM", source)
        self.assertIn("MinTrials", source)

    def test_compute_complex_gs(self) -> None:
        """Test computing the graph for a complex GenerationStrategy with
        multiple transitions including continue_trial_generation edges.
        """
        analysis = GenerationStrategyGraph()
        card = analysis.compute(generation_strategy=self.complex_gs)

        dot = card.get_digraph()
        source = dot.source

        # Check all nodes are present
        self.assertIn("Sobol", source)
        self.assertIn("MBM", source)
        self.assertIn("Refinement", source)

        # Check edges are present
        self.assertIn("Sobol -> MBM", source)
        self.assertIn("MBM -> Refinement", source)

        # Check that the AutoTransitionAfterGen edge uses dashed style
        # (for continue_trial_generation=True)
        self.assertIn("dashed", source)

    def test_current_node_highlighted(self) -> None:
        """Test that the current node is highlighted in the graph."""
        analysis = GenerationStrategyGraph()
        card = analysis.compute(generation_strategy=self.node_gs)

        dot = card.get_digraph()
        source = dot.source

        # The current node (Sobol) should be highlighted with lightblue fill
        self.assertIn("lightblue", source)
        self.assertIn("filled", source)

    def test_dataframe_content(self) -> None:
        """Test that the DataFrame contains the expected information."""
        analysis = GenerationStrategyGraph()
        card = analysis.compute(generation_strategy=self.node_gs)

        df = card.df
        self.assertEqual(len(df), 2)

        # Check first row (Sobol node)
        sobol_row = df[df["node_name"] == "Sobol"].iloc[0]
        self.assertEqual(sobol_row["generators"], "Sobol")
        self.assertIn("MBM", sobol_row["transitions"])
        self.assertTrue(sobol_row["is_current"])

        # Check second row (MBM node)
        mbm_row = df[df["node_name"] == "MBM"].iloc[0]
        self.assertEqual(mbm_row["generators"], "BoTorch")
        self.assertFalse(mbm_row["is_current"])

    def test_generation_strategy_to_graphviz(self) -> None:
        """Test the helper function that creates the graphviz Digraph."""
        dot = generation_strategy_to_graphviz(generation_strategy=self.node_gs)

        self.assertIsInstance(dot, Digraph)
        # Graph uses top-to-bottom layout
        self.assertIn("TB", dot.source)
        # Graph contains the expected nodes
        self.assertIn("Sobol", dot.source)
        self.assertIn("MBM", dot.source)

    def test_add_node_to_graph_current(self) -> None:
        """Test adding a current node to the graph."""
        dot = Digraph()
        node = self.node_gs._nodes[0]
        _add_node_to_graph(dot=dot, node=node, is_current=True)

        source = dot.source
        self.assertIn("Sobol", source)
        self.assertIn("lightblue", source)
        self.assertIn("bold", source)

    def test_add_node_to_graph_non_current(self) -> None:
        """Test adding a non-current node to the graph."""
        dot = Digraph()
        node = self.node_gs._nodes[1]
        _add_node_to_graph(dot=dot, node=node, is_current=False)

        source = dot.source
        self.assertIn("MBM", source)
        self.assertIn("rounded", source)

    def test_add_edges_for_node(self) -> None:
        """Test adding edges for a node."""
        dot = Digraph()
        # First add nodes
        for node in self.node_gs._nodes:
            _add_node_to_graph(dot=dot, node=node, is_current=False)
        # Then add edges
        _add_edges_for_node(dot=dot, node=self.node_gs._nodes[0])

        source = dot.source
        self.assertIn("Sobol -> MBM", source)
        self.assertIn("MinTrials", source)

    def test_create_generation_strategy_df(self) -> None:
        """Test the DataFrame creation helper function."""
        df = _create_generation_strategy_df(generation_strategy=self.node_gs)

        self.assertEqual(len(df), 2)
        self.assertListEqual(
            list(df.columns), ["node_name", "generators", "transitions", "is_current"]
        )

        # First node should be current
        self.assertTrue(df.iloc[0]["is_current"])
        self.assertFalse(df.iloc[1]["is_current"])
