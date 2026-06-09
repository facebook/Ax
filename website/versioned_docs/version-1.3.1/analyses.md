---
id: analyses
title: Utilizing and Creating Analyses
---

:::info

This document discusses non-API components of Ax, which may change between major
library versions. Contributor guides are most useful for developers intending to
publish PRs to Ax, not those using Ax directly or building tools on top of Ax.

:::

# Utilizing and Creating Ax Analyses

Ax’s Analysis module provides a framework for producing plots, tables, messages,
and more to help users understand their experiments. This is facilitated via the
`Analysis` protocol and its various subclasses.

Analysis classes implement a method `compute` which consumes an `Experiment`,
`GenerationStrategy`, and/or `Adapter` and outputs a collection of
`AnalysisCards`. These cards contain a dataframe with relevant data, a “blob”
which contains data to be rendered (ex. a plot), and miscellaneous metadata like
a title, subtitle, and priority level used for sorting. `compute` returns a
collection of cards so that Analyses can be composed together. For example: the
`TopSurfacesPlot` computes a `SensitivityAnalysisPlot` to understand which
parameters in the search space are most relevent, then produces `SlicePlot`s and
`ContourPlot`s for the most important surfaces.

Ax currently provides implementations for 3 base classes: (1)`Analysis` -- for
creating tables, (2) `PlotlyAnalysis` -- for producing plots using the Plotly
library, and (3) `MarkdownAnalysis` -- for producing messages. Importantly Ax is
able to save these cards to the database using `save_analysis_cards`, allowing
for analyses to be pre-computed and displayed at a later time. This is done
automatically when `Client.compute_analyses` is called.

## Using Analyses

The simplest way to use an `Analysis` is to call `Client.compute_analyses`. This
will heuristically select the most relevant analyses to compute, save the cards
to the database, return them, and display them in your IPython environment if
possible. Users can also specify which analyses to compute and pass them in
manually, for example:
`client.compute_analyses(analyses=[TopSurfacesPlot(), Summary(), ...])`.

When developing a new `Analysis` it can be useful to compute an analysis "a-la
carte". To do this, manually instantiate the `Analysis` and call its `compute`
method. This will return a collection of `AnalysisCards` which can be displayed.

```python
analysis = CrossValidationPlot()

cards = analysis.compute(
    experiment=experiment,
    generation_strategy=generation_strategy,
    adapter=adapter,
)
```

## Creating a new Analysis

Let's implement a simple Analysis that returns a table counting the number of
trials in each `TrialStatus` . We'll make a new class that implements the
`Analysis` protocol (i.e. it defines a `compute` method).

```python
class TrialStatusTable(Analysis):
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[AnalysisCard]:
        trials_by_status = experiment.trials_by_status

        records = [
            {"status": status.name, "count": len(trials)}
            for status, trials in trials_by_status.items()
        ]

        return [
            self._create_analysis_card(
                title="Trials by Status",
                subtitle="How many trials are in each status?",
                level=AnalysisCardLevel.LOW,
                category=AnalysisCardCategory.INSIGHT,
                df=pd.DataFrame.from_records(records),
            )
        ]

cards = client.compute_analyses(analyses=[TrialStatusTable()])
```

## Adding options to an Analysis

Imagine we wanted to add an option to change how this analysis is computed, say
we wish to toggle whether the analysis computes the _number_ of trials in a
given state or the _percentage_ of trials in a given state. We cannot change the
input arguments to `compute`, so this must be added elsewhere.

The analysis' initializer is a natural place to put additional settings. We'll
create a `TrialStatusTable.__init__` method which takes in the option as a
boolean, then modify `compute` to consume this option as well. Following this
patterns allows users to specify all relevant settings before calling
`Client.compute_analyses` while still allowing the underlying `compute` call to
remain unchanged. Standarization of the `compute` call simplifies logic
elsewhere in the stack.

```python
class TrialStatusTable(Analysis):
    def __init__(self, as_fraction: bool) -> None:
        super().__init__()

        self.as_fraction = as_fraction

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[AnalysisCard]:
        trials_by_status = experiment.trials_by_status
        denominator = len(experiment.trials) if self.as_fraction else 1

        records = [
            {"status": status.name, "count": len(trials) / denominator}
            for status, trials in trials_by_status.items()
        ]

        return [
            # Use _create_analysis_card rather than AnalysisCard to automatically populate relevant metadata
            self._create_analysis_card(
                title="Trials by Status",
                subtitle="How many trials are in each status?",
                level=AnalysisCardLevel.LOW,
                category=AnalysisCardCategory.INSIGHT,
                df=pd.DataFrame.from_records(records),
            )
        ]


cards = client.compute_analyses(analyses=[TrialStatusTable(as_fraction=True)])
```

## Miscellaneous tips

- Many analyses rely on the same infrastructure and utility functions -- check
  to see if what you need has already been implemented somewhere.
  - Many analyses require an `Adapter` but can use either the `Adapter` provided
    or the current `Adapter` on the `GenerationStrategy` --
    `extract_relevant_adapter` handles this in a consistent way
  - Analyses which use an `Arm` as the fundamental unit of analysis will find
    the `prepare_arm_data` utility useful; using it will also lend the
    `Analysis` useful features like relativization for free
- When writing a new `PlotlyAnalysis` check out `ax.analysis.plotly.utils` for
  guidance on using color schemes and unified tool tips
- Try to follow consistent design patterns; many analyses take an optional list
  of `metric_names` on initialization, and interpret `None` to mean the user
  wants to compute a card for each metric present. Following these conventions
  makes things easier for downstream consumers.
