---
id: api
title: APIs
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

The modular design of Ax enables three different usage modes, with different
balances of structure to flexibility and reproducibility. Navigate to the
["Tutorials" page](/docs/tutorials) for an in-depth walk-through of each API and
usage mode.

**NOTE: We recommend the Service API for the vast majority of use cases.** This
API provides an ideal balance of flexibility and simplicity for most users, and
we are in the process of consolidating Ax usage around it more formally.

From most lightweight to fullest functionality, our APIs are:

-   **Loop API** ([tutorial](/docs/tutorials/gpei_hartmann_loop)) is intended for
    synchronous optimization loops, where [trials](glossary.md#trial) can be
    evaluated right away. With this API, optimization can be executed in a single
    call and [experiment](glossary.md#experiment) introspection is available once
    optimization is complete. **Use this API only for the simplest use cases where
    running a single trial is fast and only one trial should be running at a
    time.**
-   **[RECOMMENDED] Service API**
    ([tutorial](/docs/tutorials/gpei_hartmann_service)) can be used as a
    lightweight service for parameter-tuning applications where trials might be
    evaluated in parallel and data is available asynchronously (e.g.
    hyperparameter or simulation optimization). It requires little to no knowledge
    of Ax data structures and easily integrates with various schedulers. In this
    mode, Ax suggests one-[arm](glossary.md#arm) trials to be evaluated by the
    client application, and expects them to be completed with
    [metric](glossary.md#metric) data when available. **This is our most popular
    API and a good place to start as a new user. Use it to leverage nearly full
    hyperparameter optimization functionality of Ax without the need to learn its
    architecture and how things work under the hood.**
    -   In both the Loop and the Service API, it is possible to configure the
        optimization algorithm via an Ax `GenerationStrategy`
        ([tutorial](/docs/tutorials/generation_strategy)), so use of Developer API
        is not required to control the optimization algorithm in Ax.
-   **Developer API** ([tutorial](/docs/tutorials/gpei_hartmann_developer)) is for
    ad-hoc use by data scientists, machine learning engineers, and researchers.
    The developer API allows for a great deal of customization and introspection,
    and is recommended for those who plan to use Ax to optimize A/B tests. Using
    the developer API requires some knowledge of [Ax architecture](core.md). **Use
    this API if you are looking to perform field experiments with `BatchTrial`-s,
    customize or contribute to Ax, or leverage advanced functionality that is not
    exposed in other APIs.**
    -   While not an API, the **`Scheduler`**
        ([tutorial](/docs/tutorials/scheduler)) is an important and distinct
        use-case of the Ax Developer API. With the `Scheduler`, it's possible to run
        a configurable, managed closed-loop optimization where trials are deployed
        and polled in an async fashion and no human intervention/oversight is
        required until the experiment is complete. **Use the `Scheduler` when you
        are looking to configure and start a full experiment that will need to
        interact with an external system to evaluate trials.**

Here is a comparison of the three APIs in the simple case of evaluating the
unconstrained synthetic Branin function:


<Tabs>
  <TabItem value="Loop" label="Loop" default>

```py

from ax import optimize
from ax.utils.measurement.synthetic_functions import branin

best_parameters, values, experiment, model = optimize(
    parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [-5.0, 10.0],
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0.0, 10.0],
        },
    ],
    evaluation_function=lambda p: (branin(p["x1"], p["x2"]), 0.0),
    minimize=True,
)

```

  </TabItem>
  <TabItem value="Service" label="Service">

```py

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import branin

ax_client = AxClient()
ax_client.create_experiment(
    name="branin_test_experiment",
    parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [-5.0, 10.0],
            "value_type": "float",
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0.0, 10.0],
        },
    ],
    objectives={"branin": ObjectiveProperties(minimize=True)},
)

for _ in range(15):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=branin(parameters["x1"], parameters["x2"]))

best_parameters, metrics = ax_client.get_best_parameters()

```

  </TabItem>
  <TabItem value="Developer" label="Developer">

```py

from ax import *


class MockRunner(Runner):
    def run(self, trial):
        return {"name": str(trial.index)}


branin_search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
        ),
        RangeParameter(
            name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15
        ),
    ]
)
exp = Experiment(
    name="test_branin",
    search_space=branin_search_space,
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=BraninMetric(name="branin", param_names=["x1", "x2"]),
            minimize=True,
        ),
    ),
    runner=MockRunner(),
)

sobol = Models.SOBOL(exp.search_space)
for i in range(5):
    trial = exp.new_trial(generator_run=sobol.gen(1))
    trial.run()
    trial.mark_completed()

best_arm = None
for i in range(15):
    gpei = Models.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data())
    generator_run = gpei.gen(1)
    best_arm, _ = generator_run.best_arm_predictions
    trial = exp.new_trial(generator_run=generator_run)
    trial.run()
    trial.mark_completed()

exp.fetch_data()
best_parameters = best_arm.parameters

```

  </TabItem>
  <TabItem value="Scheduler" label="Scheduler">

```py

from ax import *
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.service import Scheduler

# Full `Experiment` and `GenerationStrategy` instantiation
# omitted for brevity, refer to the "Tutorials" page for detail.
experiment = Experiment(...)
generation_strategy = GenerationStrategy(...)

scheduler = Scheduler(
    experiment=experiment,
    generation_strategy=generation_strategy,
    options=SchedulerOptions(),  # Configurations for how to run the experiment
)

scheduler.run_n_trials(100)  # Automate running 100 trials and reporting results

```

  </TabItem>
</Tabs>
