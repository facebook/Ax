---
id: api
title: APIs
---
The modular design of Ax enables three different usage modes, with different balances of structure to flexibility and reproducibility. From most lightweight to fullest functionality, they are:
  - **Loop API** is intended for synchronous optimization loops, where [trials](glossary.md#trial) can be evaluated right away. With this API, optimization can be executed in a single call and [experiment](glossary.md#experiment) introspection is available once optimization is complete.
  - **Service API** can be used as a lightweight service for parameter-tuning applications where trials might be evaluated in parallel and data is available asynchronously (e.g. hyperparameter or simulation optimization). It requires little to no knowledge of Ax data structures and easily integrates with various schedulers. In this mode, Ax suggests one-[arm](glossary.md#arm) trials to be evaluated by the client application, and expects them to be completed with [metric](glossary.md#metric) data when available.
  - **Developer API** is for ad-hoc use by data scientists, machine learning engineers, and researchers.  The developer API allows for a great deal of customization and introspection, and is recommended for those who plan to use Ax to optimize A/B tests. Using the developer API requires some knowledge of [Ax architecture](core.md).

Here is a comparison of the three APIs in the simple case of evaluating the unconstrained synthetic Branin function:

<!--DOCUSAURUS_CODE_TABS-->
<!--Loop-->
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
    evaluation_function=lambda p: branin(p["x1"], p["x2"]),
    minimize=True,
)
```

<!--Service-->
```py
from ax.service.ax_client import AxClient
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
    objective_name="branin",
    minimize=True,
)

for _ in range(15):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=branin(parameters["x1"], parameters["x2"]))

best_parameters, metrics = ax_client.get_best_parameters()
```

<!--Developer-->
```py
from ax import *

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
exp = SimpleExperiment(
    name="test_branin",
    search_space=branin_search_space,
    evaluation_function=lambda p: branin(p["x1"], p["x2"]),
    objective_name="branin",
    minimize=True,
)

sobol = Models.SOBOL(exp.search_space)
for i in range(5):
    exp.new_trial(generator_run=sobol.gen(1))

best_arm = None
for i in range(15):
    gpei = Models.GPEI(experiment=exp, data=exp.eval())
    generator_run = gpei.gen(1)
    best_arm, _ = generator_run.best_arm_predictions
    exp.new_trial(generator_run=generator_run)

best_parameters = best_arm.parameters
```

<!--END_DOCUSAURUS_CODE_TABS-->
