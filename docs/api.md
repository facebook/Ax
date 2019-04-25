---
id: api
title: APIs
---
Modular design of Ax enables multiple modes of usage, which provide different balances of lightweight structure to flexibility and reproducibility.

There are three primary usage modes, ranked from the most lightweight to fullest functionality:
  - **Loop API**. This API is intended for synchronous optimization loops. It is the recommended starting point for optimization of arbitrary functions that do not require parallelism and do not take a long time to compute. Use if:
    - You are looking to optimize an algorithm via a single function call.
    - You are not looking to control or introspect the optimization.
  - **Service API**: This is a simplified API which can be used as a lightweight service (via, for example, a Thrift or RPC-type interface) for parameter tuning applications, like hyper-parameter optimization and simulation optimization, where there are multiple parallel runs and data is available asynchronously. In this mode, Ax handles the experimentation algorithm(s), but not the execution of experiment iterations. This mode requires little to no knowledge of Ax data structures and easily integrates with various schedulers. Use if:
    - You are looking to evaluate trials locally or via an external scheduler.
    - You do not need batch optimization.
  - **Developer API**. This API is for ad hoc use by data scientists, machine learning engineers, and researchers.  The developer API allows for a great deal of customizability, and is recommended for those who plan to use Ax to optimize A/B tests. This requires some knowledge of [Ax architecture](experiment.md). Use if:
    - You are looking to have full control of the optimization and ability to introspect it in a notebook.
    - You are trying out new algorithms, acquisition functions, etc.

Here is a comparison of the three APIs in the simple case of evaluating the unconstrained synthetic Branin function:

<!--DOCUSAURUS_CODE_TABS-->
<!--Loop-->
```py
from ax import optimize
from ax.metrics.branin import branin

optimize(
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
from ax.metrics.branin import branin

ax = AxClient()
ax.create_experiment(
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
    parameters, trial_index = ax.get_next_trial()
    ax.complete_trial(trial_index=trial_index, raw_data=branin(parameters["x1"], parameters["x2"]))

best_parameters, metrics = ax.get_best_parameters()
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
