---
id: api
title: APIs
---
Modular design of Ax enables multiple modes of usage, which provide different balances of lightweight structure to flexibility and reproducibility.

There are three primary usage modes, ranked from the most lightweight to fullest functionality:
  - **Loop API**. This API is intended for synchronous optimization loops. It is the recommended starting point for optimization of arbitrary functions that do not require parallelism and do not take a long time to compute.
  - **Service API**: This is a simplified API which can be used as a lightweight service (via, for example, a Thrift interface) for parameter tuning applications, like hyper-parameter optimization and simulation optimization, where there are multiple parallel runs and data is available asynchronously. In this mode, Ax handles the experimentation algorithm(s), but not the execution of experiment iterations. This mode requires little to no knowledge of Ax data structures and easily integrates with various schedulers.
  - **Developer API**. This API is for ad hoc use by data scientists, machine learning engineers, and researchers.  The developer API allows for a great deal of customizability, and is recommended for those who plan to use Ax to optimize A/B tests. This requires some knowledge of _Ax architecture_.

Here is a comparison of the three APIs in the case of evaluating the synthetic Branin function:

<!--DOCUSAURUS_CODE_TABS-->
<!--Loop-->
```py
loop = OptimizationLoop.with_evaluation_function(
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
    experiment_name="test",
    objective_name="branin",
    evaluation_function=branin_evaluation_function,
    minimize=True,
)
loop.full_run()

best_parameters = loop.get_best_point()
```

<!--Service-->
```py
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
    ax.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

best_parameters, metrics = ax.get_best_parameters()
```

<!--Developer-->
```py
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
    evaluation_function=branin_evaluation_function,
    objective_name="branin",
    minimize=True,
)

sobol = modelbridge.get_sobol(exp.search_space)
for i in range(5):
    exp.new_trial(generator_run=sobol.gen(1))

best_arm = None
for i in range(15):
    gpei = modelbridge.get_GPEI(experiment=exp, data=exp.eval())
    generator_run = gpei.gen(1)
    best_arm, _ = generator_run.best_arm_predictions
    exp.new_trial(generator_run=generator_run)

best_parameters = best_arm.parameters
```

<!--END_DOCUSAURUS_CODE_TABS-->
