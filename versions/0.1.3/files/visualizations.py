
import numpy as np
from ax import (
    Arm,
    ComparisonOp,
    RangeParameter,
    ParameterType,
    SearchSpace, 
    SimpleExperiment, 
    OutcomeConstraint, 
)

from ax.metrics.l2norm import L2NormMetric
from ax.modelbridge.cross_validation import cross_validate
from ax.modelbridge.registry import Models
from ax.plot.contour import interact_contour, plot_contour
from ax.plot.diagnostic import interact_cross_validation
from ax.plot.scatter import(
    interact_fitted,
    plot_objective_vs_constraints,
    tile_fitted,
)
from ax.plot.slice import plot_slice
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()

noise_sd = 0.1
param_names = [f"x{i+1}" for i in range(6)]  # x1, x2, ..., x6

def noisy_hartmann_evaluation_function(parameterization):
    x = np.array([parameterization.get(p_name) for p_name in param_names])
    noise1, noise2 = np.random.normal(0, noise_sd, 2)

    return {
        "hartmann6": (hartmann6(x) + noise1, noise_sd),
        "l2norm": (np.sqrt((x ** 2).sum()) + noise2, noise_sd)
    }

hartmann_search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name=p_name, parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
        )
        for p_name in param_names
    ]
)

exp = SimpleExperiment(
    name="test_branin",
    search_space=hartmann_search_space,
    evaluation_function=noisy_hartmann_evaluation_function,
    objective_name="hartmann6",
    minimize=True,
    outcome_constraints=[
        OutcomeConstraint(
            metric=L2NormMetric(
                name="l2norm", param_names=param_names, noise_sd=0.2
            ),
            op=ComparisonOp.LEQ,
            bound=1.25,
            relative=False,
        )
    ],
)

N_RANDOM = 5
BATCH_SIZE = 1
N_BATCHES = 15

sobol = Models.SOBOL(exp.search_space)
exp.new_batch_trial(generator_run=sobol.gen(N_RANDOM))

for i in range(N_BATCHES):
    intermediate_gp = Models.GPEI(experiment=exp, data=exp.eval())
    exp.new_trial(generator_run=intermediate_gp.gen(BATCH_SIZE))

model = Models.GPEI(experiment=exp, data=exp.eval())

render(plot_contour(model=model, param_x="x1", param_y="x2", metric_name='hartmann6'))

render(interact_contour(model=model, metric_name='hartmann6'))

render(plot_objective_vs_constraints(model, 'hartmann6', rel=False))

cv_results = cross_validate(model)
render(interact_cross_validation(cv_results))

render(plot_slice(model, "x2", "hartmann6"))

render(interact_fitted(model, rel=False))
