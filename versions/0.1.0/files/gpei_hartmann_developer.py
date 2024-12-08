
import numpy as np
from ax import (
    ComparisonOp,
    ParameterType, 
    RangeParameter,
    SearchSpace, 
    SimpleExperiment, 
    OutcomeConstraint, 
)
from ax.metrics.l2norm import L2NormMetric
from ax.modelbridge.factory import Models
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()

def hartmann6(x: np.ndarray) -> float:
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = 10 ** (-4) * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )
        y = 0.0
        for j, alpha_j in enumerate(alpha):
            t = 0
            for k in range(6):
                t += A[j, k] * ((x[k] - P[j, k]) ** 2)
            y -= alpha_j * np.exp(-t)
        return y

def hartmann_evaluation_function(
    parameterization, # dict of parameter names to values of those parameters
    weight=None, # evaluation function signature requires a weight argument
):
    x = np.array([parameterization.get(f"x{i}") for i in range(6)])
    # In our case, standard error is 0, since we are computing a synthetic function.
    return {"hartmann6": (hartmann6(x), 0.0), "l2norm": (np.sqrt((x ** 2).sum()), 0.0)}

hartmann_search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name=f"x{i}", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
        )
        for i in range(6)
    ]
)

exp = SimpleExperiment(
    name="test_branin",
    search_space=hartmann_search_space,
    evaluation_function=hartmann_evaluation_function,
    objective_name="hartmann6",
    minimize=True,
    outcome_constraints=[
        OutcomeConstraint(
            metric=L2NormMetric(
                name="l2norm", param_names=[f"x{i}" for i in range(6)], noise_sd=0.2
            ),
            op=ComparisonOp.LEQ,
            bound=1.25,
            relative=False,
        )
    ],
)

print(f"Running Sobol initialization trials...")
sobol = Models.SOBOL(exp.search_space)
for i in range(5):
    exp.new_trial(generator_run=sobol.gen(1))
    
for i in range(15):
    print(f"Running GP+EI optimization trial {i+1}/15...")
    # Reinitialize GP+EI model at each step with updated data.
    gpei = Models.GPEI(experiment=exp, data=exp.eval())
    batch = exp.new_trial(generator_run=gpei.gen(1))
    
print("Done!")

trial_data = exp.eval_trial(exp.trials[1])
trial_data.df

# `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple 
# optimization runs, so we wrap out best objectives array in another array.
objective_means = np.array([[trial.objective_mean for trial in exp.trials.values()]])
best_objective_plot = optimization_trace_single_method(
        y=np.minimum.accumulate(objective_means, axis=1),
        optimum=-3.32237,  # Known minimum objective for Hartmann6 function.
)
render(best_objective_plot)
