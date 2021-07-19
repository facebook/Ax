import pandas as pd
from ax import *

import torch
import numpy as np

from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.service.utils.report_utils import exp_to_df
from ax.runners.synthetic import SyntheticRunner

# Plotting imports and initialization
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.contour import plot_contour
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier

init_notebook_plotting()

# Model registry for creating multi-objective optimization models.
from ax.modelbridge.registry import Models

# Analysis utilities, including a method to evaluate hypervolumes
from ax.modelbridge.modelbridge_utils import observed_hypervolume

from botorch.test_functions.multi_objective import DTLZ2
d = 10
tkwargs = {
    "dtype": torch.double, 
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
problem = DTLZ2(num_objectives=2, dim=d, negate=True).to(**tkwargs)

search_space = SearchSpace(
    parameters=[RangeParameter(name=f"x{i}", lower=0, upper=1, parameter_type=ParameterType.FLOAT) for i in range(d)],
)

param_names = [f"x{i}" for i in range(d)]

def f1(x) -> float:
        x_sorted = [x[p_name] for p_name in param_names]
        return float(problem(torch.tensor(x_sorted, **tkwargs).clamp(0., 1.))[0])
    
def f2(x) -> float:
        x_sorted = [x[p_name] for p_name in param_names]
        return float(problem(torch.tensor(x_sorted, **tkwargs).clamp(0., 1.))[1])

metric_a = GenericNoisyFunctionMetric("a", f=f1, noise_sd=0.0, lower_is_better=False)
metric_b = GenericNoisyFunctionMetric("b", f=f2, noise_sd=0.0, lower_is_better=False)

mo = MultiObjective(
    objectives=[Objective(metric=metric_a), Objective(metric=metric_b)],
)

objective_thresholds = [
    ObjectiveThreshold(metric=metric, bound=val, relative=False)
    for metric, val in zip(mo.metrics, problem.ref_point)
]

optimization_config = MultiObjectiveOptimizationConfig(
    objective=mo,
    objective_thresholds=objective_thresholds,
)

N_INIT = 2*(d+1)
N_BATCH = 10
BATCH_SIZE = 4

def build_experiment():
    experiment = Experiment(
        name="pareto_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )
    return experiment

## Initialize with Sobol samples

def initialize_experiment(experiment):
    sobol = Models.SOBOL(search_space=experiment.search_space)

    for _ in range(N_INIT):
        experiment.new_trial(sobol.gen(1)).run()

    return experiment.fetch_data()

experiment = build_experiment()
data = initialize_experiment(experiment)

from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
hv_list = []
model = None
for i in range(N_BATCH):   
    model = Models.FULLYBAYESIANMOO(
        experiment=experiment, 
        data=data,
        use_saas=True,  # use SAAS priors (SAASBO)
        # use fewer num_samples and warmup_steps to speed up this tutorial
        num_samples=256,
        warmup_steps=512,
        torch_device=tkwargs["device"],
    )
    generator_run = model.gen(BATCH_SIZE)
    trial = experiment.new_batch_trial(generator_run=generator_run)
    trial.run()
    data = Data.from_multiple_data([data, trial.fetch_data()])
    
    exp_df = exp_to_df(experiment)
    outcomes = torch.tensor(exp_df[['a', 'b']].values, **tkwargs)
    partitioning = DominatedPartitioning(ref_point=problem.ref_point, Y=outcomes)
    try:
        hv = partitioning.compute_hypervolume().item()
    except:
        hv = 0
        print("Failed to compute hv")
    hv_list.append(hv)
    print(f"Iteration: {i}, HV: {hv}")

outcomes = np.array(exp_to_df(experiment)[['a', 'b']], dtype=np.double)

import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

from matplotlib.cm import ScalarMappable
fig, axes = plt.subplots(1, 1, figsize=(8,6))
algos = ["qNEHVI"]
train_obj = outcomes
cm = plt.cm.get_cmap('viridis')

n_results = N_BATCH*BATCH_SIZE + N_INIT
batch_number = torch.cat([torch.zeros(N_INIT), torch.arange(1, N_BATCH+1).repeat(BATCH_SIZE, 1).t().reshape(-1)]).numpy()

sc = axes.scatter(train_obj[:n_results, 0], train_obj[:n_results,1], c=batch_number[:n_results], alpha=0.8)
axes.set_title(algos[0])
axes.set_xlabel("Objective 1")
axes.set_ylabel("Objective 2")
norm = plt.Normalize(batch_number.min(), batch_number.max())
sm =  ScalarMappable(norm=norm, cmap=cm)
sm.set_array([])
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title("Iteration")

import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

iters = np.arange(1, N_BATCH + 1)
log_hv_difference = np.log10(problem.max_hv - np.asarray(hv_list))[:N_BATCH + 1]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(iters, log_hv_difference, label="qNEHVI+SAASBO", linewidth=1.5)
ax.set(xlabel='Batch Iterations', ylabel='Log Hypervolume Difference')
ax.legend(loc="lower right")

from ax.modelbridge.cross_validation import cross_validate
from ax.plot.diagnostic import tile_cross_validation

cv = cross_validate(model)

render(tile_cross_validation(cv))

from ax.modelbridge.cross_validation import compute_diagnostics

# compute  out-of-sample log likelihood
compute_diagnostics(cv)["Log likelihood"]

map_model = Models.GPEI(experiment=experiment, data=data)

map_cv = cross_validate(map_model)
render(tile_cross_validation(map_cv))

# compute out-of-sample log likelihood
compute_diagnostics(map_cv)["Log likelihood"]


