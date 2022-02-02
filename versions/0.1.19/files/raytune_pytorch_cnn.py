import logging

from ray import tune
from ray.tune import track
from ray.tune.suggest.ax import AxSearch

logger = logging.getLogger(tune.__name__)
logger.setLevel(
    level=logging.CRITICAL
)  # Reduce the number of Ray warnings that are not relevant here.

import numpy as np
import torch
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.ax_client import AxClient
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.utils.tutorials.cnn_utils import CNN, evaluate, load_mnist, train

init_notebook_plotting()

ax = AxClient(enforce_sequential_optimization=False)

ax.create_experiment(
    name="mnist_experiment",
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
        {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
    ],
    objective_name="mean_accuracy",
)

def train_evaluate(parameterization):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, test_loader = load_mnist(data_path="~/.data")
    net = train(
        net=CNN(),
        train_loader=train_loader,
        parameters=parameterization,
        dtype=torch.float,
        device=device,
    )
    track.log(
        mean_accuracy=evaluate(
            net=net,
            data_loader=valid_loader,
            dtype=torch.float,
            device=device,
        )
    )

tune.run(
    train_evaluate,
    num_samples=30,
    search_alg=AxSearch(ax_client=ax, max_concurrent=3),
    verbose=0,  # Set this level to 1 to see status updates and to 2 to also see trial results.
    # To use GPU, specify: resources_per_trial={"gpu": 1}.
)

best_parameters, values = ax.get_best_parameters()
best_parameters

means, covariances = values
means

render(
    plot_contour(
        model=ax.generation_strategy.model,
        param_x="lr",
        param_y="momentum",
        metric_name="mean_accuracy",
    )
)

# `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple
# optimization runs, so we wrap out best objectives array in another array.
best_objectives = np.array(
    [[trial.objective_mean * 100 for trial in ax.experiment.trials.values()]]
)
best_objective_plot = optimization_trace_single_method(
    y=np.maximum.accumulate(best_objectives, axis=1),
    title="Model performance vs. # of iterations",
    ylabel="Accuracy",
)
render(best_objective_plot)
