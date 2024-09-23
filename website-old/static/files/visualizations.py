import numpy as np

from ax.modelbridge.cross_validation import cross_validate
from ax.plot.contour import interact_contour
from ax.plot.diagnostic import interact_cross_validation
from ax.plot.scatter import interact_fitted, plot_objective_vs_constraints, tile_fitted
from ax.plot.slice import plot_slice
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import init_notebook_plotting, render

init_notebook_plotting()

noise_sd = 0.1
param_names = [f"x{i+1}" for i in range(6)]  # x1, x2, ..., x6


def noisy_hartmann_evaluation_function(parameterization):
    x = np.array([parameterization.get(p_name) for p_name in param_names])
    noise1, noise2 = np.random.normal(0, noise_sd, 2)

    return {
        "hartmann6": (hartmann6(x) + noise1, noise_sd),
        "l2norm": (np.sqrt((x**2).sum()) + noise2, noise_sd),
    }

ax_client = AxClient()
ax_client.create_experiment(
    name="test_visualizations",
    parameters=[
        {
            "name": p_name,
            "type": "range",
            "bounds": [0.0, 1.0],
        }
        for p_name in param_names
    ],
    objectives={"hartmann6": ObjectiveProperties(minimize=True)},
    outcome_constraints=["l2norm <= 1.25"],
)

for i in range(20):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(
        trial_index=trial_index, raw_data=noisy_hartmann_evaluation_function(parameters)
    )

# this could alternately be done with `ax.plot.contour.plot_contour`
render(ax_client.get_contour_plot(param_x="x1", param_y="x2", metric_name="hartmann6"))

model = ax_client.generation_strategy.model
render(interact_contour(model=model, metric_name="hartmann6"))

render(plot_objective_vs_constraints(model, "hartmann6", rel=False))

cv_results = cross_validate(model)
render(interact_cross_validation(cv_results))

render(plot_slice(model, "x2", "hartmann6"))

render(interact_fitted(model, rel=False))

import plotly.io as pio
pio.renderers.default = "jupyterlab"

render(ax_client.get_contour_plot(param_x="x1", param_y="x2", metric_name="hartmann6"))
