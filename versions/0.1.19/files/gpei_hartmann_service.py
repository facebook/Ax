from ax.service.ax_client import AxClient
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()

ax_client = AxClient()

ax_client.create_experiment(
    name="hartmann_test_experiment",
    parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x3",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x4",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x5",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x6",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
    ],
    objective_name="hartmann6",
    minimize=True,  # Optional, defaults to False.
    parameter_constraints=["x1 + x2 <= 2.0"],  # Optional.
    outcome_constraints=["l2norm <= 1.25"],  # Optional.
)

import numpy as np
def evaluate(parameters):
    x = np.array([parameters.get(f"x{i+1}") for i in range(6)])
    # In our case, standard error is 0, since we are computing a synthetic function.
    return {"hartmann6": (hartmann6(x), 0.0), "l2norm": (np.sqrt((x ** 2).sum()), 0.0)}

for i in range(25):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

ax_client.get_max_parallelism()

ax_client.generation_strategy.trials_as_df

best_parameters, values = ax_client.get_best_parameters()
best_parameters

means, covariances = values
means 

hartmann6.fmin

render(ax_client.get_contour_plot())

render(ax_client.get_contour_plot(param_x="x3", param_y="x4", metric_name="l2norm"))

render(ax_client.get_optimization_trace(objective_optimum=hartmann6.fmin))  # Objective_optimum is optional.

ax_client.save_to_json_file()  # For custom filepath, pass `filepath` argument.

restored_ax_client = AxClient.load_from_json_file()  # For custom filepath, pass `filepath` argument.

from ax.storage.sqa_store.structs import DBSettings

# URL is of the form "dialect+driver://username:password@host:port/database".
db_settings = DBSettings(url="postgresql+psycopg2://sarah:c82i94d@ocalhost:5432/foobar")
# Instead of URL, can provide a `creator function`; can specify custom encoders/decoders if necessary.
new_ax = AxClient(db_settings=db_settings)

_, trial_index = ax_client.get_next_trial()
ax_client.log_trial_failure(trial_index=trial_index)

ax_client.attach_trial(parameters={"x1": 0.9, "x2": 0.9, "x3": 0.9, "x4": 0.9, "x5": 0.9, "x6": 0.9})

ax_client = AxClient()
ax_client.create_experiment(
    parameters=[
        {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
        {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
    ],
    # Sets max parallelism to 10 for all steps of the generation strategy.
    choose_generation_strategy_kwargs={"max_parallelism_override": 10},
)

ax_client.get_max_parallelism()  # Max parallelism is now 10 for all stages of the optimization.
