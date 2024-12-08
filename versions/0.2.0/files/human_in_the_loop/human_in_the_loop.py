from ax import Data, Metric, OptimizationConfig, Objective, OutcomeConstraint, ComparisonOp, json_load
from ax.modelbridge.cross_validation import cross_validate
from ax.modelbridge.factory import get_GPEI
from ax.plot.diagnostic import tile_cross_validation
from ax.plot.scatter import plot_multiple_metrics, tile_fitted
from ax.utils.notebook.plotting import render, init_notebook_plotting

import pandas as pd

init_notebook_plotting()

experiment = json_load('hitl_exp.json')

experiment.trials[0]

experiment.trials[0].time_created

# Number of arms in first experiment, including status_quo
len(experiment.trials[0].arms)

# Sample arm configuration
experiment.trials[0].arms[0]

experiment.status_quo

objective_metric = Metric(name="metric_1")
constraint_metric = Metric(name="metric_2")

experiment.optimization_config = OptimizationConfig(
    objective=Objective(objective_metric),
    outcome_constraints=[
        OutcomeConstraint(metric=constraint_metric, op=ComparisonOp.LEQ, bound=5),
    ]
)

data = Data(pd.read_json('hitl_data.json'))
data.df.head()

data.df['arm_name'].unique()

data.df['metric_name'].unique()

experiment.search_space.parameters

experiment.search_space.parameter_constraints

gp = get_GPEI(
    experiment=experiment,
    data=data,
)

cv_result = cross_validate(gp)
render(tile_cross_validation(cv_result))

render(tile_fitted(gp, rel=True))

METRIC_X_AXIS = 'metric_1'
METRIC_Y_AXIS = 'metric_2'

render(plot_multiple_metrics(
    gp,
    metric_x=METRIC_X_AXIS,
    metric_y=METRIC_Y_AXIS,
))

unconstrained = gp.gen(
    n=3,
    optimization_config=OptimizationConfig(
        objective=Objective(objective_metric),
    )
)

render(plot_multiple_metrics(
    gp,
    metric_x=METRIC_X_AXIS,
    metric_y=METRIC_Y_AXIS,
    generator_runs_dict={
        'unconstrained': unconstrained,
    }
))

constraint_5 = OutcomeConstraint(metric=constraint_metric, op=ComparisonOp.LEQ, bound=5)
constraint_5_results = gp.gen(
    n=3, 
    optimization_config=OptimizationConfig(
        objective=Objective(objective_metric),
        outcome_constraints=[constraint_5]
    )
)

from ax.plot.scatter import plot_multiple_metrics
render(plot_multiple_metrics(
    gp,
    metric_x=METRIC_X_AXIS,
    metric_y=METRIC_Y_AXIS,
    generator_runs_dict={
        'constraint_5': constraint_5_results
    }
))

constraint_1 = OutcomeConstraint(metric=constraint_metric, op=ComparisonOp.LEQ, bound=1)
constraint_1_results = gp.gen(
    n=3, 
    optimization_config=OptimizationConfig(
        objective=Objective(objective_metric),
        outcome_constraints=[constraint_1],
    )
)

render(plot_multiple_metrics(
    gp,
    metric_x=METRIC_X_AXIS,
    metric_y=METRIC_Y_AXIS,
    generator_runs_dict={
        "constraint_1": constraint_1_results,
    }
))

render(plot_multiple_metrics(
    gp,
    metric_x=METRIC_X_AXIS,
    metric_y=METRIC_Y_AXIS,
    generator_runs_dict={
        'unconstrained': unconstrained,
        'loose_constraint': constraint_5_results,
        'tight_constraint': constraint_1_results,
    }
))

# We can add entire generator runs, when constructing a new trial. 
trial = experiment.new_batch_trial().add_generator_run(unconstrained).add_generator_run(constraint_5_results)

# Or, we can hand-pick arms. 
trial.add_arm(constraint_1_results.arms[0])

experiment.trials[1].arms

experiment.trials[1]._generator_run_structs

experiment.trials[1]._generator_run_structs[0].generator_run.optimization_config
