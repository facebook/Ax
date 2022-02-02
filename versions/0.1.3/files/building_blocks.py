
import pandas as pd
from ax import *

range_param1 = RangeParameter(name="x1", lower=0.0, upper=10.0, parameter_type=ParameterType.FLOAT)
range_param2 = RangeParameter(name="x2", lower=0.0, upper=10.0, parameter_type=ParameterType.FLOAT)

search_space = SearchSpace(
    parameters=[range_param1, range_param2],
)

choice_param = ChoiceParameter(name="choice", values=["foo", "bar"], parameter_type=ParameterType.STRING)
fixed_param = FixedParameter(name="fixed", value=[True], parameter_type=ParameterType.BOOL)

sum_constraint = SumConstraint(
    parameters=[range_param1, range_param2], 
    is_upper_bound=True, 
    bound=5.0,
)

order_constraint = OrderConstraint(
    lower_parameter = range_param1,
    upper_parameter = range_param2,
)

experiment = Experiment(
    name="experiment_building_blocks",
    search_space=search_space,
)

experiment.status_quo = Arm(
    name="control", 
    parameters={"x1": 0.0, "x2": 0.0},
)

sobol = Models.SOBOL(search_space=experiment.search_space)
generator_run = sobol.gen(5)

for arm in generator_run.arms:
    print(arm)

Models.SOBOL.view_kwargs()  # Shows keyword argument names and typing.

class BoothMetric(Metric):
    def fetch_trial_data(self, trial):  
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": (params["x1"] + 2*params["x2"] - 7)**2 + (2*params["x1"] + params["x2"] - 5)**2,
                "sem": 0.0,
                "trial_index": trial.index,
            })
        return Data(df=pd.DataFrame.from_records(records))

optimization_config = OptimizationConfig(
    objective = Objective(
        metric=BoothMetric(name="booth"), 
        minimize=True,
    ),
)

experiment.optimization_config = optimization_config

outcome_constraint = OutcomeConstraint(
    metric=Metric("constraint"), 
    op=ComparisonOp.LEQ, 
    bound=0.5,
)

class MyRunner(Runner):
    def run(self, trial):
        return {"name": str(trial.index)}
    
experiment.runner = MyRunner()

experiment.new_batch_trial(generator_run=generator_run)

for arm in experiment.trials[0].arms:
    print(arm)

experiment.new_trial().add_arm(Arm(name='single_arm', parameters={'x1': 1, 'x2': 1}))

print(experiment.trials[1].arm)

experiment.trials[0].run()

data = experiment.fetch_data()

data.df

gpei = Models.BOTORCH(experiment=experiment, data=data)
generator_run = gpei.gen(5)
experiment.new_batch_trial(generator_run=generator_run)

for arm in experiment.trials[2].arms:
    print(arm)

experiment.trials[2].run()
data = experiment.fetch_data()
data.df

from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner

register_metric(BoothMetric)
register_runner(MyRunner)

save(experiment, "experiment.json")

loaded_experiment = load("experiment.json")

from ax.storage.sqa_store.db import init_engine_and_session_factory,get_engine, create_all_tables
from ax.storage.sqa_store.load import load_experiment
from ax.storage.sqa_store.save import save_experiment

init_engine_and_session_factory(url='sqlite:///foo.db')

engine = get_engine()
create_all_tables(engine)

save_experiment(experiment)

load_experiment(experiment.name)

def evaluation_function(params):
    return (params["x1"] + 2*params["x2"] - 7)**2 + (2*params["x1"] + params["x2"] - 5)**2

simple_experiment = SimpleExperiment(
    search_space=search_space,
    evaluation_function=evaluation_function,
)

simple_experiment.new_trial().add_arm(Arm(name='single_arm', parameters={'x1': 1, 'x2': 1}))

data = simple_experiment.fetch_data()

data.df
