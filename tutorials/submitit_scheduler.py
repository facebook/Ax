# This is a quickstart guide for submitit that links to more advanced resources.
# TODO(marton) Convert this to a jupyter notebook

from ax.runners.submitit import SubmitItRunner, SubmitItMetricFetcher
from ax import RangeParameter, Objective, Experiment, ParameterType, SearchSpace, OptimizationConfig
from ax.service.scheduler import Scheduler, SchedulerOptions
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from submitit import AutoExecutor

# Your function to optimize

def add(x1: float, x2: float) -> float:
    return x1 + x2

# Slurm options

slurm_executor = AutoExecutor(folder='./logdir')
slurm_executor.update_parameters(timeout_min=1, slurm_partition="dev", gpus_per_node=2)

# Parameters

parameters = [
    RangeParameter(
        name="x1",
        parameter_type=ParameterType.FLOAT,
        lower=-5,
        upper=10,
    ),
    RangeParameter(
        name="x2",
        parameter_type=ParameterType.FLOAT,
        lower=0,
        upper=15,
    ),
]

# Experiment setup

experiment = Experiment(
    name="best_numers_to_add_experiment",
    search_space=SearchSpace(parameters=parameters),
    optimization_config=OptimizationConfig(objective=Objective(metric=SubmitItMetricFetcher(), minimize=True)),
    runner=SubmitItRunner(train_evaluate_fn=add, executor=slurm_executor),
)

generation_strategy = choose_generation_strategy(
    search_space=experiment.search_space,
    max_parallelism_cap=3,
)

# Ax scheduler

scheduler = Scheduler(
    experiment=experiment,
    generation_strategy=generation_strategy,
    options=SchedulerOptions(),
)

scheduler.run_n_trials(max_trials=3)