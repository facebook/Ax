from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models, ModelRegistryBase
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.modelbridge_utils import get_pending_observation_features

from ax.utils.testing.core_stubs import get_branin_search_space, get_branin_experiment

gs = GenerationStrategy(
    steps=[
        # 1. Initialization step (does not require pre-existing data and is well-suited for 
        # initial sampling of the search space)
        GenerationStep(
            model=Models.SOBOL,
            num_trials=5,  # How many trials should be produced from this generation step
            min_trials_observed=3, # How many trials need to be completed to move to next model
            max_parallelism=5,  # Max parallelism for this step
            model_kwargs={"seed": 999},  # Any kwargs you want passed into the model
            model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
        ),
        # 2. Bayesian optimization step (requires data obtained from previous phase and learns
        # from all data available at the time of each new candidate generation call)
        GenerationStep(
            model=Models.GPEI,
            num_trials=-1,  # No limitation on how many trials should be produced from this step
            max_parallelism=3,  # Parallelism limit for this step, often lower than for Sobol
            # More on parallelism vs. required samples in BayesOpt:
            # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials
        ),
    ]
)


gs = choose_generation_strategy(
    # Required arguments:
    search_space=get_branin_search_space(),  # Ax `SearchSpace`
    
    # Some optional arguments (shown with their defaults), see API docs for more settings:
    # https://ax.dev/api/modelbridge.html#module-ax.modelbridge.dispatch_utils
    use_batch_trials=False,  # Whether this GS will be used to generate 1-arm `Trial`-s or `BatchTrials`
    no_bayesian_optimization=False,  # Use quasi-random candidate generation without BayesOpt
    max_parallelism_override=None,  # Integer, to which to set the `max_parallelism` setting of all steps in this GS   
)
gs

experiment = get_branin_experiment()

generator_run = gs.gen(
    experiment=experiment, # Ax `Experiment`, for which to generate new candidates
    data=None, # Ax `Data` to use for model training, optional.
    n=1, # Number of candidate arms to produce
    pending_observations=get_pending_observation_features(experiment),  # Points that should not be re-generated
    # Any other kwargs specified will be passed through to `ModelBridge.gen` along with `GenerationStep.model_gen_kwargs`
)
generator_run

trial = experiment.new_trial(generator_run)
trial

print(GenerationStep.__doc__)

from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.decoder import object_from_json

gs_json = object_to_json(gs)  # Can be written to a file or string via `json.dump` etc.
gs = object_from_json(gs_json)  # Decoded back from JSON (can be loaded from file, string via `json.load` etc.)
gs

from ax.storage.sqa_store.save import save_generation_strategy, save_experiment
from ax.storage.sqa_store.load import load_experiment, load_generation_strategy_by_experiment_name

from ax.storage.sqa_store.db import init_engine_and_session_factory,get_engine, create_all_tables
from ax.storage.sqa_store.load import load_experiment
from ax.storage.sqa_store.save import save_experiment

init_engine_and_session_factory(url='sqlite:///foo2.db')

engine = get_engine()
create_all_tables(engine)

save_experiment(experiment)
save_generation_strategy(gs)

experiment = load_experiment(experiment_name=experiment.name)
gs = load_generation_strategy_by_experiment_name(
    experiment_name=experiment.name, 
    experiment=experiment,  # Can optionally specify experiment object to avoid loading it from database twice
)
gs

generator_run = gs.gen(
    experiment=experiment, n=1, pending_observations=get_pending_observation_features(experiment)
)
experiment.new_trial(generator_run)
