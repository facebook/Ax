from typing import Any, Dict, Optional, Tuple, Type

# Ax wrappers for BoTorch components
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
from ax.models.torch.botorch_modular.acquisition import Acquisition

# Ax data tranformation layer
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.registry import Cont_X_trans, Y_trans, Models

# Experiment examination utilities
from ax.service.utils.report_utils import exp_to_df

# Test Ax objects
from ax.utils.testing.core_stubs import (
    get_branin_experiment, 
    get_branin_data, 
    get_branin_experiment_with_multi_objective,
    get_branin_data_multi_objective,
)

# BoTorch components
from botorch.models.model import Model
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

experiment = get_branin_experiment(with_trial=True)
data = get_branin_data(trials=[experiment.trials[0]])

# `Models` automatically selects a model + model bridge combination. 
# For `BOTORCH_MODULAR`, it will select `BoTorchModel` and `TorchModelBridge`.
model_bridge_with_GPEI = Models.BOTORCH_MODULAR(
    experiment=experiment,
    data=data,
    surrogate=Surrogate(SingleTaskGP),  # Optional, will use default if unspecified
    botorch_acqf_class=qNoisyExpectedImprovement,  # Optional, will use default if unspecified
)

generator_run = model_bridge_with_GPEI.gen(n=1)
generator_run.arms[0]

# The surrogate is not specified, so it will be auto-selected
# during `model.fit`.
GPEI_model = BoTorchModel(botorch_acqf_class=qExpectedImprovement)

# The acquisition class is not specified, so it will be 
# auto-selected during `model.gen` or `model.evaluate_acquisition`
GPEI_model = BoTorchModel(surrogate=Surrogate(FixedNoiseGP))

# Both the surrogate and acquisition class will be auto-selected.
GPEI_model = BoTorchModel()

model = BoTorchModel(
    # Optional `Surrogate` specification to use instead of default
    surrogate=Surrogate(
        # BoTorch `Model` type
        botorch_model_class=FixedNoiseGP,
        # Optional, MLL class with which to optimize model parameters
        mll_class=ExactMarginalLogLikelihood,
        # Optional, dictionary of keyword arguments to underlying 
        # BoTorch `Model` constructor
        model_options={}
    ),
    # Optional options to pass to auto-picked `Surrogate` if not
    # specifying the `surrogate` argument
    surrogate_options={},
    
    # Optional BoTorch `AcquisitionFunction` to use instead of default
    botorch_acqf_class=qExpectedImprovement,
    # Optional dict of keyword arguments, passed to the input 
    # constructor for the given BoTorch `AcquisitionFunction`
    acquisition_options={},
    # Optional Ax `Acquisition` subclass (if the given BoTorch
    # `AcquisitionFunction` requires one, which is rare)
    acquisition_class=None,
    
    # Less common model settings shown with default values, refer
    # to `BoTorchModel` documentation for detail
    refit_on_update=True,
    refit_on_cv=False,
    warm_start_refit=True,
)

from_botorch_model = BoTorchModel(
    surrogate=Surrogate.from_botorch(
        # Pre-constructed BoTorch `Model` instance, with training data already set
        model=...,  
        # Optional, MLL class with which to optimize model parameters
        mll_class=ExactMarginalLogLikelihood,
    )
)

from botorch.models.model import Model
from botorch.utils.containers import TrainingData

class MyModelClass(Model):
    
    ...   # Other contents of the `Model` type
    
    @classmethod
    def construct_inputs(cls, training_data: TrainingData, **kwargs) -> Dict[str, Any]:
        fidelity_features = kwargs.get("fidelity_features")
        if fidelity_features is None:
            raise ValueError(f"Fidelity features required for {cls.__name__}.")

        return {
            "train_X": training_data.X,
            "train_Y": training_data.Y,
            "fidelity_features": fidelity_features,
        }

surrogate = Surrogate(
    botorch_model_class=MyModelClass,  # Must implement `construct_inputs`
    # Optional dict of additional keyword arguments to `MyModelClass`
    model_options={},
)

class MyOtherModelClass(MyModelClass):
    pass

surrogate = ListSurrogate(
    botorch_submodel_class_per_outcome={
        "metric_a": MyModelClass, 
        "metric_b": MyOtherModelClass,
    },
    submodel_options_per_outcome={"metric_a": {}, "metric_b": {}},
)

surrogate = ListSurrogate(
    # Shortcut if all submodels are the same type
    botorch_submodel_class=MyModelClass,
    # Shortcut if all submodel options are the same
    submodel_options={},
)

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import acqf_input_constructor
from botorch.acquisition.objective import AcquisitionObjective
from torch import Tensor

from ax.models.torch.botorch_modular.default_options import register_default_optimizer_options


class MyAcquisitionFunctionClass(AcquisitionFunction):
    ...  # Actual contents of the acquisition function class.
    

# 1. Add input constructor    
@acqf_input_constructor(MyAcquisitionFunctionClass)
def construct_inputs_my_acqf(
    model: Model,
    training_data: TrainingData,
    objective_thresholds: Tensor,
    objective: Optional[AcquisitionObjective] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    pass

# 2. Register default optimizer options
register_default_optimizer_options(
    acqf_class=MyAcquisitionFunctionClass,
    default_options={},
)


# 3-4. Specifying `botorch_acqf_class` and `acquisition_options`
BoTorchModel(    
    botorch_acqf_class=MyAcquisitionFunctionClass,
    acquisition_options={
        "alpha": 10 ** -6, 
        # The sub-dict by the key "optimizer_options" can be passed
        # to propagate options to `optimize_acqf`, used in
        # `Acquisition.optimize`, to add/override the default
        # optimizer options registered above.
        "optimizer_options": {"sequential": False},
    },
)

model_bridge_with_GPEI = Models.BOTORCH_MODULAR(
    experiment=experiment, data=data,
)
model_bridge_with_GPEI.gen(1)

model_bridge_with_GPEI.model.botorch_acqf_class

model_bridge_with_GPEI.model.surrogate.botorch_model_class

model_bridge_with_EHVI = Models.MOO_MODULAR(
    experiment=get_branin_experiment_with_multi_objective(has_objective_thresholds=True, with_batch=True),
    data=get_branin_data_multi_objective(),
)
model_bridge_with_EHVI.gen(1)

model_bridge_with_EHVI.model.botorch_acqf_class

model_bridge_with_EHVI.model.surrogate.botorch_model_class

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from botorch.acquisition import UpperConfidenceBound
from ax.modelbridge.modelbridge_utils import get_pending_observation_features

gs = GenerationStrategy(
    steps=[
        GenerationStep(  # Initialization step
            # Which model to use for this step
            model=Models.SOBOL,
            # How many generator runs (each of which is then made a trial) 
            # to produce with this step
            num_trials=5,
            # How many trials generated from this step must be `COMPLETED` 
            # before the next one
            min_trials_observed=5, 
        ),
        GenerationStep(  # BayesOpt step
            model=Models.BOTORCH_MODULAR,
            # No limit on how many generator runs will be produced
            num_trials=-1,
            model_kwargs={  # Kwargs to pass to `BoTorchModel.__init__`
                "surrogate": Surrogate(SingleTaskGP),
                "botorch_acqf_class": qNoisyExpectedImprovement,
            },
        )
    ]
)

experiment = get_branin_experiment(minimize=True)

assert len(experiment.trials) == 0
experiment.search_space

for _ in range(10):
    # Produce a new generator run and attach it to experiment as a trial
    generator_run = gs.gen(
        experiment=experiment, 
        n=1, 
    )
    trial = experiment.new_trial(generator_run)
    
    # Mark the trial as 'RUNNING' so we can mark it 'COMPLETED' later
    trial.mark_running(no_runner_required=True)
    
    # Attach data for the new trial and mark it 'COMPLETED'
    experiment.attach_data(get_branin_data(trials=[trial]))
    trial.mark_completed()
    
    print(f"Completed trial #{trial.index}, suggested by {generator_run._model_key}.")

exp_to_df(experiment)

class CustomObjectiveAcquisition(Acquisition):
    
    def get_botorch_objective(
        self,
        botorch_acqf_class: Type[AcquisitionFunction],
        model: Model,
        objective_weights: Tensor,
        objective_thresholds: Optional[Tensor] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        X_observed: Optional[Tensor] = None,
    ) -> AcquisitionObjective:
        ...  # Produce the desired `AcquisitionObjective` instead of the default

Models.BOTORCH_MODULAR(
    experiment=experiment, 
    data=data,
    acquisition_class=CustomObjectiveAcquisition,
    botorch_acqf_class=MyAcquisitionFunctionClass,
)
