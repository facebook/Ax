from typing import Any, Dict, NamedTuple, List, Union
from time import time
from random import randint

from ax.core.base_trial import TrialStatus
from ax.utils.measurement.synthetic_functions import branin


class MockJob(NamedTuple):
    """Dummy class to represent a job scheduled on `MockJobQueue`."""
    id: int
    parameters: Dict[str, Union[str, float, int, bool]]


class MockJobQueueClient:
    """Dummy class to represent a job queue where the Ax `Scheduler` will
    deploy trial evaluation runs during optimization.
    """
    
    jobs: Dict[str, MockJob] = {}
    
    def schedule_job_with_parameters(
        self, 
        parameters: Dict[str, Union[str, float, int, bool]]
    ) -> int:
        """Schedules an evaluation job with given parameters and returns job ID.
        """
        # Code to actually schedule the job and produce an ID would go here;
        # using timestamp as dummy ID for this example.
        job_id = int(time())
        self.jobs[job_id] = MockJob(job_id, parameters)
        return job_id
    
    def get_job_status(self, job_id: int) -> TrialStatus:
        """"Get status of the job by a given ID. For simplicity of the example,
        return an Ax `TrialStatus`.
        """
        job = self.jobs[job_id]
        # Instead of randomizing trial status, code to check actual job status
        # would go here.
        if randint(0, 3) > 0:
            return TrialStatus.COMPLETED
        return TrialStatus.RUNNING
    
    def get_outcome_value_for_completed_job(self, job_id: int) -> Dict[str, float]:
        """Get evaluation results for a given completed job."""
        job = self.jobs[job_id]
        # In a real external system, this would retrieve real relevant outcomes and
        # not a synthetic function value.
        return {
            "branin": branin(job.parameters.get("x1"), job.parameters.get("x2"))
        }
    
MOCK_JOB_QUEUE_CLIENT = MockJobQueueClient()

def get_mock_job_queue_client() -> MockJobQueueClient:
    """Obtain the singleton job queue instance."""
    return MOCK_JOB_QUEUE_CLIENT

from ax.core.runner import Runner
from ax.core.base_trial import BaseTrial
from ax.core.trial import Trial

class MockJobRunner(Runner):  # Deploys trials to external system.
    
    def run(self, trial: BaseTrial) -> Dict[str,Any]:
        """Deploys a trial and returns dict of run metadata."""
        if not isinstance(trial, Trial):
            raise ValueError("This runner only handles `Trial`.")
        
        mock_job_queue = get_mock_job_queue_client()
        job_id = mock_job_queue.schedule_job_with_parameters(
            parameters=trial.arm.parameters
        )
        # This run metadata will be attached to trial as `trial.run_metadata`
        # by the base `Scheduler`.
        return {"job_id": job_id}

import pandas as pd

from ax.core.metric import Metric
from ax.core.base_trial import BaseTrial
from ax.core.data import Data

class BraninForMockJobMetric(Metric):  # Pulls data for trial from external system.
    
    def fetch_trial_data(self, trial: BaseTrial) -> Data:
        """Obtains data via fetching it from ` for a given trial."""
        if not isinstance(trial, Trial):
            raise ValueError("This metric only handles `Trial`.")
        
        mock_job_queue = get_mock_job_queue_client()
        
        # Here we leverage the "job_id" metadata created by `MockJobRunner.run`.
        branin_data = mock_job_queue.get_outcome_value_for_completed_job(
            job_id=trial.run_metadata.get("job_id")
        )
        df_dict = {
            "trial_index": trial.index,
            "metric_name": "branin",
            "arm_name": trial.arm.name,
            "mean": branin_data.get("branin"),
            # Can be set to 0.0 if function is known to be noiseless
            # or to an actual value when SEM is known. Setting SEM to
            # `None` results in Ax assuming unknown noise and inferring
            # noise level from data.
            "sem": None,
        }
        return Data(df=pd.DataFrame.from_records([df_dict]))

from ax import *

def make_branin_experiment_with_runner_and_metric() -> Experiment:
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

    objective=Objective(metric=BraninForMockJobMetric(name="branin"), minimize=True)

    return Experiment(
        name="branin_test_experiment",
        search_space=SearchSpace(parameters=parameters),
        optimization_config=OptimizationConfig(objective=objective),
        runner=MockJobRunner(),
        is_test=True,  # Marking this experiment as a test experiment.
    )

experiment = make_branin_experiment_with_runner_and_metric()

from typing import Dict, Set
from random import randint
from collections import defaultdict
from ax.service.scheduler import Scheduler, SchedulerOptions, TrialStatus

class MockJobQueueScheduler(Scheduler):

    def poll_trial_status(self) -> Dict[TrialStatus, Set[int]]:
        """Queries the external system to compute a mapping from trial statuses
        to a set indices of trials that are currently in that status. Only needs
        to query for trials that are currently running but can query for all
        trials too.
        """
        status_dict = defaultdict(set)
        for trial in self.running_trials:  # `running_trials` is exposed on base `Scheduler`
            mock_job_queue = get_mock_job_queue_client()
            status = mock_job_queue.get_job_status(job_id=trial.run_metadata.get("job_id"))
            status_dict[status].add(trial.index)
                
        return status_dict

from ax.modelbridge.dispatch_utils import choose_generation_strategy

generation_strategy = choose_generation_strategy(
    search_space=experiment.search_space, 
    max_parallelism_cap=3,
)

scheduler = MockJobQueueScheduler(
    experiment=experiment,
    generation_strategy=generation_strategy,
    options=SchedulerOptions(),
)

scheduler.run_n_trials(max_trials=3)

from ax.service.utils.report_utils import exp_to_df

exp_to_df(experiment)

scheduler.run_n_trials(max_trials=3)

exp_to_df(experiment)

scheduler.run_n_trials(max_trials=3, timeout_hours=0.00001)

from ax.storage.sqa_store.structs import DBSettings

# URL is of the form "dialect+driver://username:password@host:port/database".
# Instead of URL, can provide a `creator function`; can specify custom encoders/decoders if necessary.
db_settings = DBSettings(url="postgresql+psycopg2://sarah:c82i94d@ocalhost:5432/foobar")

stored_experiment = make_branin_experiment_with_runner_and_metric()
generation_strategy = choose_generation_strategy(search_space=experiment.search_space)
scheduler_with_storage = MockJobQueueScheduler(
    experiment=stored_experiment,
    generation_strategy=generation_strategy,
    options=SchedulerOptions(),
    db_settings=db_settings,
)

stored_experiment.name

reloaded_experiment_scheduler = MockJobQueueScheduler.from_stored_experiment(
    experiment_name='branin_test_experiment',
    options=SchedulerOptions(),
    # `DBSettings` are also required here so scheduler has access to the
    # database, from which it needs to load the experiment.
    db_settings=db_settings,
)

reloaded_experiment_scheduler.run_n_trials(max_trials=3)

print(SchedulerOptions.__doc__)

class ResultReportingScheduler(MockJobQueueScheduler):
    
    def report_results(self):
        return True, {
            "trials so far": len(self.experiment.trials),
            "currently producing trials from generation step": self.generation_strategy._curr.model_name,
            "running trials": [t.index for t in self.running_trials],
        }

experiment = make_branin_experiment_with_runner_and_metric()
scheduler = ResultReportingScheduler(
    experiment=experiment,
    generation_strategy=choose_generation_strategy(
        search_space=experiment.search_space, 
        max_parallelism_cap=3,
    ),
    options=SchedulerOptions(),
)

for reported_result in scheduler.run_trials_and_yield_results(max_trials=6):
  print("Reported result: ", reported_result)
