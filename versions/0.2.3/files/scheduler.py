#!/usr/bin/env python
# coding: utf-8

# # Configurable closed-loop optimization with Ax `Scheduler`
# 
# *We recommend reading through the ["Developer API" tutorial](https://ax.dev/tutorials/gpei_hartmann_developer.html) before getting started with the `Scheduler`, as using it in this tutorial will require an Ax `Experiment` and an understanding of the experiment's subcomponents like the search space and the runner.*
# 
# ### Contents:
# 1. **Scheduler and external systems for trial evalution** –– overview of how scheduler works with an external system to run a closed-loop optimization.
# 2. **Set up a mock external system** –– creating a dummy external system client, which will be used to illustrate a scheduler setup in this tutorial.
# 3. **Set up an experiment according to the mock external system** –– set up a runner that deploys trials to the dummy external system from part 2 and a metric that fetches trial results from that system, then leverage those runner and metric and set up an experiment.
# 4. **Set up a scheduler**, given an experiment.
#    1. Create a scheduler subclass to poll trial status.
#    2. Set up a generation strategy using an auto-selection utility.
# 5. **Running the optimization** via `Scheduler.run_n_trials`.
# 6. **Leveraging SQL storage and experiment resumption** –– resuming an experiment in one line of code.
# 7. **Configuring the scheduler** –– overview of the many options scheduler provides to configure the closed-loop down to granular detail.
# 8. **Advanced functionality**:
#    1. Reporting results to an external system during the optimization.
#    2. Using `Scheduler.run_trials_and_yield_results` to run the optimization via a generator method.

# ## 1. `Scheduler` and external systems for trial evaluation
# 
# `Scheduler` is a closed-loop manager class in Ax that continuously deploys trial runs to an arbitrary external system in an asynchronous fashion, polls their status from that system, and leverages known trial results to generate more trials.
# 
# Key features of the `Scheduler`:
# - Maintains user-set concurrency limits for trials run in parallel, keep track of tolerated level of failed trial runs, and 'oversee' the optimization in other ways,
# - Leverages an Ax `Experiment` for optimization setup (an optimization config with metrics, a search space, a runner for trial evaluations),
# - Uses an Ax `GenerationStrategy` for flexible specification of an optimization algorithm used to generate new trials to run,
# - Supports SQL storage and allows for easy resumption of stored experiments.

# This scheme summarizes how the scheduler interacts with any external system used to run trial evaluations:
# 
# ![image-2.png](attachment:image-2.png)

# ## 2. Set up a mock external execution system 

# An example of an 'external system' running trial evaluations could be a remote server executing scheduled jobs, a subprocess conducting ML training runs, an engine running physics simulations, etc. For the sake of example here, let us assume a dummy external system with the following client:

# In[1]:


from random import randint
from time import time
from typing import Any, Dict, NamedTuple, Union

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
        self, parameters: Dict[str, Union[str, float, int, bool]]
    ) -> int:
        """Schedules an evaluation job with given parameters and returns job ID."""
        # Code to actually schedule the job and produce an ID would go here;
        # using timestamp as dummy ID for this example.
        job_id = int(time())
        self.jobs[job_id] = MockJob(job_id, parameters)
        return job_id

    def get_job_status(self, job_id: int) -> TrialStatus:
        """ "Get status of the job by a given ID. For simplicity of the example,
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
        return {"branin": branin(job.parameters.get("x1"), job.parameters.get("x2"))}


MOCK_JOB_QUEUE_CLIENT = MockJobQueueClient()


def get_mock_job_queue_client() -> MockJobQueueClient:
    """Obtain the singleton job queue instance."""
    return MOCK_JOB_QUEUE_CLIENT


# ## 3. Set up an experiment according to the mock external system
# 
# As mentioned above, using a `Scheduler` requires a fully set up experiment with metrics and a runner. Refer to the "Building Blocks of Ax" tutorial to learn more about those components, as here we assume familiarity with them. 
# 
# The following runner and metric set up intractions between the `Scheduler` and the mock external system we assume:

# In[2]:


from collections import defaultdict
from typing import Iterable, Set

from ax.core.base_trial import BaseTrial
from ax.core.runner import Runner
from ax.core.trial import Trial


class MockJobRunner(Runner):  # Deploys trials to external system.
    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        """Deploys a trial based on custom runner subclass implementation.

        Args:
            trial: The trial to deploy.

        Returns:
            Dict of run metadata from the deployment process.
        """
        if not isinstance(trial, Trial):
            raise ValueError("This runner only handles `Trial`.")

        mock_job_queue = get_mock_job_queue_client()
        job_id = mock_job_queue.schedule_job_with_parameters(
            parameters=trial.arm.parameters
        )
        # This run metadata will be attached to trial as `trial.run_metadata`
        # by the base `Scheduler`.
        return {"job_id": job_id}

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
        """Checks the status of any non-terminal trials and returns their
        indices as a mapping from TrialStatus to a list of indices. Required
        for runners used with Ax ``Scheduler``.

        NOTE: Does not need to handle waiting between polling calls while trials
        are running; this function should just perform a single poll.

        Args:
            trials: Trials to poll.

        Returns:
            A dictionary mapping TrialStatus to a list of trial indices that have
            the respective status at the time of the polling. This does not need to
            include trials that at the time of polling already have a terminal
            (ABANDONED, FAILED, COMPLETED) status (but it may).
        """
        status_dict = defaultdict(set)
        for trial in trials:
            mock_job_queue = get_mock_job_queue_client()
            status = mock_job_queue.get_job_status(
                job_id=trial.run_metadata.get("job_id")
            )
            status_dict[status].add(trial.index)

        return status_dict


# In[3]:


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


# Now we can set up the experiment using the runner and metric we defined. This experiment will have a single-objective optimization config, minimizing the Branin function, and the search space that corresponds to that function.

# In[4]:


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


# ## 4. Setting up a `Scheduler`
# 
# ### 4a. Subclassing `Scheduler`
# 
# The base Ax `Scheduler` is abstract and must be subclassed, but only one method must be implemented on the subclass: `poll_trial_status`. 

# ### 4B. Auto-selecting a generation strategy
# 
# A `Scheduler` also requires an Ax `GenerationStrategy` specifying the algorithm to use for the optimization. Here we use the `choose_generation_strategy` utility that auto-picks a generation strategy based on the search space properties. To construct a custom generation strategy instead, refer to the ["Generation Strategy" tutorial](https://ax.dev/tutorials/generation_strategy.html).
# 
# Importantly, a generation strategy in Ax limits allowed parallelism levels for each generation step it contains. If you would like the `Scheduler` to ensure parallelism limitations, set `max_examples` on each generation step in your generation strategy.

# In[5]:


from ax.modelbridge.dispatch_utils import choose_generation_strategy

generation_strategy = choose_generation_strategy(
    search_space=experiment.search_space, 
    max_parallelism_cap=3,
)


# Now we have all the components needed to start the scheduler:

# In[6]:


from ax.service.scheduler import Scheduler, SchedulerOptions


scheduler = Scheduler(
    experiment=experiment,
    generation_strategy=generation_strategy,
    options=SchedulerOptions(),
)


# ## 5. Running the optimization
# 
# Once the `Scheduler` instance is set up, user can execute `run_n_trials` as many times as needed, and each execution will add up to the specified `max_trials` trials to the experiment. The number of trials actually run might be less than `max_trials` if the optimization was concluded (e.g. there are no more points in the search space).

# In[7]:


scheduler.run_n_trials(max_trials=3)


# We can examine `experiment` to see that it now has three trials:

# In[8]:


from ax.service.utils.report_utils import exp_to_df

exp_to_df(experiment)


# Now we can run `run_n_trials` again to add three more trials to the experiment.

# In[9]:


scheduler.run_n_trials(max_trials=3)


# Examiniming the experiment, we now see 6 trials, one of which is produced by Bayesian optimization (GPEI):

# In[10]:


exp_to_df(experiment)


# For each call to `run_n_trials`, one can specify a timeout; if `run_n_trials` has been running for too long without finishing its `max_trials`, the operation will exit gracefully:

# In[11]:


scheduler.run_n_trials(max_trials=3, timeout_hours=0.00001)


# ## 6. Leveraging SQL storage and experiment resumption
# 
# When a scheduler is SQL-enabled, it will automatically save all updates it makes to the experiment in the course of the optimization. The experiment can then be resumed in the event of a crash or after a pause.
# 
# To store state of optimization to an SQL backend, first follow [setup instructions](https://ax.dev/docs/storage.html#sql) on Ax website. Having set up the SQL backend, pass `DBSettings` to the `Scheduler` on instantiation (note that SQLAlchemy dependency will have to be installed – for installation, refer to [optional dependencies](https://ax.dev/docs/installation.html#optional-dependencies) on Ax website):

# In[12]:


from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner
from ax.storage.sqa_store.db import (
    create_all_tables,
    get_engine,
    init_engine_and_session_factory,
)
from ax.storage.sqa_store.structs import DBSettings


register_metric(BraninForMockJobMetric)
register_runner(MockJobRunner)

# URL is of the form "dialect+driver://username:password@host:port/database".
# Instead of URL, can provide a `creator function`; can specify custom encoders/decoders if necessary.
db_settings = DBSettings(url="sqlite:///foo.db")

# The following lines are only necessary because it is the first time we are using this database
# in practice, you will not need to run these lines every time you initialize your scheduler
init_engine_and_session_factory(url=db_settings.url)
engine = get_engine()
create_all_tables(engine)

stored_experiment = make_branin_experiment_with_runner_and_metric()
generation_strategy = choose_generation_strategy(search_space=experiment.search_space)

scheduler_with_storage = Scheduler(
    experiment=stored_experiment,
    generation_strategy=generation_strategy,
    options=SchedulerOptions(),
    db_settings=db_settings,
)


# In[13]:


stored_experiment.name


# To resume a stored experiment:

# In[14]:


reloaded_experiment_scheduler = Scheduler.from_stored_experiment(
    experiment_name="branin_test_experiment",
    options=SchedulerOptions(),
    # `DBSettings` are also required here so scheduler has access to the
    # database, from which it needs to load the experiment.
    db_settings=db_settings,
)


# With the newly reloaded experiment, the `Scheduler` can continue the optimization:

# In[15]:


reloaded_experiment_scheduler.run_n_trials(max_trials=3)


# ## 7. Configuring the scheduler
# 
# `Scheduler` exposes many options to configure the exact settings of the closed-loop optimization to perform. A few notable ones are:
# - `trial_type` –– currently only `Trial` and not `BatchTrial` is supported, but support for `BatchTrial`-s will follow,
# - `tolerated_trial_failure_rate` and `min_failed_trials_for_failure_rate_check` –– together these two settings control how the scheduler monitors the failure rate among trial runs it deploys. Once `min_failed_trials_for_failure_rate_check` is deployed, the scheduler will start checking whether the ratio of failed to total trials is greater than `tolerated_trial_failure_rate`, and if it is, scheduler will exit the optimization with a `FailureRateExceededError`,
# - `ttl_seconds_for_trials` –– sometimes a failure in a trial run means that it will be difficult to query its status (e.g. due to a crash). If this setting is specified, the Ax `Experiment` will automatically mark trials that have been running for too long (more than their 'time-to-live' (TTL) seconds) as failed,
# - `run_trials_in_batches` –– if `True`, the scheduler will attempt to run trials not by calling `Scheduler.run_trial` in a loop, but by calling `Scheduler.run_trials` on all ready-to-deploy trials at once. This could allow for saving compute in cases where the deployment operation has large overhead and deploying many trials at once saves compute. Note that using this option successfully will require your scheduler subclass to implement `MySchedulerSubclass.run_trials` and `MySchedulerSubclass.poll_available_capacity`.
# 
# The rest of the options is described in the docstring below:

# In[16]:


print(SchedulerOptions.__doc__)


# ## 8. Advanced functionality
# 
# ### 8a. Reporting results to an external system
# 
# The `Scheduler` can report the optimization result to an external system each time there are new completed trials if the user-implemented subclass implements `MySchedulerSubclass.report_results` to do so. For example, the folliwing method:
# 
# ```
# class MySchedulerSubclass(Scheduler):
#     ...
#     
#     def report_results(self):
#         write_to_external_database(len(self.experiment.trials))
#         return (True, {})  # Returns optimization success status and optional dict of outputs.
# ```
# could be used to record number of trials in experiment so far in an external database.
# 
# Since `report_results` is an instance method, it has access to `self.experiment` and `self.generation_strategy`, which contain all the information about the state of the optimization thus far.

# ### 8b. Using `run_trials_and_yield_results` generator method
# 
# In some systems it's beneficial to have greater control over `Scheduler.run_n_trials` instead of just starting it and needing to wait for it to run all the way to completion before having access to its output. For this purpose, the `Scheduler` implements a generator method `run_trials_and_yield_results`, which yields the output of `Scheduler.report_results` each time there are new completed trials and can be used like so:

# In[17]:


class ResultReportingScheduler(Scheduler):
    def report_results(self):
        return True, {
            "trials so far": len(self.experiment.trials),
            "currently producing trials from generation step": self.generation_strategy._curr.model_name,
            "running trials": [t.index for t in self.running_trials],
        }


# In[18]:


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

