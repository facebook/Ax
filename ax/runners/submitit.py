#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterable, Optional, Set

import pandas as pd

from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data

from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.core.runner import Runner
from ax.core.trial import Trial
from ax.utils.common.result import Err, Ok

from submitit import (
    AutoExecutor,
    DebugExecutor,
    DebugJob,
    LocalExecutor,
    LocalJob,
    SlurmJob,
)
from submitit.core.core import Executor


class SubmitItRunner(Runner):
    """
    SubmitIt runner for Ax. SubmitIt is a 
    """

    def __init__(self, train_evaluate_fn, executor: Optional[Executor] = None) -> None:
        self.train_evaluate_fn = train_evaluate_fn
        self.executor = executor or AutoExecutor()
        self.jobs = {}

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        job = self.executor.submit(self.train_evaluate_fn, trial)
        self.jobs[trial.index] = job

        # store trial metadata needed to reconstitute it
        # in the metric
        if isinstance(self.executor, AutoExecutor):
            metadata = {
                "args": {
                    "job_id": job.job_id,
                    "folder": job._paths._folder,
                    "tasks": job._tasks,
                },
                "jobtype": "slurm",
            }

        elif isinstance(self.executor, LocalExecutor):
            ############
            ##### this fails to pickle, doesn't work
            ############
            metadata = {
                "args": {
                    "job_id": job.job_id,
                    "folder": job._paths._folder,
                    "tasks": (job._paths.task_id,),
                    "process": job._process,
                },
                "jobtype": "local",
            }

        elif isinstance(self.executor, DebugExecutor):
            metadata = {
                "args": {"folder": job._paths._folder, "submission": job.submission()},
                "jobtype": "debug",
            }

            # debug jobs don't ever return unless we force them, I think?
            _ = job.result()

        else:
            raise NotImplementedError

        return metadata

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
        # map SLURM trial status to Ax
        # https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES
        trial_status_map = {
            # the obvious ones
            "COMPLETED": TrialStatus.COMPLETED,
            "FAILED": TrialStatus.FAILED,
            "PENDING": TrialStatus.STAGED,
            "RUNNING": TrialStatus.RUNNING,
            "QUEUED": TrialStatus.STAGED,
            "DONE": TrialStatus.COMPLETED,  # this happens with local executor, not slurm
            # less clear how to map these, maybe some should be failed
            # but all should be transient?
            "UNKNOWN": TrialStatus.STAGED,
            "PREEMPTED": TrialStatus.RUNNING,
            "COMPLETING": TrialStatus.RUNNING,  # for Ax's purposes if it's not done, it's not done
            "SUSPENDED": TrialStatus.RUNNING,  # if the cluster suspended our job we count it as still running for Ax?
            "STOPPED": TrialStatus.RUNNING,
        }
        trial_status = {t: [] for t in set(trial_status_map.values())}
        for t in trials:
            # get the job
            j = self.jobs[t.index]
            # get status
            submitit_job_state = j.state
            ax_job_state = trial_status_map[submitit_job_state]
            trial_status[ax_job_state].append(t.index)

        return trial_status
    
    def stop(
        self, trial: BaseTrial, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Need to make sure to kill jobs when the scheduler exits."""
        del reason
        self.jobs[trial.index].stop()


class SubmitItMetricFetcher(Metric):
    """
    SubmitIt metric fetcher for ax. TODO fetch intermediate metrics from
    Tensorboard or similar instead.
    """
    def fetch_trial_data(self, trial: BaseTrial) -> MetricFetchResult:
        """Obtains data via fetching it from ` Ifor a given trial."""
        if not isinstance(trial, Trial):
            raise ValueError("This metric only handles `Trial`.")

        try:
            # Here we leverage the "job_id" metadata
            if trial.run_metadata["jobtype"] == "slurm":
                job = SlurmJob(**trial.run_metadata["args"])
            elif trial.run_metadata["jobtype"] == "local":
                job = LocalJob(**trial.run_metadata["args"])
            elif trial.run_metadata["jobtype"] == "debug":
                job = DebugJob(**trial.run_metadata["args"])
            else:
                raise NotImplementedError
            fval = job.result()
            df_dict = {
                "trial_index": trial.index,
                "metric_name": self.name,
                "arm_name": trial.arm.name,
                "mean": fval,
                # Can be set to 0.0 if function is known to be noiseless
                # or to an actual value when SEM is known. Setting SEM to
                # `None` results in Ax assuming unknown noise and inferring
                # noise level from data.
                "sem": None,
            }
            return Ok(value=Data(df=pd.DataFrame.from_records([df_dict])))
        except Exception as e:
            return Err(
                MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )
