#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect

from logging import Logger
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Set

from ax.core import Trial
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.runner import Runner
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none

logger: Logger = get_logger(__name__)


try:
    from torchx.runner import get_runner, Runner as torchx_Runner
    from torchx.specs import AppDef, AppState, AppStatus, CfgVal

    TORCHX_APP_HANDLE: str = "torchx_app_handle"
    TORCHX_RUNNER: str = "torchx_runner"
    TORCHX_TRACKER_BASE: str = "torchx_tracker_base"

    # Maps TorchX AppState to Ax's TrialStatus.
    APP_STATE_TO_TRIAL_STATUS: Dict[AppState, TrialStatus] = {
        AppState.UNSUBMITTED: TrialStatus.CANDIDATE,
        AppState.SUBMITTED: TrialStatus.STAGED,
        AppState.PENDING: TrialStatus.STAGED,
        AppState.RUNNING: TrialStatus.RUNNING,
        AppState.SUCCEEDED: TrialStatus.COMPLETED,
        AppState.CANCELLED: TrialStatus.ABANDONED,
        AppState.FAILED: TrialStatus.FAILED,
        AppState.UNKNOWN: TrialStatus.FAILED,
    }

    class TorchXRunner(Runner):
        """
        An implementation of ``ax.core.runner.Runner`` that delegates job submission to
        the TorchX Runner. This runner is coupled with the TorchX component since Ax
        runners run trials of a single component with different parameters.

        It is expected that the experiment parameter names and types match EXACTLY with
        component's function args. Component function args that are NOT part of the
        search space can be passed as ``component_const_params``. The following args
        are passed automatically if declared in the component function's signature:
            *   ``trial_idx (int)``: current trial's index
            *   ``tracker_base (str)``: torchx tracker's base (typically a URL
                indicating the base dir of the tracker)

        Example:

        .. code-block:: python

        def trainer_component(
            x1: int,
            x2: float,
            trial_idx: int,
            tracker_base: str,
            x3: float,
            x4: str) -> spec.AppDef:
            # ... implementation omitted for brevity ...
            pass

        The experiment should be set up as:

        .. code-block:: python

        parameters=[
        {
            "name": "x1",
            "value_type": "int",
            # ... other options...
        },
        {
            "name": "x2",
            "value_type": "float",
            # ... other options...
        }
        ]

        And the rest of the arguments can be set as:

        .. code-block:: python

        TorchXRunner(
            tracker_base="s3://foo/bar",
            component=trainer_component,
            # trial_idx and tracker_base args passed automatically
            # if the function signature declares those args
            component_const_params={"x3": 1.2, "x4": "barbaz"})

        Running the experiment as set up above results in each trial running:

        .. code-block:: python

        appdef = trainer_component(
                    x1=trial.params["x1"],
                    x2=trial.params["x2"],
                    trial_idx=trial.index,
                    tracker_base="s3://foo/bar",
                    x3=1.2,
                    x4="barbaz")

        torchx.runner.get_runner().run(appdef, ...)

        """

        def __init__(
            self,
            tracker_base: str,
            component: Callable[..., AppDef],
            component_const_params: Optional[Dict[str, Any]] = None,
            scheduler: str = "local",
            cfg: Optional[Mapping[str, CfgVal]] = None,
        ) -> None:
            self._component: Callable[..., AppDef] = component
            self._scheduler: str = scheduler
            self._cfg: Optional[Mapping[str, CfgVal]] = cfg
            # need to use the same runner in case it has state
            # e.g. torchx's local_scheduler has state hence need to poll status
            # on the same scheduler instance
            self._torchx_runner: torchx_Runner = get_runner()
            self._tracker_base = tracker_base
            self._component_const_params: Dict[str, Any] = component_const_params or {}

        def run(self, trial: BaseTrial) -> Dict[str, Any]:
            """
            Submits the trial (which maps to an AppDef) as a job
            onto the scheduler using ``torchx.runner``.

            ..  note:: only supports `Trial` (not `BatchTrial`).
            """

            if not isinstance(trial, Trial):
                raise ValueError(
                    f"{type(trial)} is not supported. Check your experiment setup"
                )

            parameters = dict(self._component_const_params)
            parameters.update(not_none(trial.arm).parameters)
            component_args = inspect.getfullargspec(self._component).args
            if "trial_idx" in component_args:
                parameters["trial_idx"] = trial.index

            if "experiment_name" in component_args:
                parameters["experiment_name"] = trial.experiment.name

            if "tracker_base" in component_args:
                parameters["tracker_base"] = self._tracker_base

            appdef = self._component(**parameters)
            app_handle = self._torchx_runner.run(appdef, self._scheduler, self._cfg)
            return {
                TORCHX_APP_HANDLE: app_handle,
                TORCHX_RUNNER: self._torchx_runner,
                TORCHX_TRACKER_BASE: self._tracker_base,
            }

        def poll_trial_status(
            self, trials: Iterable[BaseTrial]
        ) -> Dict[TrialStatus, Set[int]]:
            trial_statuses: Dict[TrialStatus, Set[int]] = {}

            for trial in trials:
                app_handle: str = trial.run_metadata[TORCHX_APP_HANDLE]
                torchx_runner = trial.run_metadata[TORCHX_RUNNER]
                app_status: AppStatus = torchx_runner.status(app_handle)
                trial_status = APP_STATE_TO_TRIAL_STATUS[app_status.state]

                indices = trial_statuses.setdefault(trial_status, set())
                indices.add(trial.index)

            return trial_statuses

        def stop(
            self, trial: BaseTrial, reason: Optional[str] = None
        ) -> Dict[str, Any]:
            """Kill the given trial."""
            app_handle: str = trial.run_metadata[TORCHX_APP_HANDLE]
            self._torchx_runner.stop(app_handle)
            return {"reason": reason} if reason else {}

except ImportError:
    logger.warning(
        "torchx package not found. If you would like to use TorchXRunner, please "
        "install torchx."
    )
    pass
