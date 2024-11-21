# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import Any, Mapping

from ax.core.base_trial import TrialStatus
from ax.preview.api.protocols.utils import _APIRunner
from ax.preview.api.types import TParameterization
from pyre_extensions import override


class IRunner(_APIRunner):
    @override
    def run_trial(
        self, trial_index: int, parameterization: TParameterization
    ) -> dict[str, Any]:
        """
        Given an index and parameterization, run a trial and return a dictionary of any
        appropriate metadata. This metadata will be used to identify the trial when
        polling its status, stopping, fetching data, etc. This may hold information
        such as the trial's unique identifier on the system its running on, a
        directory where the trial is logging results to, etc.

        The metadata MUST be JSON-serializable (i.e. dict, list, str, int, float, bool,
        or None) so that Trials may be properly serialized in Ax.
        """
        ...

    @override
    def poll_trial(
        self, trial_index: int, trial_metadata: Mapping[str, Any]
    ) -> TrialStatus:
        """
        Given trial index and metadata, poll the status of the trial.
        """
        ...

    @override
    def stop_trial(
        self, trial_index: int, trial_metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        """
        Given trial index and metadata, stop the trial. Returns a dictionary of any
        appropriate metadata.

        The metadata MUST be JSON-serializable (i.e. dict, list, str, int, float, bool,
        or None) so that Trials may be properly serialized in Ax.
        """
        ...
