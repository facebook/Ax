# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

EXPERIMENT_DESIGN_KEY: str = "experiment_design"

# Sentinel used to serialize ``None`` dict keys in JSON, which does not
# support non-string keys.
_NULL_TRIAL_TYPE_KEY: str = "null"


@dataclass
class AutomationSettings:
    """Per-trial-type settings that govern how the Orchestrator automates
    experiment execution for a given trial type.

    Multi-type experiments specify one ``AutomationSettings`` per trial type;
    single-type experiments use a single entry keyed by the experiment's
    default trial type (typically ``None``).

    NOTE: For now all durations are expressed in seconds for convenience;
    units may change later.

    Args:
        concurrency_limit: Maximum number of arms to run concurrently for
            this trial type. ``None`` means unlimited.
        generation_lookahead: Number of candidate arms to pre-generate even
            when concurrency is reached, so users can choose whether to
            deploy them instead of existing ones. ``None`` means no
            lookahead.
        budget: Maximum total number of arms to run for this trial type
            across the entire experiment. ``None`` means unlimited.
        stage_after_seconds: Seconds to wait before automatically staging
            a trial after it is created. ``None`` means do not auto-stage.
            0 means auto-stage without waiting.
        run_after_seconds: Seconds to wait before automatically running a
            staged or candidate (if staging is not required) trial.
            ``None`` means do not auto-run. 0 means auto-run
            without waiting.
    """

    concurrency_limit: int | None = None
    generation_lookahead: int | None = None
    # NOTE: In the future, we may want a more complex notion for an overarching
    # budget in the experiment, across multiple trial types. When we get there,
    # we will likely want to hold that in `ExperimentDesign` and validate that
    # this `budget` is `None` when that is specified.
    budget: int | None = None
    stage_after_seconds: int | None = None
    run_after_seconds: int | None = None


@dataclass
class ExperimentDesign:
    """Holds experiment-level execution configuration.

    Experiment-level settings (frequencies) live directly on this class,
    while per-trial-type settings are stored in ``automation_settings``,
    a dictionary mapping trial type names to ``AutomationSettings``
    instances.

    During prototyping, this is serialized into ``experiment.properties``
    via storage encoders. First-class storage support will follow.

    NOTE: in ax/storage/sqa_store/encoder.py, attributes of this class
    are automatically serialized and stored in experiment.properties.

    Args:
        automation_settings: Mapping from trial type to per-trial-type
            automation configuration.
        analysis_frequency_seconds: How often (in seconds) to poll trial
            statuses, fetch data, and run automated analysis across all
            trial types. ``None`` means no automated analysis.
        generation_frequency_seconds: How often (in seconds) to trigger
            automated candidate generation across all trial types.
            ``None`` means no automated candidate generation.
    """

    automation_settings: dict[str | None, AutomationSettings] = field(
        default_factory=dict,
    )
    analysis_frequency_seconds: int | None = None
    generation_frequency_seconds: int | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for storage in
        ``experiment.properties``.
        """
        return {
            "automation_settings": {
                (_NULL_TRIAL_TYPE_KEY if k is None else k): dataclasses.asdict(v)
                for k, v in self.automation_settings.items()
            },
            "analysis_frequency_seconds": self.analysis_frequency_seconds,
            "generation_frequency_seconds": self.generation_frequency_seconds,
        }

    @classmethod
    def from_json(cls, json_dict: dict[str, Any]) -> ExperimentDesign:
        """Deserialize from a JSON dict stored in ``experiment.properties``."""
        return cls(
            automation_settings={
                (None if k == _NULL_TRIAL_TYPE_KEY else k): AutomationSettings(**v)
                for k, v in json_dict["automation_settings"].items()
            },
            analysis_frequency_seconds=json_dict.get("analysis_frequency_seconds"),
            generation_frequency_seconds=json_dict.get("generation_frequency_seconds"),
        )
