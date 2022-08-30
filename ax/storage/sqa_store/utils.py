#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Any, List, Optional

from ax.core.experiment import Experiment
from ax.core.search_space import SearchSpace
from ax.exceptions.storage import SQADecodeError
from ax.utils.common.base import Base, SortableBase


JSON_ATTRS = ["baseline_workflow_inputs"]
# Skip over the following attrs in `copy_db_ids`:
# * _experiment (to prevent infinite loops)
# * most generator run and generation strategy metadata
#   (since no Base objects are nested in there,
#   and we don't have guarantees about the structure of some
#   of that data, so the recursion could fail somewhere)
COPY_DB_IDS_ATTRS_TO_SKIP = {
    "_best_arm_predictions",
    "_bridge_kwargs",
    "_candidate_metadata_by_arm_signature",
    "_curr",
    "_experiment",
    "_gen_metadata",
    "_memo_df",
    "_model_kwargs",
    "_model_predictions",
    "_model_state_after_gen",
    "_model",
    "_seen_trial_indices_by_status",
    "_steps",
    "analysis_scheduler",
}
SKIP_ATTRS_ERROR_SUFFIX = "Consider adding to COPY_DB_IDS_ATTRS_TO_SKIP if appropriate."


def is_foreign_key_field(field: str) -> bool:
    """Return true if field name is a foreign key field, i.e. ends in `_id`."""
    return len(field) > 3 and field[-3:] == "_id"


# pyre-fixme[2]: Parameter annotation cannot be `Any`.
def copy_db_ids(source: Any, target: Any, path: Optional[List[str]] = None) -> None:
    """Takes as input two objects, `source` and `target`, that should be identical,
    except that `source` has _db_ids set and `target` doesn't. Recursively copies the
    _db_ids from `source` to `target`.

    Raise a SQADecodeError when the assumption of equality on `source` and `target`
    is violated, since this method is meant to be used when returning a new
    user-facing object after saving.
    """
    if not path:
        path = []

    error_message_prefix = (
        f"Error encountered while traversing source {path + [str(source)]} and "
        f"target {path + [str(target)]}: "
    )

    if len(path) > 15:
        # This shouldn't happen, but is a precaution against accidentally
        # introducing infinite loops
        raise SQADecodeError(error_message_prefix + "Encountered path of length > 10.")

    if type(source) != type(target):
        if not issubclass(type(target), type(source)):
            if source is None and isinstance(target, SearchSpace):
                warnings.warn(
                    error_message_prefix + "Encountered two objects of different "
                    f"types: {type(source)} and {type(target)}. Continuing in the "
                    "special case that `source is None` and `target` is a "
                    "`SearchSpace`."
                )
            else:
                raise SQADecodeError(
                    error_message_prefix + "Encountered two objects of different "
                    f"types: {type(source)} and {type(target)}."
                    + SKIP_ATTRS_ERROR_SUFFIX
                )

    if isinstance(source, Base):
        for attr, val in source.__dict__.items():
            if attr.endswith("_db_id"):
                # we're at a "leaf" node; copy the db_id and return
                setattr(target, attr, val)
                continue

            # Skip attrs that are doubly private or in COPY_DB_IDS_TO_SKIP.
            if attr.startswith("__") or attr in COPY_DB_IDS_ATTRS_TO_SKIP:
                continue

            # For Json attributes we would like to simply test for equality and not
            # recurse through the Json.
            # TODO: Add json_attrs as an argument and plumb through to `copy_db_ids`.
            if attr in JSON_ATTRS:
                source_json = getattr(source, attr)
                target_json = getattr(target, attr)
                if source_json != target_json:
                    SQADecodeError(
                        error_message_prefix + f"Json attribute {attr} not matching "
                        f"between source: {source_json} and target: {target_json}."
                    )
                continue

            # Arms are referenced twice on an Experiment object; once in
            # experiment.arms_by_name/signature and once in
            # trial.arms_by_name/signature. When copying db_ids, we should
            # ignore the former, since it will "collapse" arms of the same
            # name/signature that appear in more than one trial.
            if isinstance(source, Experiment) and attr in {
                "_arms_by_name",
                "_arms_by_signature",
            }:
                continue

            copy_db_ids(val, getattr(target, attr), path + [attr])

    elif isinstance(source, (list, set)):
        source = list(source)
        target = list(target)

        if len(source) != len(target):
            raise SQADecodeError(
                error_message_prefix + "Encountered lists of different lengths."
            )

        if len(source) == 0:
            return

        if isinstance(source[0], Base) and not isinstance(source[0], SortableBase):
            raise SQADecodeError(
                error_message_prefix + f"Cannot sort instances of {type(source[0])}; "
                "sorting is only defined on instances of SortableBase."
            )

        try:
            source = sorted(source)
            target = sorted(target)
        except TypeError as e:
            raise SQADecodeError(
                error_message_prefix + f"TypeError encountered during sorting: {e}"
            )

        for index, x in enumerate(source):
            copy_db_ids(x, target[index], path + [str(index)])

    elif isinstance(source, dict):
        for k, v in source.items():
            if k not in target:
                raise SQADecodeError(
                    error_message_prefix + "Encountered key only present "
                    f"in source dictionary: {k}."
                )
            copy_db_ids(v, target[k], path + [k])

    else:
        return
