# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.storage.sqa_store.db import session_scope
from ax.storage.sqa_store.sqa_classes import SQAExperiment


def delete_experiment(exp_name: str) -> None:
    """Delete experiment by name.

    Args:
        experiment_name: Name of the experiment to delete.
    """
    with session_scope() as session:
        exp = session.query(SQAExperiment).filter_by(name=exp_name).one_or_none()
        session.delete(exp)
        session.flush()
