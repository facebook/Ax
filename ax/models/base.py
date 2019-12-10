#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict


class Model:
    """Base class for an Ax model.

    Note: the core methods each model has: `fit`, `predict`, `gen`,
    `cross_validate`, and `best_point` are not present in this base class,
    because the signatures for those methods vary based on the type of the model.
    This class only contains the methods that all models have in common and for
    which they all share the signature.
    """

    def _get_state(self) -> Dict[str, Any]:
        """Obtain the state of this model, in order to be able to serialize it
        and restore it from the serialized version.

        While most models in Ax aren't stateful, some models, like `SobolGenerator`,
        are. For Sobol, the value of the `init_position` changes throughout the
        generation process as more arms are generated, and restoring the Sobol
        generator with all the same settings as it was initialized with, will not
        result in the same model, because the `init_position` setting changed
        throughout optimization. Stateful settings like that are returned from
        this method, so that a model can be reinstantiated and 'pick up where it
        left off' –– more arms can be generated as if the model just continued
        generation and was never interrupted and serialized.
        """
        return {}  # pragma: no cover

    def feature_importances(self) -> Any:
        raise NotImplementedError(
            "Feature importance not available for this Model type"
        )
