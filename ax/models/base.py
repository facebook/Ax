#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any


class Model:
    """Base class for an Ax model.

    Note: the core methods each model has: `fit`, `predict`, `gen`,
    `cross_validate`, and `best_point` are not present in this base class,
    because the signatures for those methods vary based on the type of the model.
    This class only contains the methods that all models have in common and for
    which they all share the signature.
    """

    @classmethod
    def serialize_state(cls, raw_state: dict[str, Any]) -> dict[str, Any]:
        """Serialized output of `self._get_state` to a JSON-ready dict.
        This may involve storing part of state in files / external storage and
        saving handles for that storage in the resulting serialized state.
        """
        return raw_state

    @classmethod
    def deserialize_state(cls, serialized_state: dict[str, Any]) -> dict[str, Any]:
        """Restores model's state from its serialized form, to the format it
        expects to receive as kwargs.
        """
        return serialized_state

    def _get_state(self) -> dict[str, Any]:
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

        NOTE: In most cases, `state` is passed into the model's initialization as
        kwargs, so keys in the state dict should correspond to model's kwargs.
        """
        return {}

    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def feature_importances(self) -> Any:
        raise NotImplementedError(
            "Feature importance not available for this Model type"
        )

    def __repr__(self) -> str:
        return self.__class__.__name__
