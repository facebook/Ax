# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractproperty
from typing import Any, Callable, ChainMap, Dict, Optional, Type

from ax.core.metric import Metric
from ax.core.runner import Runner
from ax.storage.json_store.registry import (
    CORE_CLASS_DECODER_REGISTRY,
    CORE_CLASS_ENCODER_REGISTRY,
    CORE_DECODER_REGISTRY,
    CORE_ENCODER_REGISTRY,
)
from ax.storage.metric_registry import register_metrics
from ax.storage.runner_registry import register_runners
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.sqa_config import SQAConfig


class RegistryBundleBase(ABC):
    """An abstraction to help with storing experiments with custom Metrics and Runners.

    Rather than managing registries individually, the RegistryBundle consumes custom
    metrics, runners, and configuration information and is lazily creates the storage
    registries needed for saving and loading.

    Args:
        metric_clss: A dictionary from Metric classes to the int their type should be
            encoded as in the associated SQAMetric. If no None is passed for the int
            a hash will be generated.
        runner_clss: A dictionary from Runner classes to the int their type should be
            encoded as in the associated SQARunner. If no None is passed for the int
            a hash will be generated.
        json_encoder_registry: A dictionary from Types to methods from an instance of
            the type to JSON.
        json_class_encoder_registry: A dictionary from Types to methods from the type's
            class to JSON.
        json_decoder_registry: A dictionary from str class labels to their associated
            Type.
        json_class_decoder_registry: A dictionary from str class labels to an
            associated method for reconstruction.
    """

    def __init__(
        self,
        metric_clss: Dict[Type[Metric], Optional[int]],
        runner_clss: Dict[Type[Runner], Optional[int]],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        json_encoder_registry: Dict[Type, Callable[[Any], Dict[str, Any]]],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        json_class_encoder_registry: Dict[Type, Callable[[Any], Dict[str, Any]]],
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        json_decoder_registry: Dict[str, Type],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        json_class_decoder_registry: Dict[str, Callable[[Dict[str, Any]], Any]],
    ) -> None:
        # pyre-fixme[4]: Attribute must be annotated.
        self._metric_registry, encoder_registry, decoder_registry = register_metrics(
            metric_clss=metric_clss,
            encoder_registry=json_encoder_registry,
            decoder_registry=json_decoder_registry,
        )
        (
            # pyre-fixme[4]: Attribute must be annotated.
            self._runner_registry,
            # pyre-fixme[4]: Attribute must be annotated.
            self._encoder_registry,
            # pyre-fixme[4]: Attribute must be annotated.
            self._decoder_registry,
        ) = register_runners(
            runner_clss=runner_clss,
            encoder_registry=encoder_registry,
            decoder_registry=decoder_registry,
        )

        self._json_class_encoder_registry = json_class_encoder_registry
        self._json_class_decoder_registry = json_class_decoder_registry

    @property
    def metric_registry(self) -> Dict[Type[Metric], int]:
        return self._metric_registry

    @property
    def runner_registry(self) -> Dict[Type[Runner], int]:
        return self._runner_registry

    @property
    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type` to avoid runtime subscripting errors.
    def encoder_registry(self) -> Dict[Type, Callable[[Any], Dict[str, Any]]]:
        return self._encoder_registry

    @property
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type` to avoid runtime subscripting errors.
    def decoder_registry(self) -> Dict[str, Type]:
        return self._decoder_registry

    @property
    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type` to avoid runtime subscripting errors.
    def class_encoder_registry(self) -> Dict[Type, Callable[[Any], Dict[str, Any]]]:
        return self._json_class_encoder_registry

    @property
    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def class_decoder_registry(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        return self._json_class_decoder_registry

    @abstractproperty
    def sqa_config(self) -> SQAConfig:
        pass

    @abstractproperty
    def encoder(self) -> Encoder:
        pass

    @abstractproperty
    def decoder(self) -> Decoder:
        pass

    @classmethod
    def from_registry_bundles(
        cls, *registry_bundles: RegistryBundleBase
    ) -> RegistryBundleBase:
        return cls(
            metric_clss={},
            runner_clss={},
            json_encoder_registry=dict(
                # pyre-ignore[29] `typing._Alias` is not a function.
                ChainMap(*[bundle.encoder_registry for bundle in registry_bundles])
            ),
            json_class_encoder_registry=dict(
                # pyre-ignore[29] `typing._Alias` is not a function.
                ChainMap(
                    *[bundle.class_encoder_registry for bundle in registry_bundles]
                )
            ),
            json_decoder_registry=dict(
                # pyre-ignore[29] `typing._Alias` is not a function.
                ChainMap(*[bundle.decoder_registry for bundle in registry_bundles])
            ),
            json_class_decoder_registry=dict(
                # pyre-ignore[29] `typing._Alias` is not a function.
                ChainMap(
                    *[bundle.class_decoder_registry for bundle in registry_bundles]
                )
            ),
        )


class RegistryBundle(RegistryBundleBase):
    """A concrete implementation of RegistryBundleBase with sensible defaults."""

    def __init__(
        self,
        metric_clss: Dict[Type[Metric], Optional[int]],
        runner_clss: Dict[Type[Runner], Optional[int]],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        json_encoder_registry: Dict[
            Type, Callable[[Any], Dict[str, Any]]
        ] = CORE_ENCODER_REGISTRY,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        json_class_encoder_registry: Dict[
            Type, Callable[[Any], Dict[str, Any]]
        ] = CORE_CLASS_ENCODER_REGISTRY,
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        json_decoder_registry: Dict[str, Type] = CORE_DECODER_REGISTRY,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        json_class_decoder_registry: Dict[
            str, Callable[[Dict[str, Any]], Any]
        ] = CORE_CLASS_DECODER_REGISTRY,
    ) -> None:
        super().__init__(
            metric_clss=metric_clss,
            runner_clss=runner_clss,
            json_encoder_registry=json_encoder_registry,
            json_class_encoder_registry=json_class_encoder_registry,
            json_decoder_registry=json_decoder_registry,
            json_class_decoder_registry=json_class_decoder_registry,
        )
        self._sqa_config = SQAConfig(
            json_encoder_registry={**self.encoder_registry, **CORE_ENCODER_REGISTRY},
            json_decoder_registry={**self.decoder_registry, **CORE_DECODER_REGISTRY},
            metric_registry=self.metric_registry,
            runner_registry=self.runner_registry,
            json_class_encoder_registry=self.class_encoder_registry,
            json_class_decoder_registry=self.class_decoder_registry,
        )

        self._encoder = Encoder(self._sqa_config)
        self._decoder = Decoder(self._sqa_config)

    # TODO[mpolson64] change @property to @cached_property once we deprecate 3.7
    @property
    def sqa_config(self) -> SQAConfig:
        return self._sqa_config

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    @property
    def decoder(self) -> Decoder:
        return self._decoder
