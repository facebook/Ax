# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.methods.sobol import get_sobol_benchmark_method


@dataclass
class BenchmarkMethodRegistryEntry:
    factory_fn: Callable[..., BenchmarkMethod]
    factory_kwargs: Mapping[str, Any]


BENCHMARK_METHOD_REGISTRY = {
    "Sobol": BenchmarkMethodRegistryEntry(
        factory_fn=get_sobol_benchmark_method,
        factory_kwargs={"distribute_replications": False},
    )
}


def get_benchmark_method(
    method_key: str,
    registry: Mapping[str, BenchmarkMethodRegistryEntry] | None = None,
    **additional_kwargs: Any,
) -> BenchmarkMethod:
    """
    Generate a benchmark method from a key, registry, and additional arguments.

    Args:
        method_key: The key by which a `BenchmarkMethodRegistryEntry` is
            looked up in the registry; a method will then be generated from
            that entry and `additional_kwargs`.
        registry: If not provided, uses `BENCHMARK_METHOD_REGISTRY` to use
            problems defined within Ax.
        additional_kwargs: Additional kwargs to pass to the factory function of
            the `BenchmarkMethodRegistryEntry`.
    """
    registry = BENCHMARK_METHOD_REGISTRY if registry is None else registry
    entry = registry[method_key]
    kwargs = copy.deepcopy(dict(entry.factory_kwargs))
    kwargs.update(additional_kwargs)
    return entry.factory_fn(**kwargs)
