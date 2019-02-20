#!/usr/bin/env python3

import importlib
import sys

from ae import (
    benchmark,
    core,
    exceptions,
    generator,
    metrics,
    models,
    plot,
    runners,
    storage,
    tests,
    utils,
)


sys.modules["ae.lazarus.ae.benchmark"] = benchmark
sys.modules["ae.lazarus.ae.core"] = core
sys.modules["ae.lazarus.ae.exceptions"] = exceptions
sys.modules["ae.lazarus.ae.generator"] = generator
sys.modules["ae.lazarus.ae.metrics"] = metrics
sys.modules["ae.lazarus.ae.models"] = models
sys.modules["ae.lazarus.ae.plot"] = plot
sys.modules["ae.lazarus.ae.runners"] = runners
sys.modules["ae.lazarus.ae.storage"] = storage
sys.modules["ae.lazarus.ae.tests"] = tests
sys.modules["ae.lazarus.ae.utils"] = utils

# need to import after all other modules are aliased, since it uses aliased
# paths on import
api = importlib.import_module("ae.api")
sys.modules["ae.lazarus.ae.api"] = api
