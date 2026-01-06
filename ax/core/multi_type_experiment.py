#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.experiment import Experiment


class MultiTypeExperiment(Experiment):
    """
    Deprecated.

    Functionality has been upstreamed to Experiment; please use Experiment directly
    instead of MultiTypeExperiment.

    Class remains instantiable for backwards compatibility.
    """

    pass
