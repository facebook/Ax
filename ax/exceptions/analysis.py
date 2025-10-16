#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.exceptions.core import AxError


class AnalysisNotApplicableStateError(AxError):
    """
    Raised when either the Experiment, GenerationStrategy, or Adapter are not in an
    appropriate state to compute the given Analysis with its specific settings.
    """
