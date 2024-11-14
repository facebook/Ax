# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.trial import Trial


def async_runtime_func_from_pi(trial: Trial) -> int:
    # First 49 digits of pi, not including the decimal
    pi_digits_str = "3141592653589793115997963468544185161590576171875"
    idx = trial.index % len(pi_digits_str)
    return int(pi_digits_str[idx])
