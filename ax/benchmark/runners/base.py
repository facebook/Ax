# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABC, abstractmethod, abstractproperty
from math import sqrt
from typing import Any, Union

import torch
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.batch_trial import BatchTrial
from ax.core.runner import Runner
from ax.core.trial import Trial

from ax.utils.common.typeutils import checked_cast
from torch import Tensor


class BenchmarkRunner(Runner, ABC):
    """
    A Runner that produces both observed and ground-truth values.

    Observed values equal ground-truth values plus noise, with the noise added
    according to the standard deviations returned by `get_noise_stds()`.

    This runner does require that every benchmark has a ground truth, which
    won't necessarily be true for real-world problems. Such problems fall into
    two categories:
        - If they are deterministic, they can be used with this runner by
          viewing them as noiseless problems where the observed values are the
          ground truth. The observed values will be used for tracking the
          progress of optimization.
        - If they are not deterministc, they are not supported. It is not
          conceptually clear how to benchmark such problems, so we decided to
          not over-engineer for that before such a use case arrives.
    """

    @abstractproperty
    def outcome_names(self) -> list[str]:
        """The names of the outcomes of the problem (in the order of the outcomes)."""
        pass  # pragma: no cover

    def get_Y_true(self, arm: Arm) -> Tensor:
        """
        Return the ground truth values for a given arm.

        Synthetic noise is added as part of the Runner's `run()` method.
        """
        ...

    @abstractmethod
    def get_noise_stds(self) -> Union[None, float, dict[str, float]]:
        """
        Return the standard errors for the synthetic noise to be applied to the
        observed values.
        """
        ...

    def run(self, trial: BaseTrial) -> dict[str, Any]:
        """Run the trial by evaluating its parameterization(s).

        Args:
            trial: The trial to evaluate.

        Returns:
            A dictionary with the following keys:
                - Ys: A dict mapping arm names to lists of corresponding outcomes,
                    where the order of the outcomes is the same as in `outcome_names`.
                - Ystds: A dict mapping arm names to lists of corresponding outcome
                    noise standard deviations (possibly nan if the noise level is
                    unobserved), where the order of the outcomes is the same as in
                    `outcome_names`.
                - Ys_true: A dict mapping arm names to lists of corresponding ground
                    truth outcomes, where the order of the outcomes is the same as
                    in `outcome_names`. If the benchmark problem does not provide a
                    ground truth, this key will not be present in the dict returned
                    by this function.
                - "outcome_names": A list of metric names.
        """
        Ys, Ys_true, Ystds = {}, {}, {}
        noise_stds = self.get_noise_stds()

        if noise_stds is not None:
            # extract arm weights to adjust noise levels accordingly
            if isinstance(trial, BatchTrial):
                # normalize arm weights (we assume that the noise level is defined)
                # w.r.t. to a single arm allocated all of the sample budget
                nlzd_arm_weights = {
                    arm: weight / sum(trial.arm_weights.values())
                    for arm, weight in trial.arm_weights.items()
                }
            else:
                nlzd_arm_weights = {checked_cast(Trial, trial).arm: 1.0}
            # generate a tensor of noise levels that we'll reuse below
            if isinstance(noise_stds, float):
                noise_stds_tsr = torch.full(
                    (len(self.outcome_names),),
                    noise_stds,
                    dtype=torch.double,
                )
            else:
                noise_stds_tsr = torch.tensor(
                    [noise_stds[metric_name] for metric_name in self.outcome_names],
                    dtype=torch.double,
                )

        for arm in trial.arms:
            # Case where we do have a ground truth
            Y_true = self.get_Y_true(arm)
            Ys_true[arm.name] = Y_true.tolist()
            if noise_stds is None:
                # No noise, so just return the true outcome.
                Ystds[arm.name] = [0.0] * len(Y_true)
                Ys[arm.name] = Y_true.tolist()
            else:
                # We can scale the noise std by the inverse of the relative sample
                # budget allocation to each arm. This works b/c (i) we assume that
                # observations per unit sample budget are i.i.d. and (ii) the
                # normalized weights sum to one.
                std = noise_stds_tsr.to(Y_true) / sqrt(nlzd_arm_weights[arm])
                Ystds[arm.name] = std.tolist()
                Ys[arm.name] = (Y_true + std * torch.randn_like(Y_true)).tolist()

        run_metadata = {
            "Ys": Ys,
            "Ystds": Ystds,
            "outcome_names": self.outcome_names,
            "Ys_true": Ys_true,
        }
        return run_metadata
