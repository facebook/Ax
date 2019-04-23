#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-strict

from fractions import Fraction
from math import ceil, log
from typing import List, NamedTuple, Optional


class BracketShape(NamedTuple):
    """
    The shape of a batch launched by hyper band.

    If this is the first batch of the bracket the arms are generated via
    `get_random_hyperparameter_configuration`. Otherwise we take the `num_arms`
    best arms from the previous batch in the bracket.
    """

    scale: Fraction
    num_configs: int


def scale_schedule(brs: List[BracketShape], desired: int) -> List[BracketShape]:
    """
    Scales a hyperband schedule to have the desired number of initial
    arms.
    """
    if desired == 0:
        return []
    num_arms = sum(br.num_configs for br in brs)
    if desired == num_arms:
        return brs
    scale_factor = desired / num_arms
    bracket = brs[0]
    # We round up the scale for this bracket because we favor allocating
    # more arms towards the heavily downscaled brackets.
    new_bracket = BracketShape(bracket.scale, ceil(bracket.num_configs * scale_factor))
    return [new_bracket] + scale_schedule(brs[1:], desired - new_bracket.num_configs)


class HyperbandSchedule:
    """
    A naive implementation of hyperband.

    This does not actually run the hyperband algorithm but instead returns the
    sizes and downsampling of all the batches that would be scheduled.


    The inner loop of hyperband performs successive halving (start with `n` runs
    for `r` iterations and then select the `n/eta` best runs and run those for
    `r * eta` iterations and repeat that process until `r` reaches `max_iter`).
    Each run of the inner loop is called a Bracket

    The outer loop launches [0...s_max] Brackets starting at exponentially
    bigger number of iterations (`r`) and adjust the number of initial
    configurations (`n`) to make sure we allocated the same budget (`B`) to each
    bracket.

    :see: https://people.eecs.berkeley.edu/~kjamieson/hyperband.html

    :param max_iter: maximum number of iterations/epoch per configuration
    :param eta: downsampling rate
    """

    eta: float
    brackets: List[BracketShape]

    def __init__(
        self,
        max_iter: int = 81,
        eta: float = 3.0,
        num_brackets: Optional[int] = None,
        budget: Optional[int] = None,
    ) -> None:
        self.eta = eta
        # number of unique executions of Successive Halving - 1
        s_max = (
            int(log(max_iter) / log(eta)) if num_brackets is None else num_brackets - 1
        )
        # total number of iterations (without reuse) per execution of Successive
        # Halving (n,r)
        B = (s_max + 1) * max_iter if budget is None else budget / (s_max + 1)
        self.brackets = [
            BracketShape(
                # initial number of iterations to run configurations for
                Fraction(int(max_iter * eta ** (-s)), max_iter),
                # initial number of configurations to get as close to the budget B as
                # possible (this is an approximation that assumes we don't round the
                # num_arms we use to an int)
                round(B / max_iter / (s + 1) * eta ** s),
            )
            for s in reversed(range(s_max + 1))
        ]

    def scale(self, num_configs: int) -> None:
        """Resize all the brackets in that schedule to make us use ``num_configs``
        """
        self.brackets = scale_schedule(self.brackets, num_configs)

    @property
    def num_configs(self) -> int:
        """The number of base configs used by all the brackets"""
        return sum(br.num_configs for br in self.brackets)
