#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# pyre-strict

# The graphs in this note are rendered by http://ditaa.sourceforge.net/
# and were drawn in emacs picture mode
"""
Terminology:
============

Quoting `Ben Recht <http://www.argmin.net/2016/06/23/hyperband/>`_

  The idea behind successive halving is remarkably simple. We first try out N
  hyperparameter settings for some fixed amount of time T. Then, we keep the N/2 best
  performing algorithms and run for time 2T. Repeating this procedure log2(M) times, we
  end up with N/M configurations run for MT time.

This description is a bit oversimplified: successive halving doesn't always keep the
N/2 best performing algorithms (e.g.: you can the N/3 best ones and run them for 3T).
This scaling factor is called **ETA** in our code.

A **rung** is one level of the above procedure (run N configurations
on a fraction (**scale**) of the whole dataset; this is known as downsampling the data).
Each slot in a rung is called a cell. The whole set of rungs required to do succesive
halving is called a **bracket**.

The **budget** of a bracket is a tally of how many runs will be launched at each
**scale** (we can't represent the budget as a static number because we always update
our estimates of how scaling affects the real cpu cost of a run, so it's a moving
target).

..
  Bracket

    /=====================================\
    :+---+---+---+---+---+---+---+---+---+:
    :|   |   |   |   |   |   |   |   |   |:
    :+---+---+---+---+---+---+---+---+---+:
    :                                     :
    :+---+---+---+                        :
    :|   |   |   |                        :
    :+---+---+---+                        :
    :                                     :
    :+---+                                :
    :|   |                                :
    :+---+                                :
    \=====================================

  Rung

    /=====================================\
    :+---+---+---+---+---+---+---+---+---+:
    :|   |   |   |   |   |   |   |   |   |:
    :+---+---+---+---+---+---+---+---+---+:
    \=====================================


  Cell

    /=====\
    :+---+:
    :|   |:
    :+---+:
    \=====/


.. fb:px:: jn7k


Successive halving:
===================

A bracket is composed of several rungs (each rung is represented by a row in the
diagram below), in each rung all the configurations are run at the same level of
downsamping.

..
  +-----+-----+-----+-----+-----+-----+-----+-----+-----+   Run the 9 configurations on
  |     |     |     |     |     |     |     |     |     |   1/9th of the data
  +-----+-----+-----+-----+-----+-----+-----+-----+-----+
                        |
                        | Promote the 3 best configs of rung 1
  +-----+-----+-----+   | and run them on 1/3rd of the data
  |     |     |     |<--+
  +-----+-----+-----+
            |
            | Keep the best config and run it on all the data
  +-----+   |
  |     |<--+
  +-----+


.. fb:px:: jmFm


Inter-rung parallelism
=======================

The algorithm given in the previous section waits for every configuration in a rung to
be evaluated before running anything in the rung below it. If we are going to promote
the 3 best configs out of 9 we just need to keep the 6 worst runs seen so far in the
top rung; in other words if the we sort the content configs in the rung by worst to best
objective we know that any config that lands in one of the 3 last cells can be promoted
immediately.

..
  +-----+-----+-----+-----+-----+-----+   +-----+-----+-----+
  |     |     |     |     |     |     |-+ |cEEE |cEEE |cEEE |
  +-----+-----+-----+-----+-----+-----+ | +-----+-----+-----+
        +-------------------------------+
        |
        V
  +-----+-----+   +-----+
  |     |     |-+ |cEEE |
  +-----+-----+ | +-----+
     +----------+
     |
     V
  +-----+
  |     |
  +-----+


.. fb:px:: jmKB

Bracket-scaling
================

Bracket scaling is useful when you want to evaluate a specific number of configs.
In order to support running succesive halving on an arbitrary number of configs we just
resize all the rungs starting from the bottom one. There are two caveats:

* No rung must end up with a size of 0
* If a rung is exactly the same size as the one below it we should skip it entirely
(selecting the N best runs out of N is trivial)

The diagram shows the previous bracket scaled down to 3 initial configs:

..
  +-----+-----+-----+
  |     |     |     |
  +-----+-----+-----+
            |
  +-----+   |
  :     :   | The middle rung is useless because it is the same
  +-----+   | size as the rung below it.
            |
            |
  +-----+   |
  |     |<--+
  +-----+

.. fb:px:: jmK6

Resizing running brackets
=========================

Resizing non-empty brackets creates one more corner case we should worry about: if a
rung is already full we shouldn't launch anything from any of the rungs above it. In the
example below resizing the bracket shrunk the second rung by 1 (marked with dashed
lines) and made it full; we should restart from the second rung even though the first
rung is still missing a config (in red).

..
  +-----+-----+-----+-----+-----+-----+-----+-----+------+
  :  X  :  X  :  X  :  X  :  X  :  X  :  X  :  X  : cF99 |
  +-----+-----+-----+-----+-----+-----+-----+-----+------+

  ===========================

  +-----+-----+-----+-----+
  |  X  |  X  |  X  |     :
  +-----+-----+-----+-----+

  +-----+-----+
  |     |     |
  +-----+-----+

  +-----+
  |     |
  +-----+

.. fb:px:: jmSl

"""  # noqa W605

import heapq
from fractions import Fraction
from typing import Dict, List, NamedTuple, Optional, Tuple, Union


class _Empty:
    """
    A class used to tell that a value is missing.

    (``Optional`` is not a proper ``maybe` `type, if you have an ``Optional[T]`` and
    ``T`` is nullable there's no way to tell apart the cases where there's no value
    from the cases where the value was ``None``).
    """

    pass


_EMPTY: _Empty = _Empty()


class _RungCell(NamedTuple):
    """An config and objective

    To simplify the typing we only accept "int" as the type for configs; you can use an
    index in a separate list of configs.
    """

    objective: float
    config: int


class Trial(NamedTuple):
    """A run to launch (they are called suggestions in the sweep).
    """

    config: int
    scale: Fraction


class _Rung:
    """
    A rung represents one generation in the sequential successive halving
    algorithm (but provides the proper abstraction to run that algorithm
    asynchronously).
    We keep the cells in a min heap to make it easy to promote cells
    """

    cells: List[_RungCell]
    level: int  # Our index in the Bracket
    pending: int
    launched: int
    size: int  # Number of configs that passed through this rung
    scale: Fraction

    def __init__(
        self, level: int, scale: Fraction, cells: Optional[List[_RungCell]] = None
    ) -> None:
        self.scale = scale
        self.level = level
        self.pending = 0
        self.launched = 0
        self.cells = cells or []
        heapq.heapify(self.cells)
        self.size = len(self.cells)

    def push(self, objective: float, config: int) -> None:
        """Add an config to that rung
        """
        heapq.heappush(self.cells, _RungCell(objective, config))

    def promote(self) -> int:
        """Returns an config to promote to the next level
        """
        return heapq.heappop(self.cells).config


def _compute_active_rungs(rungs: List[_Rung], target_sizes: List[int]) -> List[_Rung]:
    """Compute the list of rungs we are going to do successive halving on
    """
    assert len(rungs) == len(target_sizes)
    # We start successive halving from the upper most full rung
    # If we launched configs under that rung they might never get promoted all the way
    # to the top even if they are the best in every rung.
    start_rung = None
    for rung in reversed(rungs):
        if target_sizes[rung.level] <= rung.size:
            start_rung = rung
            break
    assert start_rung is not None, "The bottom most rung should always be full"

    # These are all the candidate rungs
    window = rungs[start_rung.level :]

    # And now we get rid of all the useless rungs in the bracket
    # A useless rung is a rung that has the exact same size as the one above it and is
    # empty, We will just end up promoting every config that passes through that rung.
    return [
        rung
        for rung, next_rung in zip(window[:-1], window[1:])
        if rung.cells != [] or rung.pending != 0
        # If we have two rungs in a row with the same number of configs we should
        # just skip the lower one
        or target_sizes[next_rung.level] != target_sizes[rung.level]
    ] + [window[-1]]


Budget = Tuple[Tuple[int, Fraction], ...]


def _compute_budget(
    rungs: List[_Rung], active_rungs: List[_Rung], target_sizes: List[int]
) -> Budget:
    """Compute how many runs we need to launch at each downsampling level to run to
    completion.
    """
    active = {r.level for r in active_rungs}

    def _compute_rung_budget(rung: _Rung) -> int:
        rung_budget = rung.launched
        if rung.level in active:
            prospective_budget = target_sizes[rung.level] - rung.size
            rung_budget += max(0, prospective_budget)
        return rung_budget

    rung_budgets = ((_compute_rung_budget(rung), rung.scale) for rung in rungs[1:])
    return tuple(x for x in rung_budgets if x[0] != 0)


class _Shape(NamedTuple):
    """The result of resizing a bracket runner
    """

    target_sizes: List[int]
    active_rungs: List[_Rung]
    budget: Budget


class BracketRunner:
    """Controller for a bracket of successive halving.

    Hyperband runs several instances of a simple successive halving algorithm side by
    side (in *brackets*). This class keeps track of all the internals required to run
    one of those brackets.
    """

    _rungs: List[_Rung]
    _active_rungs: List[_Rung]
    _target_sizes: List[int]
    _running: Dict[int, int]  # {configuration_index -> rung index}
    _budget: Budget
    _next_shape_up_cache: Union[_Empty, Optional[_Shape]]
    _next_shape_down_cache: Union[_Empty, Optional[_Shape]]

    def __init__(self, eta: float, start_scale: Fraction, configs: List[int]) -> None:
        """
        We have a list of increasingly smaller rungs that keep the worst
        performing `rung.width` iterations they've seen so far.

        A config is either:

        * Waiting to be scheduled at the next rung ``num_iters``
        * Stuck in a rung (either because the rung isn't full yet or because it
        was never promoted out of it)
        * Being run...
        """
        self.eta = eta
        self._running = {}
        assert len(configs) > 0

        self._rungs = [
            _Rung(
                level=0,
                scale=Fraction(0),
                cells=[_RungCell(0.0, cfg) for cfg in configs],
            ),
            _Rung(level=1, scale=start_scale),
        ]
        while self._rungs[-1].scale < Fraction(1):
            level = len(self._rungs)
            numerator = min(
                start_scale.denominator,  # Make sure we don't go over 1
                int(start_scale.numerator * self.eta ** (level - 1)),
            )  # r_i
            self._rungs.append(
                _Rung(level=level, scale=Fraction(numerator, start_scale.denominator))
            )
        # All of these guys are set by size.setter we just have them there to keep the
        # typechecker happy
        self._target_sizes = [len(r.cells) for r in self._rungs]
        self._active_rungs = []
        self._budget: Budget = ()
        self._next_shape_down_cache = _EMPTY
        self._next_shape_up_cache = _EMPTY
        self._set_shape(self._compute_shape(len(configs)))

    def _step_size(self, rung: _Rung) -> float:
        """The number of configs that need to be added to the base run for this rung to
        grow by 1.
        """
        return 1 / (self.eta ** min(0, -rung.level + 1))

    @property
    def size(self) -> int:
        """Base number of runs used to compute the current schedule we are using.
        """
        return self._target_sizes[1]

    def _compute_shape(self, size: int) -> _Shape:
        target_sizes = [
            max(round(size / self._step_size(rung)), 1) if rung.level > 0 else rung.size
            for rung in self._rungs
        ]
        active_rungs = _compute_active_rungs(self._rungs, target_sizes)
        budget = _compute_budget(self._rungs, active_rungs, target_sizes)
        return _Shape(
            target_sizes=target_sizes, active_rungs=active_rungs, budget=budget
        )

    @size.setter
    def size(self, n: int) -> None:
        """Sets shape size."""
        self._set_shape(self._compute_shape(n))

    def _set_shape(self, shape: _Shape) -> None:
        self._target_sizes = shape.target_sizes
        self._active_rungs = shape.active_rungs
        self._budget = shape.budget
        self._clear_shape_cache()

    def _clear_shape_cache(self) -> None:
        self._next_shape_down_cache = _EMPTY
        self._next_shape_up_cache = _EMPTY

    @property
    def num_configs(self) -> int:
        """The number of configs this was initialized with.

        This is also the max size that we can set this bracket to.
        """
        return self._target_sizes[0]

    def _from_to_rungs(self) -> Optional[Tuple[_Rung, _Rung]]:
        """Which rung we would promote from and to.

        We always return the bottom most available rung.
        """
        for src_rung, dst_rung in zip(self._active_rungs[:-1], self._active_rungs[1:]):
            src_tgt_size = self._target_sizes[src_rung.level]
            dst_tgt_size = self._target_sizes[dst_rung.level]
            if len(src_rung.cells) > src_tgt_size - dst_tgt_size:
                return src_rung, dst_rung
        return None

    def suggest_next(self) -> Optional[Trial]:
        """Returns an index and a num_iter of a run to schedule.

        The ``BracketRunner`` doesn't actually keep a copy of the configs nor does it
        know how to set the downscaling so we just pass a config index an num_iters.

        Return:
          Trial(optional) a config to run
        """
        from_to_rungs = self._from_to_rungs()
        if from_to_rungs is None:
            return
        src_rung, dst_rung = from_to_rungs
        dst_rung.pending += 1
        dst_rung.launched += 1
        # Increment the number of seen configurations for all the rungs between
        # src_rungs and dst_rung
        for r in self._rungs[src_rung.level + 1 : dst_rung.level + 1]:
            r.size += 1
        # If the target rung is now full it becomes the new starting point for
        # active_rungs
        if dst_rung.size >= self._target_sizes[dst_rung.level]:
            self._active_rungs = [
                r for r in self._active_rungs if r.level >= dst_rung.level
            ]
        to_promote = src_rung.promote()
        self._running[to_promote] = dst_rung.level
        self._clear_shape_cache()
        return Trial(config=to_promote, scale=dst_rung.scale)

    def running_config(self, config: int) -> bool:
        """Whether this bracket is running the given config."""
        return config in self._running

    def update(self, config: int, objective: float) -> None:
        """Update the runner after running a config

        Arguments:
           config(int): id of the config that was scheduled
           objective(float): the metric that we are trying to minimize
        """
        level = self._running.pop(config)
        rung = self._rungs[level]
        rung.pending -= 1
        rung.push(objective, config)
        self._clear_shape_cache()

    @property
    def done(self) -> bool:
        """Is this bracket done running?"""
        return self._running == {} and len(self._active_rungs) == 1
