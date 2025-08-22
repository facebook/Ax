# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
This module contains the defaults for optimization settings used in botorch_modular.
These are important parameters for the optimization of the acquisition function.
We try to find values that are rather conservative: working well in most cases
at the expense of speed.

NUM_RESTARTS: The number of points the acq. function is iteratively optimized on.
    The restarts are evaluated in parallel, don't let yourself be fooled by the name.
    The higher the number, the more likely it is that the global optimum is found.

RAW_SAMPLES: The number of initial sobol-sampled points evaluated. The best
    NUM_RESTARTS of which are iteratively optimized. The higher the number,
    the likelier the global optimum is found, but the slower the optimization.
    NUM_RESTARTS should be larger than RAW_SAMPLES.

INIT_BATCH_LIMIT: The maximum number of initial points to evaluate in parallel.
    The higher the number, the faster the optimization, but memory tends
    to grow roughly linear with INIT_BATCH_LIMIT.

BATCH_LIMIT: The maximum number of points to evaluate in parallel. The higher
    the number, the faster the optimization, but memory tends to grow
    roughly linear with BATCH_LIMIT.

**Summary**

The effects of the above constants on our standard use cases, are roughly that
performance improves with NUM_RESTARTS and RAW_SAMPLES, and time/memory behave as:

Time: FORWARD_CONST * ACQ_TIME(INIT_BATCH_LIMIT) * (RAW_SAMPLES / INIT_BATCH_LIMIT)
    + OPT_CONST * ACQ_TIME(BATCH_LIMIT) * (NUM_RESTARTS / BATCH_LIMIT)

Memory: FORWARD_MEM_CONST * INIT_BATCH_LIMIT + OPT_MEM_CONST * BATCH_LIMIT

We use ACQ_TIME for the time the acquisition function takes to evaluate the given
number of points. ACQ_TIME is typically sub-linear, as parallel processing can be used.
We use *_CONST for constants that depend on your hardware, application and
acquisition function.


MAX_OPT_AGG_SIZE: The maximum BATCH_LIMIT when the optimizer does not support batching.
    This is currently only the case, if you use complicated constraints or have other
    special optimizer arguments.
    See `botorch.generation.gen.get_reasons_against_fast_path`.

    In this case, the optimization is done on the sum of the acquisition function
    outputs of the batch. This interacts in a complicated way with performance and time:
    Higher values can slow things down, as they lead to more optimiziation steps
    with the problem getting harder, but higher values can also speed things as we
    batch evaluations of the acquisition function.
    MAX_OPT_AGG_SIZE is ignored if BATCH_LIMIT <= MAX_OPT_AGG_SIZE.
"""

NUM_RESTARTS = 20
RAW_SAMPLES = 1024
INIT_BATCH_LIMIT = 32
BATCH_LIMIT = 20
MAX_OPT_AGG_SIZE = 5
