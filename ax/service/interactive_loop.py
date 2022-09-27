# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from logging import Logger
from queue import Queue
from threading import Event, Lock, Thread
from typing import Callable, Tuple

from ax.core.types import TEvaluationOutcome, TParameterization

from ax.exceptions.core import DataRequiredError
from ax.exceptions.generation_strategy import MaxParallelismReachedException
from ax.service.ax_client import AxClient
from ax.utils.common.logger import get_logger

logger: Logger = get_logger(__name__)

IDLE_SLEEP_SEC = 0.1


# TODO[mpolson64] Create `optimize` method that constructs its own ax_client
def optimize_with_client(
    ax_client: AxClient,
    num_trials: int,
    candidate_queue_maxsize: int,
    elicitation_function: Callable[[TParameterization], TEvaluationOutcome],
) -> None:
    """
    Function to facilitate running Ax experiments with candidate pregeneration (the
    generation of candidate points while waiting for trial evaluation). This can be
    useful in many contexts, especially in interactive experiments in which trial
    evaluation entails eliciting a response from a human. Candidate pregeneration
    uses the time waiting for trail evaluation to generate new candidates from the
    data available. Note that this is a tradeoff -- a larger pregeneration queue
    will result in more trials being generated with less data compared to a smaller
    pregeneration queue (or no pregeneration as in conventional Ax usage) and should
    only be used in contexts where it is necessary for the user to not experience any
    "lag" while candidates are being generated.

    Extract results of the experiment from the AxClient passed in.

    The basic structure is as follows: One thread tries for a lock on the AxClient,
    generates a candidate, and enqueues it to a candidate queue. Another thread tries
    for the same lock, takes all the trial outcomes in the outcome queue, and attaches
    them to the AxClient. The main thread pops a candidate off the candidate queue,
    elicits response from the user, and puts the response onto the outcome queue.

    Args:
        ax_client: An AxClient properly configured for the experiment intended to be
            run. Construct and configure this before passing in.
        num_trials: The total number of trials to be run.
        candidate_queue_maxsize: The maximum number of candidates to pregenerate.
        elicitation_function: Function from parameterization (as returned by
        `AxClient.get_next_trial`) to outcome (as expected by
        `AxClient.complete_trial`).
    """

    # Construct a lock to ensure only one thread my access the AxClient at any moment
    ax_client_lock = Lock()

    # Construct queues to buffer inputs and outputs of the AxClient
    candidate_queue: "Queue[Tuple[TParameterization, int]]" = Queue(
        maxsize=candidate_queue_maxsize
    )
    data_queue: "Queue[Tuple[int, TEvaluationOutcome]]" = Queue()

    # Construct events to allow us to stop the generator and attacher threads
    candidate_generator_stop_event = Event()
    data_attacher_stop_event = Event()

    # Construct threads to run candidate thread-safe pregeneration and thread-safe
    # data attaching respectively
    candidate_generator_thread = Thread(
        target=_candidate_generator,
        args=(
            ax_client,
            ax_client_lock,
            candidate_generator_stop_event,
            candidate_queue,
            num_trials,
        ),
    )
    data_attacher_thread = Thread(
        target=_data_attacher,
        args=(ax_client, ax_client_lock, data_attacher_stop_event, data_queue),
    )

    candidate_generator_thread.start()
    data_attacher_thread.start()

    for _i in range(num_trials):
        parametrization, trial_index = candidate_queue.get()

        raw_data = elicitation_function(parametrization)

        data_queue.put((trial_index, raw_data))
        candidate_queue.task_done()

    # Clean up threads (if they have not been stopped already)
    candidate_generator_stop_event.set()
    data_queue.join()
    data_attacher_stop_event.set()


def _candidate_generator(
    ax_client: AxClient,
    lock: Lock,
    stop_event: Event,
    queue: "Queue[Tuple[TParameterization, int]]",
    num_trials: int,
) -> None:
    """Thread-safe method for generating the next trial from the AxClient and
    enqueueing it to the candidate queue. The number of candidates pre-generated is
    controlled by the maximum size of the queue. Generation stops when num_trials
    trials are attached to the AxClient's experiment.
    """
    while not stop_event.is_set():
        if not queue.full():
            with lock:
                try:
                    parameterization_with_trial_index = ax_client.get_next_trial()
                    queue.put(parameterization_with_trial_index)

                    # Check if candidate generation can end
                    if len(ax_client.experiment.arms_by_name) >= num_trials:
                        stop_event.set()

                except (MaxParallelismReachedException, DataRequiredError) as e:
                    logger.warning(
                        f"Encountered error {e}, sleeping for {IDLE_SLEEP_SEC} "
                        "seconds and trying again."
                    )
                    pass  # Try again later

        time.sleep(IDLE_SLEEP_SEC)


def _data_attacher(
    ax_client: AxClient,
    lock: Lock,
    stop_event: Event,
    queue: "Queue[Tuple[int, TEvaluationOutcome]]",
) -> None:
    """Thread-safe method for attaching evaluation outcomes to the AxClient from the
    outcome queue. If the AxClient's lock is acquired all data in the outcome queue
    is attached at once, then the lock released. Stops when the event is set.
    """

    while not stop_event.is_set():
        if not queue.empty():
            with lock:
                while not queue.empty():
                    trial_index, raw_data = queue.get()

                    ax_client.complete_trial(
                        trial_index=trial_index,
                        raw_data=raw_data,
                    )

                    queue.task_done()

        time.sleep(IDLE_SLEEP_SEC)
