from __future__ import annotations

import threading
import time

import pytest

from video_txt.parallel import map_in_parallel


def test_results_come_back_in_the_order_the_jobs_were_given():
    """Callers zip the results against their own inputs, so order is the contract."""
    delays = [0.03, 0.0, 0.015]

    def work(index: int) -> int:
        time.sleep(delays[index])
        return index

    assert map_in_parallel([0, 1, 2], work, workers=3) == [0, 1, 2]


@pytest.mark.parametrize("workers", [1, 4])
def test_a_failed_job_takes_the_batch_down_with_it(workers):
    def refuse(_job: int) -> None:
        raise RuntimeError("no quota for this")

    with pytest.raises(RuntimeError, match="no quota for this"):
        map_in_parallel([1, 2, 3], refuse, workers=workers)


def test_the_jobs_still_queued_behind_a_failure_are_dropped():
    """A batch that has already failed should not spend more quota or synthesis time."""
    started: list[int] = []
    lock = threading.Lock()

    def work(job: int) -> None:
        with lock:
            started.append(job)
        if job == 0:
            raise RuntimeError("the API turned us away")
        time.sleep(0.05)

    with pytest.raises(RuntimeError):
        map_in_parallel(list(range(12)), work, workers=2)

    assert len(started) < 12


def test_one_worker_needs_no_pool():
    assert map_in_parallel([1, 2], lambda job: job * 2, workers=1) == [2, 4]
    assert map_in_parallel([], lambda job: job, workers=8) == []
