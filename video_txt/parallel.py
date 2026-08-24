"""Running independent jobs on a small pool of threads.

Everything this tool parallelizes waits on something else: an API answering, a
speech engine writing a file, ffprobe reading one. A handful of threads is the
whole of it — there is no CPU work here worth a process pool.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait


def map_in_parallel[T, R](jobs: Sequence[T], work: Callable[[T], R], *, workers: int) -> list[R]:
    """Run `work` over every job, in order, and hand back what each one returned.

    A job that raises takes the batch down with it and the queued jobs are
    dropped rather than started: they would spend API quota, synthesis time and
    the caller's patience on a run that has already failed.
    """
    limit = max(1, min(workers, len(jobs)))
    if limit == 1:
        return [work(job) for job in jobs]

    pool = ThreadPoolExecutor(max_workers=limit)
    futures = []
    try:
        futures = [pool.submit(work, job) for job in jobs]
        done, _pending = wait(futures, return_when=FIRST_EXCEPTION)
        failed = next(
            (
                future
                for future in done
                if not future.cancelled() and future.exception() is not None
            ),
            None,
        )
        if failed is not None:
            failed.result()
        results = [future.result() for future in futures]
    except BaseException:
        for queued in futures:
            queued.cancel()
        # Running network calls cannot be killed safely, but they must not keep
        # this function from surfacing a sibling failure immediately.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
        return results
