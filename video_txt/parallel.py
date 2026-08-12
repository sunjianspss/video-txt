"""Running independent jobs on a small pool of threads.

Everything this tool parallelizes waits on something else: an API answering, a
speech engine writing a file, ffprobe reading one. A handful of threads is the
whole of it — there is no CPU work here worth a process pool.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor


def map_in_parallel[T, R](jobs: Sequence[T], work: Callable[[T], R], *, workers: int) -> list[R]:
    """Run `work` over every job, in order, and hand back what each one returned.

    A job that raises takes the batch down with it and the queued jobs are
    dropped rather than started: they would spend API quota, synthesis time and
    the caller's patience on a run that has already failed.
    """
    limit = max(1, min(workers, len(jobs)))
    if limit == 1:
        return [work(job) for job in jobs]

    with ThreadPoolExecutor(max_workers=limit) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for queued in futures:
                queued.cancel()
            raise
