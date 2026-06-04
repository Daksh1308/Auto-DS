"""In-memory cache of ML training results, keyed by (job_id, target)."""

from __future__ import annotations

from threading import Lock

from .train import MLReport


class MLResultStore:
    def __init__(self) -> None:
        self._results: dict[tuple[str, str], MLReport] = {}
        self._lock = Lock()

    def get(self, job_id: str, target: str) -> MLReport | None:
        with self._lock:
            return self._results.get((job_id, target))

    def put(self, job_id: str, target: str, report: MLReport) -> None:
        with self._lock:
            self._results[(job_id, target)] = report

    def clear(self, job_id: str | None = None, target: str | None = None) -> None:
        with self._lock:
            if job_id is None and target is None:
                self._results.clear()
                return
            keys = list(self._results.keys())
            for k in keys:
                jid, tgt = k
                if (job_id is None or jid == job_id) and (
                    target is None or tgt == target
                ):
                    self._results.pop(k, None)


ml_result_store = MLResultStore()
