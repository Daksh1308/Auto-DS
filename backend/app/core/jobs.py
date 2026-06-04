"""In-memory job store for cleaned datasets.

Phase 1 keeps everything in process memory. A future phase will move this to a
persistent backend (database or object storage).
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from threading import Lock

import pandas as pd


@dataclass
class Job:
    job_id: str
    df: pd.DataFrame
    original_filename: str


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()

    def create(self, df: pd.DataFrame, original_filename: str) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = Job(
                job_id=job_id, df=df, original_filename=original_filename
            )
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def serialize_csv(self, job: Job) -> bytes:
        buf = io.StringIO()
        job.df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

    def serialize_xlsx(self, job: Job) -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            job.df.to_excel(writer, index=False, sheet_name="cleaned")
        return buf.getvalue()


job_store = JobStore()
