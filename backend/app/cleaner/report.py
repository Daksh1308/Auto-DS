"""Cleaning-report data structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ColumnReport:
    name: str
    detected_type: str
    nulls_before: int
    nulls_after: int
    action: str


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    cols_in: int = 0
    cols_out: int = 0
    duplicates_removed: int = 0
    columns: list[ColumnReport] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    renamed_columns: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "cols_in": self.cols_in,
            "cols_out": self.cols_out,
            "duplicates_removed": self.duplicates_removed,
            "dropped_columns": list(self.dropped_columns),
            "renamed_columns": dict(self.renamed_columns),
            "columns": [
                {
                    "name": c.name,
                    "detected_type": c.detected_type,
                    "nulls_before": c.nulls_before,
                    "nulls_after": c.nulls_after,
                    "action": c.action,
                }
                for c in self.columns
            ],
        }
