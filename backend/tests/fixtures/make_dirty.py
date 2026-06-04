"""Generate the dirty CSV/XLSX fixtures used by the test suite.

Run from backend/:
    PYTHONPATH=. python tests/fixtures/make_dirty.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


FIXTURES_DIR = Path(__file__).resolve().parent


def build_dirty_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Customer ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 11],
            "Full Name": [
                "  Alice Smith  ",
                "Bob Jones",
                "carol o'brien",
                "Dave Miller",
                None,
                "Eve Adams",
                "Frank White",
                " Grace Lee",
                "Heidi Black",
                "Ivan Brown",
                "  Alice Smith  ",
                "Judy Green",
            ],
            "SignUp Date": [
                "2024-01-05",
                "2024-02-10",
                "2024-02-21",
                "not-a-date",
                "2024-03-15",
                "2024-04-01",
                None,
                "2024-05-19",
                "2024-06-30",
                "2024-07-04",
                "2024-01-05",
                "2024-08-12",
            ],
            "Age": [25, 30, None, 40, 22, 28, 35, None, 45, 50, 25, 33],
            "City": [
                "NYC",
                "LA",
                "NYC",
                "Chicago",
                "LA",
                "Chicago",
                "NYC",
                "Boston",
                "NYC",
                "LA",
                "NYC",
                "Boston",
            ],
            "Subscribed": [
                "yes",
                "no",
                "Yes",
                "no",
                None,
                "yes",
                "no",
                "yes",
                "No",
                "yes",
                "yes",
                "no",
            ],
            "Mostly Empty": [None, None, None, None, None, None, None, 1.0, None, None, None, None],
        }
    )


def main() -> int:
    df = build_dirty_df()

    csv_path = FIXTURES_DIR / "dirty.csv"
    xlsx_path = FIXTURES_DIR / "dirty.xlsx"

    df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="raw")

    print(f"wrote {csv_path} ({len(df)} rows)")
    print(f"wrote {xlsx_path} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
