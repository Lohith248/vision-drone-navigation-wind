#!/usr/bin/env python3
"""
Legacy hook kept so pipelines calling ``fill_report.py`` keep working.

The Markdown narrative ``FINAL_REPORT.md`` was removed to avoid duplicating
``report_artifacts/report/report.tex`` (single authoritative submission PDF).

After new evaluations, regenerate CSV summaries then splice LaTeX:

  python scripts/aggregate_all.py --episodes 200 --device cuda
  python scripts/wilson_ci.py --in report_artifacts/all_runs.csv \\
      --episodes 200 --out report_artifacts/all_runs_ci.csv
  python scripts/figures/make_figures.py
  python scripts/fill_report_tex.py
"""
from __future__ import annotations


def main() -> None:
    print(
        "fill_report.py: skipped (FINAL_REPORT.md retired — "
        "use report.tex + scripts/fill_report_tex.py)."
    )


if __name__ == "__main__":
    main()
