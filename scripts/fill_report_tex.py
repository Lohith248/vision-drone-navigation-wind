#!/usr/bin/env python3
"""
Splice aggregated evaluation results (with Wilson 95% CIs) into the LaTeX
report. The report contains marker comment lines of the form

    % AUTO-ABL:abl_no_time_penalty/ppo_seed0
    No time penalty & ... & ... & 1.000 & 0.000 & 124.95\\

and

    % AUTO-RES:abl_no_time_penalty/ppo_seed0
    PPO no time-penalty       & ppo  & 42 &   507,904 & 1,002,311 & --- [..., ...] & --- [..., ...] & ... & ---\\

For each marker, this script:
  - finds the next non-blank, non-comment line (the data row)
  - replaces the trailing numeric fields with values pulled from
    report_artifacts/all_runs.csv
  - injects Wilson 95% CIs computed from the success / collision counts
    using report_artifacts/all_runs_ci.csv (already merged numbers in [%]).

Inputs:
  report_artifacts/all_runs.csv
  report_artifacts/all_runs_ci.csv  (output of scripts/wilson_ci.py)

Output:
  report_artifacts/report/report.tex (updated in place, .bak written first)
"""
from __future__ import annotations

import csv
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "report_artifacts/report/report.tex"
ALL_CSV = ROOT / "report_artifacts/all_runs.csv"
CI_CSV = ROOT / "report_artifacts/all_runs_ci.csv"


def _read_runs() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if ALL_CSV.exists():
        with open(ALL_CSV) as f:
            for r in csv.DictReader(f):
                out[r["run"]] = dict(r)
    if CI_CSV.exists():
        with open(CI_CSV) as f:
            for r in csv.DictReader(f):
                if r["run"] in out:
                    out[r["run"]].update(r)
                else:
                    out[r["run"]] = dict(r)
    return out


def _f(r: dict, key: str, default="---"):
    v = r.get(key, "")
    if v in ("", None):
        return default
    try:
        return float(v)
    except ValueError:
        return v


def _fmt_pct_ci(r: dict, base: str) -> str:
    lo = r.get(f"{base}_lo")
    hi = r.get(f"{base}_hi")
    if lo and hi:
        try:
            lo_f, hi_f = float(lo), float(hi)
            return f"[{100 * lo_f:.1f}, {100 * hi_f:.1f}]"
        except ValueError:
            pass
    return "[---, ---]"


def _fmt_rate(r: dict, base: str) -> str:
    p = _f(r, base, default=None)
    if p is None or p == "---":
        return "---"
    return f"{float(p):.3f}"


def _fmt(value, fmt: str = "{:.2f}") -> str:
    if value in (None, "---", ""):
        return "---"
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def update_tex() -> None:
    if not TEX.exists():
        sys.exit(f"missing {TEX}")
    runs = _read_runs()
    if not runs:
        sys.exit("no run rows found - did you execute aggregate_all.py?")

    text = TEX.read_text()
    backup = TEX.with_suffix(".tex.bak")
    backup.write_text(text)
    lines = text.splitlines()

    abl_re = re.compile(r"^\s*%\s*AUTO-ABL:(\S+)")
    res_re = re.compile(r"^\s*%\s*AUTO-RES:(\S+)")

    # We re-emit the data row under each marker.
    out_lines: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = abl_re.match(line)
        if m:
            run_key = m.group(1)
            r = runs.get(run_key)
            out_lines.append(line)
            i += 1
            # find the next non-blank non-comment non-marker data line
            while i < n and (not lines[i].strip()
                             or lines[i].lstrip().startswith("%")):
                out_lines.append(lines[i])
                i += 1
            if i >= n:
                continue
            data_line = lines[i]
            if r is not None:
                # Replace the trailing "X & Y & Z\\" (success, coll, reward).
                # The data row has NO leading & because the row continued from
                # earlier lines via column separators.
                succ = _fmt_rate(r, "success_rate")
                coll = _fmt_rate(r, "collision_rate")
                rew = _fmt(_f(r, "mean_reward", default=None), "{:.2f}")
                # Use \g<0> to avoid backref issues; escape backslashes
                # (re.sub treats `\\` in repl as a literal `\`).
                replacement = f"{succ} & {coll} & {rew}\\\\\\\\"
                new_data = re.sub(
                    r"[0-9.\-]+\s*&\s*[0-9.\-]+\s*&\s*-?[0-9.]+\\\\\s*$",
                    replacement,
                    data_line,
                )
                out_lines.append(new_data)
            else:
                out_lines.append(data_line)
            i += 1
            continue

        m = res_re.match(line)
        if m:
            run_key = m.group(1)
            r = runs.get(run_key)
            out_lines.append(line)
            i += 1
            while i < n and (not lines[i].strip()
                             or lines[i].lstrip().startswith("%")):
                out_lines.append(lines[i])
                i += 1
            if i >= n:
                continue
            data_line = lines[i]
            if r is not None:
                # Replace last 4 columns: success_ci, coll_ci, reward, len.
                succ = _fmt_rate(r, "success_rate")
                coll = _fmt_rate(r, "collision_rate")
                succ_ci = _fmt_pct_ci(r, "success_rate")
                coll_ci = _fmt_pct_ci(r, "collision_rate")
                rew = _fmt(_f(r, "mean_reward", default=None), "{:.2f}")
                ep_len = _fmt(_f(r, "mean_length", default=None), "{:.1f}")
                # Escape backslashes for re.sub: 4 in source -> 2 in output -> "\\".
                replacement = (
                    f"& {succ} {succ_ci} & {coll} {coll_ci} & {rew} & {ep_len}\\\\\\\\"
                )
                new_data = re.sub(
                    r"&\s*[^&]*&\s*[^&]*&\s*[^&]*&\s*[^&\\]*\\\\\s*$",
                    replacement,
                    data_line,
                )
                out_lines.append(new_data)
            else:
                out_lines.append(data_line)
            i += 1
            continue

        out_lines.append(line)
        i += 1

    TEX.write_text("\n".join(out_lines) + "\n")
    print(f"\u2713 Updated {TEX}")
    print(f"  backup: {backup}")


if __name__ == "__main__":
    update_tex()
