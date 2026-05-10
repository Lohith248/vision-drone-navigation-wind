#!/usr/bin/env python3
"""
Add Wilson 95% confidence intervals for success / collision / timeout rates to
an aggregated evaluation CSV.

Wilson score interval (no continuity correction):
    p_hat ± z * sqrt(p_hat (1-p_hat) / n + z^2/(4n^2)) / (1 + z^2/n)
adjusted for the standard Wilson form below.

Usage:
  python scripts/wilson_ci.py \
      --in  report_artifacts/all_runs.csv \
      --episodes 100 \
      --out report_artifacts/all_runs_ci.csv \
      --md-out report_artifacts/all_runs_ci.md

You can also pass --episodes-col to read n per row from a column instead of a flag.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

Z_95 = 1.959964  # two-sided 95% normal quantile


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Returns (lo, hi) Wilson 95% CI for k successes out of n trials."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


RATE_COLS = ("success_rate", "collision_rate", "timeout_rate", "oob_rate")


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}"


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{fmt_pct(lo)}, {fmt_pct(hi)}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--episodes", type=int, required=True,
                    help="Number of episodes per run (constant n)")
    ap.add_argument("--episodes-col", default=None,
                    help="If set, read n per row from this CSV column instead of --episodes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args()

    inp = Path(args.inp)
    rows: list[dict[str, str]] = []
    with open(inp) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for r in reader:
            rows.append(dict(r))

    new_cols: list[str] = []
    for col in RATE_COLS:
        if col in fieldnames:
            new_cols.extend([f"{col}_lo", f"{col}_hi", f"{col}_ci_str"])

    out_fields = list(fieldnames) + [c for c in new_cols if c not in fieldnames]

    for r in rows:
        if args.episodes_col and args.episodes_col in r and r[args.episodes_col].strip():
            try:
                n = int(float(r[args.episodes_col]))
            except ValueError:
                n = args.episodes
        else:
            n = args.episodes
        for col in RATE_COLS:
            if col not in r or not r[col].strip():
                continue
            try:
                p = float(r[col])
            except ValueError:
                continue
            k = int(round(p * n))
            lo, hi = wilson_interval(k, n)
            r[f"{col}_lo"] = f"{lo:.4f}"
            r[f"{col}_hi"] = f"{hi:.4f}"
            r[f"{col}_ci_str"] = fmt_ci(lo, hi)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\u2713 Wrote {out}")

    if args.md_out:
        md_out = Path(args.md_out)
        md_cols = [c for c in (
            "run", "algorithm", "seed", "total_timesteps", "params",
            "success_rate", "success_rate_ci_str",
            "collision_rate", "collision_rate_ci_str",
            "timeout_rate", "mean_reward", "mean_length",
        ) if c in out_fields]
        with open(md_out, "w") as f:
            f.write("# All Runs \u2014 Evaluation Summary (with Wilson 95% CIs)\n\n")
            f.write(f"_n_episodes per run = {args.episodes}_\n\n")
            f.write("| " + " | ".join(md_cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(md_cols)) + "|\n")
            for r in rows:
                f.write("| " + " | ".join(str(r.get(c, "")) for c in md_cols) + " |\n")
        print(f"\u2713 Wrote {md_out}")


if __name__ == "__main__":
    main()
