# How to compile this report on Overleaf

1. **Zip this entire `report/` folder** (it contains `report.tex` + `figures/`).
2. On Overleaf: **New Project → Upload Project → Upload Zip**.
3. In the Overleaf project menu, set **Compiler = pdfLaTeX** (default) and **TeX Live = 2023+**.
4. Press **Recompile**. The PDF is two-column A4, ~5 pages.

## Files

| File                            | Purpose                              |
|---------------------------------|--------------------------------------|
| `report.tex`                    | Single-file LaTeX source             |
| `figures/fig_architecture.png`  | (matplotlib backup) architecture     |
| `figures/fig_demo_montage.png`  | Environment/demo snapshots from MP4  |
| `figures/fig_reward_curves.png` | Training reward curves               |
| `figures/fig_curriculum_progression.png` | Curriculum advancement     |
| `figures/fig_comparison_bars.png` | Algorithm + ablation comparison    |
| `figures/fig_generalization.png` | Per-scenario bars                   |
| `figures/fig_generalization_heatmap.png` | OOD success heatmap         |

The TikZ architecture inside `report.tex` is generated natively by LaTeX and
is sharper than the PNG fallback at any zoom. The report is two-column, but
large figures/tables that previously overflowed are now `figure*`/`table*`
with `\\resizebox`, so they fit cleanly in Overleaf.

## Editing tips (inside Overleaf)

- **Author/roll-number block**: edit the `\author{...}` block near the top.
- **Length control**: the document targets 4–6 pages on A4 two-column
  with `top=18mm, bottom=20mm, left=15mm, right=15mm, columnsep=6mm`.
  Adjust geometry in the preamble if your venue requires different
  margins.
- **All metrics** come from `report_artifacts/all_runs.csv` and
  `report_artifacts/generalization.csv`. If you re-evaluate, re-run
  `scripts/aggregate_all.py` and `scripts/figures/make_figures.py`,
  then copy the regenerated PNGs into `figures/`.
