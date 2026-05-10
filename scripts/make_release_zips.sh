#!/usr/bin/env bash
# Build archives at repo root:
#   project_github_upload.zip  — code + report artifacts + scripts (no runs/, no venv)
#   project_full_download.zip  — same + runs/ (checkpoints + metrics; still no venv)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Refreshing Overleaf / submission bundles ..."
(
  cd report_artifacts
  rm -f overleaf_report_bundle.zip submission_bundle.zip
  rm -rf _bundle_stage && mkdir -p _bundle_stage/figures
  cp report/report.tex report/README_OVERLEAF.md _bundle_stage/
  cp report/figures/*.png _bundle_stage/figures/
  (cd _bundle_stage && zip -qr ../overleaf_report_bundle.zip .)
  rm -rf _bundle_stage
  zip -qr submission_bundle.zip \
    overleaf_report_bundle.zip \
    demo_ppo_wind.mp4 demo_ppo_wind.gif \
    all_runs.csv all_runs_ci.csv all_runs.md all_runs_ci.md \
    generalization.csv generalization_ci.csv generalization_ci.md
)

rm -f project_github_upload.zip project_full_download.zip

echo "Building project_github_upload.zip ..."
zip -qr project_github_upload.zip \
  drone_rl \
  drone_nav_env \
  scripts \
  requirements.txt \
  README.md \
  project_proposal.md \
  .gitignore \
  report_artifacts/report \
  report_artifacts/*.csv \
  report_artifacts/*.md \
  report_artifacts/*.mp4 \
  report_artifacts/*.gif \
  report_artifacts/overleaf_report_bundle.zip \
  report_artifacts/submission_bundle.zip \
  -x "**/__pycache__/*" "**/*.pyc"

echo "Building project_full_download.zip (includes runs/, ~800MB+) ..."
zip -qr project_full_download.zip \
  drone_rl \
  drone_nav_env \
  scripts \
  requirements.txt \
  README.md \
  project_proposal.md \
  .gitignore \
  report_artifacts/report \
  report_artifacts/*.csv \
  report_artifacts/*.md \
  report_artifacts/*.mp4 \
  report_artifacts/*.gif \
  report_artifacts/overleaf_report_bundle.zip \
  report_artifacts/submission_bundle.zip \
  runs \
  -x "**/__pycache__/*" "**/*.pyc"

ls -lh project_github_upload.zip project_full_download.zip \
   report_artifacts/overleaf_report_bundle.zip report_artifacts/submission_bundle.zip
