# Vision-Based Continuous Drone Navigation in Wind

RL final project: PPO with a Vision Transformer for continuous-control drone navigation in a PyBullet corridor with moving obstacles, sensor noise, and Ornstein-Uhlenbeck wind.

## Highlights

- Multimodal PPO actor-critic: RGB camera + 15D state vector.
- Vision Transformer image encoder with state MLP fusion.
- Continuous action space: `(vx, vy, vz)`.
- Six-stage curriculum from easy corridors to full wind + clutter.
- Baselines: DDPG and SAC.
- Eight controlled ablations: no advantage normalization, low reward scale, **no time penalty**, **no goal bonus**, no curriculum, state-only PPO, CNN encoder, domain randomization.
- Generalization stress tests (scripted knobs): no wind, strong wind, higher labeled obstacle density (clamped to training saturation in the current builder), `turns` curriculum flag (no true L-turn mesh yet), narrow corridor (width coupled with lighter clutter/wind), noisy sensors, long corridor.

## Key Results

- PPO+ViT at N=200 episodes: **100% success [98.1, 100.0]**, **0% collision [0.0, 1.9]**, mean reward **124.80**.
- Reward-component ablations show that the +30 goal bonus and the -0.002 time penalty are *efficiency* terms, not learning drivers: removing either keeps 100% success but lengthens episodes by 60-70%.
- DDPG and SAC state-only baselines collapse under the final windy cluttered stage (100% collision).
- Scripted generalization at N=100 per scenario: the policy stays robust to strong wind and higher sensor noise; the **15 m** corridor drops to **19%** success (training length **10 m**). Interpret `dense clutter` / `turns` / `narrow corridor` rows with the simulator caveats in `report.tex` (density clipping, incomplete bend geometry, composite perturbations).

## Main Files

- `drone_rl/` - RL environment, algorithms, networks, trainers, configs.
- `drone_nav_env/` - wind model and navigation environment support code.
- `scripts/` - training queue, evaluation, demo rendering, figure generation.
- `report_artifacts/report/report.tex` - final Overleaf-ready LaTeX report.
- `report_artifacts/report/figures/` - PNG figures consumed by `report.tex` (also inside `overleaf_report_bundle.zip`).
- `report_artifacts/demo_ppo_wind.mp4` - demo video.
- `report_artifacts/all_runs.csv` - final run comparison metrics.
- `report_artifacts/generalization.csv` - OOD evaluation metrics.

## Report vs Markdown notes

- **`report_artifacts/report/report.tex`** is the single authoritative submission write-up: compile this on Overleaf to PDF. Keeping **`FINAL_REPORT.md`** / **`PROJECT_EVOLUTION_*.md`** in parallel caused the prose and tables to drift out of sync, so those narrative Markdown exports were removed.
- **`project_proposal.md`** is only the original mid-semester proposal text (what you promised before implementation). Keep it if the course asks for “proposal vs final”; otherwise it is optional and **not** the same as the final LaTeX report.
- **`report_artifacts/all_runs*.md`** and **`generalization*.md`** are tiny machine-written previews derived from CSV summaries—not prose reports.

## Reproduce

```bash
pip install -r requirements.txt

# (Optional) Train the two reward-component ablations from scratch.
python -m drone_rl.train --config drone_rl/configs/ablations/ppo_no_time_penalty.yaml --no-resume
python -m drone_rl.train --config drone_rl/configs/ablations/ppo_no_goal_bonus.yaml  --no-resume

# Aggregate every trained run with N=200 evaluation episodes.
python scripts/aggregate_all.py --episodes 200 --device cuda

# Generalization evaluation at N=100 per scenario.
python scripts/eval_generalization.py \
  --checkpoint runs/best_vit/ppo_seed0_t1p5m/checkpoints/checkpoint_1507328.pt \
  --config drone_rl/configs/ppo_best.yaml \
  --episodes 100 --device cuda \
  --out report_artifacts/generalization.csv

# Wilson 95% confidence intervals for both tables.
python scripts/wilson_ci.py --in report_artifacts/all_runs.csv      --episodes 200 --out report_artifacts/all_runs_ci.csv
python scripts/wilson_ci.py --in report_artifacts/generalization.csv --episodes 100 --out report_artifacts/generalization_ci.csv

# Regenerate figures and splice numbers into the LaTeX report.
python scripts/figures/make_figures.py
python scripts/fill_report_tex.py
```

## Checkpoints (OneDrive / SharePoint — not on GitHub)

Weights are under **`runs/<experiment>/<run_name>/checkpoints/`**, usually:

- `checkpoint_<timesteps>.pt` — latest snapshot (used by `scripts/aggregate_all.py`)
- `best_model.pt` — optional; some workflows still evaluate the latest checkpoint on the final curriculum stage

**Main PPO+ViT policy:** `runs/best_vit/ppo_seed0_t1p5m/checkpoints/checkpoint_1507328.pt` (same folder may also contain `best_model.pt`).

**Baselines / ablations:** same layout, e.g. `runs/baseline_ddpg/ddpg_seed0/checkpoints/`, `runs/abl_no_goal_bonus/ppo_seed0/checkpoints/`, etc.

To back up everything: zip the whole **`runs/`** directory and upload that archive to OneDrive (too large for typical GitHub repos).

## Release archives

Rebuild both root-level zips after code or `runs/` changes:

```bash
bash scripts/make_release_zips.sh
```

- **`project_github_upload.zip`** — source, scripts, report artifacts, demos; **no** `runs/` (GitHub-sized).
- **`project_full_download.zip`** — same **plus** full **`runs/`** (checkpoints + metrics; large).

## Links

- GitHub: https://github.com/Lohith248/vision-drone-navigation-wind
- Demo video: https://iiitbac-my.sharepoint.com/:f:/g/personal/vishal_sriram_iiitb_ac_in/IgD556iWK6AsR59jMQwmS-3IARv652K0kij4iknw9YbVtEg?e=tKtRGe

