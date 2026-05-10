# Vision-Based Continuous Drone Navigation in Wind

RL final project: PPO with a Vision Transformer for continuous-control drone navigation in a PyBullet corridor with moving obstacles, sensor noise, and Ornstein-Uhlenbeck wind.

## Highlights

- Multimodal PPO actor-critic: RGB camera + 15D state vector.
- Vision Transformer image encoder with state MLP fusion.
- Continuous action space: `(vx, vy, vz)`.
- Six-stage curriculum from easy corridors to full wind + clutter.
- Baselines: DDPG and SAC.
- Ablations: no advantage normalization, low reward scale, no curriculum, state-only PPO, CNN encoder, domain randomization.
- Generalization tests: no wind, strong wind, dense clutter, turns, narrow corridor, noisy sensors, long corridor.

## Key Results

- PPO+ViT final-stage evaluation: **100% success**, **0% collision**, mean reward **124.85** over 30 episodes.
- DDPG and SAC state-only baselines collapse under the final windy cluttered stage.
- PPO+ViT generalizes to wind/noise/clutter/turns/narrow corridors; long-corridor extrapolation remains a limitation.

## Main Files

- `drone_rl/` - RL environment, algorithms, networks, trainers, configs.
- `drone_nav_env/` - wind model and navigation environment support code.
- `scripts/` - training queue, evaluation, demo rendering, figure generation.
- `report_artifacts/report/report.tex` - final Overleaf-ready LaTeX report.
- `report_artifacts/report/figures/` - figures used by the report.
- `report_artifacts/demo_ppo_wind.mp4` - demo video.
- `report_artifacts/all_runs.csv` - final run comparison metrics.
- `report_artifacts/generalization.csv` - OOD evaluation metrics.

## Reproduce

```bash
pip install -r requirements.txt

# Aggregate trained runs
python scripts/aggregate_all.py --episodes 30

# Generalization evaluation
python scripts/eval_generalization.py \
  --checkpoint runs/best_vit/ppo_seed0_t1p5m/checkpoints/checkpoint_1507328.pt \
  --config drone_rl/configs/ppo_best.yaml \
  --episodes 30 \
  --out report_artifacts/generalization.csv

# Generate report figures
python scripts/figures/make_figures.py
```

## Links

- GitHub: https://github.com/vishal36-pop/project
- Demo video: https://iiitbac-my.sharepoint.com/:f:/g/personal/vishal_sriram_iiitb_ac_in/IgD556iWK6AsR59jMQwmS-3IARv652K0kij4iknw9YbVtEg?e=tKtRGe

