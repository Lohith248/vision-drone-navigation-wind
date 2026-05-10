# All Runs — Evaluation Summary

_n_episodes per run = 30_

| run | algorithm | seed | total_timesteps | params | success_rate | collision_rate | timeout_rate | mean_reward | std_reward | mean_length |
|---|---|---|---|---|---|---|---|---|---|---|
| best_vit/ppo_seed0_t1p5m | ppo | 0 | 1507328 | 1002311 | 1.0 | 0.0 | 0.0 | 124.847 | 0.274 | 150.6 |
| baseline_ddpg/ddpg_seed0 | ddpg | 0 | 700000 | 141572 | 0.0 | 1.0 | 0.0 | -53.396 | 3.362 | 19.1 |
| baseline_sac/sac_seed0 | sac | 0 | 300000 | 213256 | 0.0 | 1.0 | 0.0 | -42.164 | 1.437 | 12.3 |
| multi_seed/ppo_seed1 | ppo | 1 | 802816 | 1002311 | 1.0 | 0.0 | 0.0 | 124.329 | 0.23 | 279.6 |
| abl_no_advnorm/ppo_seed0 | ppo | 42 | 507904 | 1002311 | 1.0 | 0.0 | 0.0 | 124.192 | 0.138 | 223.7 |
| abl_low_reward_scale/ppo_seed0 | ppo | 42 | 507904 | 1002311 | 1.0 | 0.0 | 0.0 | 28.899 | 0.052 | 1023.1 |
| abl_no_curriculum/ppo_seed0 | ppo | 42 | 507904 | 1002311 | 1.0 | 0.0 | 0.0 | 124.449 | 0.277 | 226.8 |
| abl_state_only/ppo_seed0 | ppo | 42 | 606208 | 70919 | 0.867 | 0.133 | 0.0 | 105.242 | 49.055 | 314.4 |
| abl_cnn_encoder/ppo_seed0 | ppo | 42 | 606208 | 1028295 | 0.0 | 1.0 | 0.0 | 19.768 | 0.309 | 126.2 |
| abl_domain_random/ppo_seed0 | ppo | 0 | 704512 | 1002311 | 1.0 | 0.0 | 0.0 | 124.574 | 0.276 | 205.1 |
