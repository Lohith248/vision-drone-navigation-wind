# All Runs — Evaluation Summary (with Wilson 95% CIs)

_n_episodes per run = 200_

| run | algorithm | seed | total_timesteps | params | success_rate | success_rate_ci_str | collision_rate | collision_rate_ci_str | timeout_rate | mean_reward | mean_length |
|---|---|---|---|---|---|---|---|---|---|---|---|
| best_vit/ppo_seed0_t1p5m | ppo | 0 | 1507328 | 1002311 | 1.0 | [98.1, 100.0] | 0.0 | [0.0, 1.9] | 0.0 | 124.795 | 150.5 |
| baseline_ddpg/ddpg_seed0 | ddpg | 0 | 700000 | 141572 | 0.0 | [0.0, 1.9] | 1.0 | [98.1, 100.0] | 0.0 | -53.758 | 19.7 |
| baseline_sac/sac_seed0 | sac | 0 | 300000 | 213256 | 0.0 | [0.0, 1.9] | 1.0 | [98.1, 100.0] | 0.0 | -41.967 | 12.2 |
| multi_seed/ppo_seed1 | ppo | 1 | 802816 | 1002311 | 1.0 | [98.1, 100.0] | 0.0 | [0.0, 1.9] | 0.0 | 124.36 | 279.2 |
| abl_no_advnorm/ppo_seed0 | ppo | 42 | 507904 | 1002311 | 1.0 | [98.1, 100.0] | 0.0 | [0.0, 1.9] | 0.0 | 124.238 | 224.9 |
| abl_low_reward_scale/ppo_seed0 | ppo | 42 | 507904 | 1002311 | 1.0 | [98.1, 100.0] | 0.0 | [0.0, 1.9] | 0.0 | 28.893 | 1026.0 |
| abl_no_curriculum/ppo_seed0 | ppo | 42 | 507904 | 1002311 | 1.0 | [98.1, 100.0] | 0.0 | [0.0, 1.9] | 0.0 | 124.52 | 227.7 |
| abl_state_only/ppo_seed0 | ppo | 42 | 606208 | 70919 | 0.89 | [83.9, 92.6] | 0.11 | [7.4, 16.1] | 0.0 | 107.984 | 319.8 |
| abl_cnn_encoder/ppo_seed0 | ppo | 42 | 606208 | 1028295 | 0.0 | [0.0, 1.9] | 1.0 | [98.1, 100.0] | 0.0 | 19.739 | 126.0 |
| abl_domain_random/ppo_seed0 | ppo | 0 | 704512 | 1002311 | 1.0 | [98.1, 100.0] | 0.0 | [0.0, 1.9] | 0.0 | 124.548 | 205.6 |
| abl_no_time_penalty/ppo_seed0 | ppo | 42 | 507904 | 1002311 | 1.0 | [98.1, 100.0] | 0.0 | [0.0, 1.9] | 0.0 | 124.749 | 252.1 |
| abl_no_goal_bonus/ppo_seed0 | ppo | 42 | 507904 | 1002311 | 1.0 | [98.1, 100.0] | 0.0 | [0.0, 1.9] | 0.0 | 94.372 | 239.5 |
