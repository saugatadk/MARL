### Comparative Traffic Signal Control Benchmarking: IOE Pashchimanchal Campus
*Evaluated across 5 random seeds ([42, 43, 44, 45, 46]) with simulation horizon = 3600s.*

| Metric | Fixed-Time | Actuated | DQN | PPO | Improvement (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Avg Waiting Time (s)** | 1.61 ± 0.17 | 0.23 ± 0.03 | 0.23 ± 0.03 | 0.40 ± 0.09 | **+85.6%** (DQN) |
| **Avg Queue Length (veh)** | 0.17 ± 0.02 | 0.07 ± 0.01 | 0.07 ± 0.01 | 0.08 ± 0.01 | **+61.5%** (DQN) |
| **Total Throughput (veh)** | 665.20 ± 0.40 | 665.40 ± 0.49 | 665.40 ± 0.49 | 665.40 ± 0.49 | **+0.0%** (DQN) |
