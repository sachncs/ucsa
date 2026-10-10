Run-to-run noise (given): 0.0560 bits/token.

| arm | params | tok/s | ppl | bits/byte | gap vs base (bits/token) [95% CI] | verdict |
|---|---|---|---|---|---|---|
| base-seed43 | 20.7M | 9759 | 545.4 | 1.9861 | -0.0792 [-0.0843, -0.0738] | within run-to-run noise |
| base | 20.7M | 9859 | 576.2 | 2.0034 | - | reference |
| no-jepa | 20.7M | 9920 | 569.3 | 1.9996 | -0.0175 [-0.0210, -0.0140] | within run-to-run noise |
| no-state | 20.7M | 10037 | 497.7 | 1.9573 | -0.2112 [-0.2155, -0.2063] | better |
