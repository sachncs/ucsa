Run-to-run noise (given): 0.0145 bits/token.

| arm | params | tok/s | ppl | bits/byte | gap vs base (bits/token) [95% CI] | verdict |
|---|---|---|---|---|---|---|
| base-seed43 | 19.2M | 9644 | 468.9 | 1.9385 | +0.0178 [+0.0125, +0.0229] | within run-to-run noise |
| base | 19.2M | 9701 | 463.2 | 1.9346 | - | reference |
| no-state | 19.2M | 10337 | 446.0 | 1.9227 | -0.0544 [-0.0578, -0.0510] | better |
| no-window | 19.2M | 9754 | 494.7 | 1.9554 | +0.0951 [+0.0917, +0.0990] | worse |
| read-gate | 19.2M | 9742 | 436.0 | 1.9155 | -0.0871 [-0.0911, -0.0829] | better |
