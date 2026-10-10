Run-to-run noise (given): 0.0145 bits/token.

| arm | params | tok/s | ppl | bits/byte | gap vs base (bits/token) [95% CI] | verdict |
|---|---|---|---|---|---|---|
| base-seed43 | 19.2M | 9637 | 444.3 | 1.9215 | +0.0008 [-0.0048, +0.0060] | no significant difference |
| base | 19.2M | 9748 | 444.0 | 1.9213 | - | reference |
| lr-1.2e-3 | 19.2M | 9763 | 363.5 | 1.8582 | -0.2887 [-0.2955, -0.2818] | better |
| lr-2.4e-3 | 19.2M | 9723 | 351.1 | 1.8473 | -0.3389 [-0.3489, -0.3293] | better |
| lr-4.8e-3 | 19.2M | 9806 | 486.1 | 1.9498 | +0.1307 [+0.1233, +0.1377] | worse |
| no-jepa | 19.2M | 9772 | 435.9 | 1.9155 | -0.0266 [-0.0297, -0.0234] | within run-to-run noise |
| no-state | 19.2M | 9948 | 443.8 | 1.9211 | -0.0007 [-0.0042, +0.0027] | no significant difference |
| surprise | 19.2M | 9628 | 440.4 | 1.9187 | -0.0120 [-0.0152, -0.0089] | within run-to-run noise |
| weight-ema | 19.2M | 9661 | 455.0 | 1.9290 | +0.0351 [+0.0316, +0.0386] | worse |
| zfilter | 19.2M | 9702 | 437.3 | 1.9165 | -0.0222 [-0.0294, -0.0147] | within run-to-run noise |
