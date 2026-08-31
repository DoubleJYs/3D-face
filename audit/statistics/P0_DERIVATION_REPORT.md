# V15 P0 statistical derivation report

Status: **PASS**

The frozen 18 comparisons were reproduced exactly for medians, 95% identity-bootstrap intervals, sign counts, raw two-sided exact-sign p-values, within-family Holm p-values, and descriptive relative effects. Quartiles, IQRs, and MAE 8-bit equivalents were added without changing the confirmatory design.

## FaceScape hidden-support sensitivity

| Threshold | Comparator | Pairs | Identities | Median effect | 95% CI | +/0/- | Relative effect |
|---:|---|---:|---:|---:|---|---|---:|
| 1 | NoCond | 148 | 20 | 0.001504 | [-0.001228, 0.002328] | 13/0/7 | 0.76% |
| 1 | B-lite-FT | 148 | 20 | -0.034625 | [-0.043769, -0.024841] | 1/0/19 | -28.73% |
| 1 | FreeUV-conserved | 148 | 20 | 0.015879 | [0.000313, 0.032479] | 14/0/6 | 10.11% |
| 5 | NoCond | 148 | 20 | 0.001504 | [-0.001228, 0.002328] | 13/0/7 | 0.76% |
| 5 | B-lite-FT | 148 | 20 | -0.034625 | [-0.043769, -0.024841] | 1/0/19 | -28.73% |
| 5 | FreeUV-conserved | 148 | 20 | 0.015879 | [0.000313, 0.032479] | 14/0/6 | 10.11% |
| 10 | NoCond | 140 | 20 | 0.001162 | [-0.001051, 0.002213] | 13/0/7 | 0.71% |
| 10 | B-lite-FT | 140 | 20 | -0.039715 | [-0.043826, -0.027378] | 0/0/20 | -30.36% |
| 10 | FreeUV-conserved | 140 | 20 | 0.014664 | [-0.004405, 0.027388] | 12/0/8 | 7.32% |
| 20 | NoCond | 119 | 20 | 0.000079 | [-0.001382, 0.001974] | 10/0/10 | 0.06% |
| 20 | B-lite-FT | 119 | 20 | -0.040017 | [-0.049298, -0.025716] | 0/0/20 | -36.04% |
| 20 | FreeUV-conserved | 119 | 20 | 0.017376 | [-0.006673, 0.027438] | 12/0/8 | 10.36% |
| 50 | NoCond | 97 | 20 | 0.000347 | [-0.001126, 0.001906] | 11/0/9 | 0.25% |
| 50 | B-lite-FT | 97 | 20 | -0.045052 | [-0.048996, -0.038510] | 0/0/20 | -41.88% |
| 50 | FreeUV-conserved | 97 | 20 | 0.003810 | [-0.007919, 0.020344] | 12/0/8 | 1.82% |

No new significance test was run for the threshold analysis. Its intervals and sign counts are descriptive robustness evidence and do not enter the four frozen Holm families.

## Confirmatory table coverage

- Comparisons: 18
- MAE comparisons with 8-bit conversion: 6
- Comparisons with defined relative effects: 12
- All 18 comparisons explicitly state the fixed-model inference condition.
