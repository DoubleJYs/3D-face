# FaceUV-Eval command matrix

| Purpose | Command | Expected exit |
|---|---|---:|
| Verify exact public release | `python3 -B tools/verify_public_release.py` | 0 |
| Visibility protocol smoke | `python3 -B START_HERE.py smoke` | 0 |
| Source synthetic suite | `python3 -B START_HERE.py test-source` | 0 |
| Recompute anonymous statistics | `python3 -B audit/statistics/recompute_public_statistics.py` | 0 |

These commands use only the bytes present in this public release. Empirical
execution requires separately licensed assets obtained from their providers.
