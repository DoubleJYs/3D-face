# Exploratory robustness analyses

The tables in this directory are derived from the same frozen five-seed result archive used by the confirmatory analysis. They do not replace or modify the original 18 comparisons, two-sided exact sign tests, or within-family Holm corrections, and they introduce no additional confirmatory p-values.

`REALY_H_SUPPORT_SENSITIVITY.csv` applies minimum target-visible hidden-region support thresholds of 1, 5, 10, 20, and 50 texels before retaining the original aggregation order: median across view pairs within identity and seed, median across the five fixed seeds within identity, and median across identities. Effects for MAE are comparator minus FrugalFace3D-Lite, so positive values indicate lower FrugalFace3D-Lite error.

`FACESCAPE_SFACE_COVERAGE_SENSITIVITY.csv` evaluates all three planned FaceScape SFace comparisons under three identity-coverage rules. It reports the retained identity and pair counts, median identity effect, unadjusted 95% identity-bootstrap interval, and sign counts. The reduced-coverage analyses remain exploratory because the stricter rules were evaluated after the frozen confirmatory design.

Any additional support-threshold table is admitted to this directory only after its input archive hash, aggregation order, row counts, bootstrap rule, and non-confirmatory status have been independently validated and recorded in `DERIVED_OUTPUT_REGISTRY.json`.

## REALY directed-view analysis

`REALY_DIRECTION_SUPPORT_SUMMARY.csv` describes all 12 non-self directed
source-target combinations formed by V01, V02, V03, and V04. Every direction
contains 100 anonymous identities. The table reports hidden-region and
target-visible support distributions, the H/A ratio, and fixed support-threshold
coverage without filtering directions by the observed method effects.

`REALY_DIRECTION_EFFECT_SUMMARY.csv` contains 12 directions by three
comparisons: FrugalFace3D-Lite relative to NoCond, B-lite same-task
fine-tuning, and FreeUV with observed texture preserved. For MAE, effects are
comparator minus FrugalFace3D-Lite, so positive values indicate lower
FrugalFace3D-Lite error. Each identity effect is the median of the five fixed
seed effects. Direction summaries report the median across 100 identities,
an unadjusted 95% identity-bootstrap interval using 10,000 resamples, and
positive, zero, and negative identity counts.

`REALY_DIRECTION_IDENTITY_EFFECTS.csv` is the public audit table. It contains
anonymous identity tokens, directions, comparisons, five fixed-seed effects,
and identity medians. Pair identifiers and per-pair support fields are removed.
Running `recompute_realy_directional_summary.py` verifies the redaction schema
and reproduces all 36 summary rows. The directional analysis is exploratory;
it introduces no p-values and does not enter the four confirmatory Holm
families.
