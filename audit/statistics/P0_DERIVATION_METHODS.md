# V15 P0 statistical derivation methods

## Scope

This directory is derived only from the frozen V14 formal-results archive. No
training, inference, metric recomputation, sample replacement, or confirmatory
redesign was performed.

## Confirmatory reporting completion

The frozen 18 identity-level comparisons retain their original aggregation,
two-sided exact sign tests, four pre-specified Holm families, and 10,000-draw
identity bootstrap intervals. For each comparison, the derivation adds the
first quartile, third quartile, interquartile range, positive/zero/negative
identity counts, descriptive relative effect when defined, and the 8-bit RGB
equivalent for MAE. The 8-bit conversion is the normalized MAE difference
multiplied by 255; it is not a new metric or test.

All inferences condition on the five fixed trained models. Comparisons with
FreeUV additionally condition on the one frozen FreeUV generation. The
identity bootstrap does not resample training or generation randomness.

## FaceScape hidden-support sensitivity

For FaceScape, hidden-region support thresholds are fixed at 1, 5, 10, 20,
and 50 texels. At each threshold, pairs with support below the threshold are
excluded before applying the unchanged aggregation: pair median within each
fixed seed, then median across the five seeds, then median across identities.
Positive effects indicate lower MAE for FrugalFace3D-Lite. The three fixed
comparisons are NoCond, B-lite-FT, and FreeUV-conserved.

The sensitivity analysis is exploratory. It reports coverage, identity-level
effects, quartiles, unadjusted 95% identity bootstrap intervals, sign counts,
relative effects, and 8-bit equivalents. It generates no p-value and does not
recalculate Holm corrections. Threshold 1 is required to reproduce the frozen
formal FaceScape results exactly.
