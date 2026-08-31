# V15 Anonymous Frozen-Results Audit Package

This package supports independent audit of the frozen identity-level statistics, the separately locked REALY directional analysis, and reconstruction of the eight public non-face figures mapped to the V15 manuscript. The original V14 confirmatory plan and its 16 frozen core files remain byte-identical; V15 adds reader-facing mappings and descriptive derivatives without changing the confirmatory design or any frozen result.

## Package contents

- `protocol/STATISTICAL_ANALYSIS_PLAN.md`: immutable prespecified confirmatory plan.
- `protocol/REALY_DIRECTIONAL_EXPLORATORY_ANALYSIS_PLAN.md`: separately locked plan for all 12 directed REALY view pairs; it adds no significance test.
- `provenance/FROZEN_V14_CORE_SHA256.json`: fixed hashes for the 16 immutable V14 core files.
- `statistics/`: anonymous identity effects, the 18 confirmatory comparisons, and the complete-reporting derivative.
- `statistics/recompute_public_statistics.py`: verifies medians, fixed 10,000-resample identity-bootstrap intervals, two-sided exact sign tests, four Holm families, and decision flags.
- `robustness/`: support-threshold and coverage analyses plus the public 12-direction support and effect summaries. The directional identity table contains anonymous identity tokens and no pair identifiers or paths.
- `robustness/recompute_realy_directional_summary.py`: recomputes 3,600 anonymous identity-direction-comparison rows and all 36 direction summaries.
- `figures/source_data/`: public source tables for the eight active non-face figures.
- `figures/scripts/rebuild_public_figures.py`: the only supported figure reconstruction entry point.
- `figures/reference_outputs/`: synchronized reference PNGs for all eight figures; Figures 2 and 4 also include TIFF, PDF, and SVG assets.
- `mappings/OUTPUT_MAP.csv`: V15 figure and table mapping.
- `validation/validate_package.py`: integrity, anonymity, scope, immutable-core, statistics, directional-analysis, and fresh-directory figure checks.
- `validation/build_anonymous_archive.py`: maintenance utility that creates a deterministic delivery archive with normalized ownership and no AppleDouble or extended-attribute metadata.

Internal method identifiers are retained where required by the frozen evidence. Reader-facing names are defined in `mappings/METHOD_NAME_MAP.csv`.

## Verify the statistics

Python 3.10 or newer is sufficient for the two statistical entry points:

```bash
python statistics/recompute_public_statistics.py
python robustness/recompute_realy_directional_summary.py
```

The first command starts from anonymous identity effects and verifies all 18 confirmatory comparisons. The second command verifies 3,600 anonymous directional identity effects and all 36 exploratory direction summaries. Neither command reads images, checkpoints, pair-level formal metrics, or input bindings.

## Rebuild the public figures

Install the packages in `requirements.txt`, then run:

```bash
python figures/scripts/rebuild_public_figures.py
```

The command reconstructs:

1. Figure 1, the FrugalFace3D-Lite architecture;
2. Figure 2, visibility-region decomposition and effect attenuation;
3. Figure 3, paired FreeUV output-form analysis;
4. Figure 4, the three-metric identity-effect forest plot;
5. Figure 5, the quality-resource comparison;
6. Supplementary Figure S1, REALY hidden-region support sensitivity;
7. Supplementary Figure S2, all 12 directed REALY effect matrices; and
8. Supplementary Figure S3, the descriptive multi-metric panorama.

Outputs are written to `figures/rebuilt_public_outputs/`. Figure 6 and Supplementary Figure S4 contain licensed face derivatives and are not included. No face image, model checkpoint, tensor array, or pair-level metric is distributed by this package.

The reader-layer terminology matches the current manuscript: the condition ablation is shown as “无显式条件残差（NoCond）”; the same-task baseline is “B-lite 同任务微调”; and FreeUV is limited to the unified-UV-input, observed-texture-preserving output setting. The multi-metric panorama retains raw descriptive medians and does not construct a composite score or total ranking.

Reference PNG hashes are fixed by the package manifest. Fresh reconstruction is checked for successful generation, active figure count, source bindings, and canvas dimensions within an 8% tolerance. Raster bytes and tight bounding boxes may vary with the Matplotlib backend and installed font versions.

## Validate the package

From the package root, run:

```bash
python validation/validate_package.py
```

The validator checks the package manifest, the 16 immutable V14 core hashes, active V15 manuscript mappings, all 18 confirmatory comparisons, the 3,600/36 directional analysis, and a fresh-directory rebuild of all eight public non-face figures. It rejects absolute local or cloud paths, personal identifiers, credentials, model files, arrays, face images, system metadata, undocumented files, and retired figure mappings.

## Statistical boundary

- Confidence intervals are 95% identity-bootstrap intervals and are not multiplicity adjusted.
- Confirmatory significance uses two-sided exact sign tests with Holm correction within four prespecified families.
- Visibility-region decomposition, support and coverage sensitivity, and the REALY 12-direction analysis are descriptive or exploratory and add no confirmatory p-values.
- Inference is conditional on the five fixed trained models per seeded method and the single frozen FreeUV execution represented by the released effects.
- Favorable, unfavorable, and indeterminate comparisons are retained together.

## Reproducibility boundary

This is an auditable frozen-results and quantitative-figure reconstruction package. It is not an end-to-end training or inference reproduction package. Training data, licensed evaluation images, checkpoints, pair-level formal metrics, input-binding payloads, and complete training and inference code are outside its public scope.
