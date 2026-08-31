# FaceUV-Eval

FaceUV-Eval is the public software and audit companion for a visibility-aware
cross-view evaluation protocol for single-image facial UV completion. The
repository provides author-created evaluation and adaptation code, deterministic
synthetic checks, anonymous statistical inputs, and non-face audit figures.

## Start here

Run these commands from the repository root with Python 3.10 or newer:

```bash
python3 -B tools/verify_public_release.py
python3 -B START_HERE.py smoke
python3 -B START_HERE.py test-source
python3 -B audit/statistics/recompute_public_statistics.py
```

The commands respectively verify the exact public file set, exercise the
visibility protocol on synthetic inputs, run the source-only synthetic suite,
and recompute the anonymous confirmatory statistics.

## Repository contents

- `frugalface3d/` contains the author-created implementation interfaces.
- `reproducibility/` contains matched-control and CanonReg source checks.
- `smoke/` contains a deterministic protocol fixture without face data.
- `audit/` contains anonymous tables, analysis plans, validation code, source
  data for non-face figures, and public provenance hashes.
- `tools/verify_public_release.py` enforces the public manifest and safety
  boundary offline.

## Reproducibility boundary

This repository redistributes no portraits, biometric tensors, datasets,
canonical-mask bytes, pretrained assets, or model checkpoints. Empirical
reproduction therefore requires separately licensed assets obtained from their
official providers. The public tree supports source inspection, synthetic
execution, anonymous-statistics recomputation, and audit of the documented
evaluation interface. See `RELEASE_SCOPE.md` and the access documents for the
exact boundary.

## Licensing

Author-created source and documentation are licensed under Apache-2.0.
Anonymous derived tables and non-face audit figures are licensed under
CC-BY-4.0. Third-party assets are not redistributed and are not covered by
these licenses. Per-file rights appear in `PUBLIC_FILE_MANIFEST.json`.
