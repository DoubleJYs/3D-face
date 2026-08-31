# CanonReg V1 five-seed sensitivity run

This is a source-only, review-triggered sensitivity package. It leaves the
historical V14 source, all Word files, private caches, and historical results
unchanged. It trains one variant only: the Full architecture with the same
89,386 trainable parameters and the same optimizer, loss weights, batches,
512-step budget, and five seeds. Only `Lres` and `LTV` change.

## Frozen scientific change

- The residual field remains `R=(completed-base)*(1-Vs)`. `Lres` uses
  `R^2*Mcanon` and is normalized per sample by
  `3*sum(Mcanon*(1-Vs))`; this also fixes the definition for soft masks.
- Horizontal and vertical TV use only edges whose two endpoints are in
  `Mcanon`, with each axis normalized by its valid edge count times three.
- Because TV edge selection depends only on `Mcanon`, visible-hidden edges
  inside the canonical domain are retained.
- All other Full loss terms and weights are copied exactly from the locked V14
  core.

The complete machine-readable definition is in `contract.json`.

## Deliberate boundaries

- Five seeds are exactly `2026080447` through `2026080451`.
- Every formal unit runs exactly 512 AdamW steps (`lr=5e-4`,
  `weight_decay=1e-4`), giving five units and 2,560 optimizer steps.
- There is no retry loop, resume mode, best-of-n rule, early stopping, or
  checkpoint selection.
- Formal training and raw D1/D2 inference require the historical Python
  3.10.20, PyTorch 2.4.0+cu121, CUDA 12.1, cuDNN 90100, RTX 4090 (compute
  capability 8.9) environment and deterministic CUBLAS configuration.
- Private fit/evaluation caches and the B-lite checkpoint are supplied by
  absolute runtime paths. They are not included here.
- Inference writes raw native and observed-texture-conserved UV tensors only.
  It does not read target-pair rosters or compute paper metrics.

## Required repository and private inputs

The upload must contain the historical repository at the exact source hashes
listed in `contract.json`, plus separately uploaded private assets:

1. the 288-row fit cache;
2. the fixed B-lite checkpoint with SHA-256
   `2e7799205711a3bb0c809e47ea916e75e8e164dc1d4260f0fd97a8f2f1ac5da9`;
3. the 160-source FaceScape/D1 evaluation cache;
4. the 400-source REALY/D2 evaluation cache.

Use these upload-relative roots and frozen file hashes:

| Input root | Manifest SHA-256 | Tensor SHA-256 |
|---|---|---|
| `private/training_fit_cache` | `c407b911230ef749afeca7c0b1a571a67869a92200e5ef8698be3927068610ec` | `98db0b5962f8505c84f15df74a3302e30210df8eb4d5493c5894a971a40917fc` |
| `private/d1_facescape_eval_cache` | `b8ac441086a519295b1deda0afbe397ea7c420d8c827b3e9bc5882fb8dfcd278` | `5fa2e9b576561e0751058bd72758a90d6db583d5fd08aad30642302860fbbeb4` |
| `private/d2_realy_source_cache` | `f61f4ca6868267ca5d3fa5f45bde90c9412554c5bbbff5393f46e048e0e957a7` | `7981a40b74a8e4828a001aef22c37dc0404944594943bd8b27bd79ffb27d1562` |

The runner refuses any cache that does not match both its historical manifest
and tensor hashes. The paths are upload layout recommendations; command-line
paths may differ, but the bytes may not.

No provider data, biometric tensors, checkpoints, email, or permission record
is stored in this directory.

## One local preflight

From this directory, run only the combined preflight. It performs the source
audit, analytic loss tests, and one CPU optimizer/inference smoke in one call:

```bash
python test_local.py --repository-root /absolute/path/to/frugalface3d_lite/code
```

Expected final status: `PASS_CANONREG_LOCAL_PREFLIGHT`. It produces no
scientific result and writes no persistent output.

## Cloud plan and single preflight

Inspect the fixed schedule without opening private data:

```bash
python run.py --repository-root /workspace/frugalface3d_lite/code plan
```

On the target RTX 4090, run the CUDA smoke once and retain its receipt:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python test_smoke.py \
  --repository-root /workspace/frugalface3d_lite/code \
  --device cuda \
  --receipt /workspace/run_receipts/CANONREG_CUDA_SMOKE.json
```

The formal training command validates that receipt before creating its output
root. Do not repeat the smoke after it passes.

## Formal run: one training campaign, two inference passes

Use new, nonexistent output roots:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python run.py \
  --repository-root /workspace/frugalface3d_lite/code \
  train \
  --fit-cache /workspace/upload/private/training_fit_cache \
  --b-lite-checkpoint /workspace/upload/private/checkpoints/selected_b_lite.pt \
  --cuda-smoke-receipt /workspace/run_receipts/CANONREG_CUDA_SMOKE.json \
  --output-root /workspace/results/CANONREG_TRAIN_V1

CUBLAS_WORKSPACE_CONFIG=:4096:8 python run.py \
  --repository-root /workspace/frugalface3d_lite/code \
  infer \
  --dataset facescape \
  --eval-cache /workspace/upload/private/d1_facescape_eval_cache \
  --training-root /workspace/results/CANONREG_TRAIN_V1 \
  --output-root /workspace/results/CANONREG_D1_RAW_V1

CUBLAS_WORKSPACE_CONFIG=:4096:8 python run.py \
  --repository-root /workspace/frugalface3d_lite/code \
  infer \
  --dataset realy \
  --eval-cache /workspace/upload/private/d2_realy_source_cache \
  --training-root /workspace/results/CANONREG_TRAIN_V1 \
  --output-root /workspace/results/CANONREG_D2_RAW_V1
```

Each command refuses to overwrite an existing root. A failed root is retained
as evidence and is never retried automatically.

## Bind the finished raw closure

After the five checkpoints and both raw-inference terminals exist, validate
their hashes and create one combined closure receipt:

```bash
python run.py --repository-root /workspace/frugalface3d_lite/code close \
  --training-root /workspace/results/CANONREG_TRAIN_V1 \
  --d1-root /workspace/results/CANONREG_D1_RAW_V1 \
  --d2-root /workspace/results/CANONREG_D2_RAW_V1 \
  --output-root /workspace/results/CANONREG_RAW_CLOSURE_V1
```

Successful outputs end with:

- `CANONREG_TRAIN_V1/TRAINING_TERMINAL.json`;
- `CANONREG_D1_RAW_V1/INFERENCE_TERMINAL.json`;
- `CANONREG_D2_RAW_V1/INFERENCE_TERMINAL.json`;
- `CANONREG_RAW_CLOSURE_V1/CLOSURE_TERMINAL.json`.

Copy the complete four result roots back without renaming files. Post-processing
and D1/D2 paired statistical comparison occur only after the raw closure is
hash-verified locally.

## Windows note

The formal run targets Linux/CUDA. For a source-only Windows check, use the
same Python commands. In PowerShell, environment variables use
`$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"`; this does not make Windows a formal
result environment.
