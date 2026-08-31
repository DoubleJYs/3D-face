# W5B49N V14 matched controls V1 / post-processing V1.1

This is a source-only run package for the two matched controls requested after
the V13/V14 review. It creates no manuscript claim by itself and never writes
to either manuscript directory.

## What is tested

| Method | What changes | What stays matched |
|---|---|---|
| `Condition0` (paper label: `NoStruct`) | All six coordinate-normal channels and the 128-D expression token are set to zero | Full residual architecture, 89,386 trainable parameters, B-lite inputs, masks, data, batching, optimizer, steps, and loss family |
| `B-lite-FT` | All 122,164 B-lite parameters are updated from the frozen B-lite checkpoint | Paired-view training samples, random seeds, optimizer budget, and the task-supervision terms used by Full |
| `Full` | Explicit structure and expression inputs are retained | It is trained with both controls on the same CUDA device, data, batching, optimizer, steps, and seed roster |

`B-lite-FT` has no structure gate. Its gate-only non-collapse regularizer is
therefore an exact differentiable zero. All supervised terms and the residual
and smoothness regularizers remain active. This is a same-task-supervision
control, not a claim that the two architectures have identical internal
regularizers.

## Seed plan and cost

The single confirmatory `cuda_five` plan uses seeds `2026080447` through
`2026080451`. Full, Condition0, and B-lite-FT are all trained in one CUDA
campaign: 15 new units and 7,680 optimizer steps. The three historical Full
checkpoints were trained on MPS, so they are retained only as provenance
references and are never mixed into this CUDA confirmatory comparison.

## Inputs kept outside the source package

- The existing 288-sample anonymous fit cache from
  `w5b49n_mechanism_closure_v1`.
- The existing FaceScape 160-sample and/or REALY 400-source evaluation cache.
- The pinned B-lite checkpoint. Historical MPS Full checkpoints are optional
  provenance references only and are not read by the confirmatory runner.

The cache tensor files remain private derived biometric data and must not be
placed in a public source archive. All paths are supplied at runtime.

## Commands

From the repository root:

```bash
python reproducibility/w5b49n_v14_matched_controls_v1/source_check.py --runtime
python reproducibility/w5b49n_v14_matched_controls_v1/test_synthetic.py

CUBLAS_WORKSPACE_CONFIG=:4096:8 python \
  reproducibility/w5b49n_v14_matched_controls_v1/test_synthetic.py \
  --device cuda \
  --receipt /path/to/new_CUDA_SMOKE_RECEIPT.json

python reproducibility/w5b49n_v14_matched_controls_v1/run.py plan --seed-plan cuda_five

CUBLAS_WORKSPACE_CONFIG=:4096:8 python \
  reproducibility/w5b49n_v14_matched_controls_v1/run.py train \
  --seed-plan cuda_five \
  --device cuda \
  --fit-cache /path/to/private_fit_cache \
  --cuda-smoke-receipt /path/to/new_CUDA_SMOKE_RECEIPT.json \
  --output-root /path/to/new_matched_training_root

CUBLAS_WORKSPACE_CONFIG=:4096:8 python \
  reproducibility/w5b49n_v14_matched_controls_v1/run.py infer \
  --device cuda \
  --dataset facescape \
  --eval-cache /path/to/private_facescape_eval_cache \
  --training-root /path/to/new_matched_training_root \
  --output-root /path/to/new_facescape_raw_root
```

For REALY, change `--dataset facescape` to `--dataset realy` and provide the
400-source REALY cache. Inference writes one `RAW_OUTPUTS.pt` and one terminal
receipt for every method-seed route. Pair formation and metrics remain a later
stage, so inference never reads a target-pair roster.

## Outputs

Training produces an immutable root with `ATTEMPT.json`, one terminal
checkpoint and trace for each of the 15 newly trained units, and
`TRAINING_TERMINAL.json`. The attempt and terminal receipts record the GPU,
CUDA, cuDNN, PyTorch, deterministic-algorithm and CUBLAS workspace settings.

Inference produces `routes/<method>/<seed>/RAW_OUTPUTS.pt`, route receipts, and
`INFERENCE_TERMINAL.json`. Each payload contains native and conserved UV
tensors. The runner checks exact equality on source-observed texels before it
writes a successful receipt.

Source checks and synthetic tests should finish in under a minute on a local
CPU. Before the formal root is created, the CUDA smoke runs one optimizer step
for each method under the same deterministic runtime and writes a bound PASS
receipt. Formal training runtime is intentionally not guessed; record the
measured wall time and hardware in the terminal receipt from the target RTX
4090 run.

## Frozen post-processing order

Training/inference and perceptual evaluation use separate environments. The
15 training units and both source-only inference runs stay in the RTX 4090
CUDA environment. LPIPS, YuNet/SFace, shared rendering, and metric assembly
run in the already prepared Linux CPU post-freeze environment recorded in
`requirements-postfreeze.txt`. Do not activate the FreeUV environment and do
not install packages or access the network during qualification or metrics.

The first SFace qualification attempt used OpenCV 4.7.0 and failed inside
YuNet with `getLayerData id=-1`. That attempt completed before any real-image
read (`real_image_reads=0`, `metric_rows=0`) and its failure terminal must be
retained unchanged. Before viewing any experimental image or metric, one
environment correction was recorded: a fresh isolated CPU environment pins
`opencv-python-headless==4.10.0.84` while keeping the same YuNet/SFace assets,
probe inputs, single-thread setting, disabled OpenCL, and zero CUDA calls. The
OpenCV 4.10.0 environment passed the synthetic YuNet probe and produced the
expected 128-dimensional SFace feature. A new qualification root must be used;
the OpenCV 4.7 failure terminal is provenance only and is never relabeled or
reused as PASS. No further environment change is allowed after qualification.

Run the source-only checks before opening any private cache:

```bash
python reproducibility/w5b49n_v14_matched_controls_v1/postprocess.py source-check
python reproducibility/w5b49n_v14_matched_controls_v1/test_postprocess_synthetic.py
python reproducibility/w5b49n_v14_matched_controls_v1/test_statistics_synthetic.py
```

Create a new qualification root. This phase reads only the frozen evaluator
assets. A failed LPIPS or SFace probe writes a retained failure terminal with
zero real-image reads and the execution phase must not be started.

```bash
python reproducibility/w5b49n_v14_matched_controls_v1/postprocess.py qualify \
  --evaluator-assets /absolute/path/evaluator_portable_source \
  --output-root /absolute/new/path/V14_EVALUATOR_QUALIFICATION
```

Only after both qualification terminals are PASS, run post-processing once in
a fresh output root:

```bash
python reproducibility/w5b49n_v14_matched_controls_v1/postprocess.py execute \
  --training-terminal /absolute/path/V14_TRAINING/TRAINING_TERMINAL.json \
  --d1-v14-raw-root /absolute/path/V14_D1_RAW \
  --d2-v14-raw-root /absolute/path/V14_D2_RAW \
  --d1-eval-cache /absolute/path/FACESCAPE_EVAL_CACHE \
  --d2-eval-cache /absolute/path/REALY_EVAL_CACHE \
  --d2-roster-root /absolute/path/REALY_PAIR_ROSTER_ROOT \
  --d1-b-lite-route-root /absolute/path/stage13/routes/b_lite__aligned \
  --d2-b-lite-route-root /absolute/path/stage15/routes/b_lite__aligned \
  --d1-lama-root /absolute/path/normalized_batches/D1/lama \
  --d2-lama-root /absolute/path/normalized_batches/D2/lama \
  --d1-zits-root /absolute/path/normalized_batches/D1/zits \
  --d2-zits-root /absolute/path/normalized_batches/D2/zits \
  --freeuv-root /absolute/path/W5B49N_FREEUV_D1D2_20260820V12 \
  --freeuv-archive /absolute/path/W5B49N_FREEUV_D1D2_20260820V12_PRIVATE_RESULTS_V1_2.tar.gz \
  --freeuv-safe-sample-map /absolute/path/safe_sample_map.v1.json \
  --qualification-root /absolute/new/path/V14_EVALUATOR_QUALIFICATION \
  --evaluator-assets /absolute/path/evaluator_portable_source \
  --output-root /absolute/new/path/V14_POSTPROCESS
```

The post-processor consumes only frozen outputs. It performs no training,
baseline inference, geometry estimation, or FreeUV forward pass. It writes the
complete analysis keyspace for 1,348 evaluable pairs, reuses the 2,696 existing
FreeUV endpoint renders, creates 24,264 renders for the other 18 routes, and
records all SFace failures explicitly. A source/target detection failure is
applied symmetrically to all 16 analysis routes for that pair. SFace
confirmatory testing remains eligible only when at least 18 FaceScape and 90
REALY identities retain one complete all-route pair; otherwise the statistical
program keeps descriptive results and closes that claim family.

Do not rerun FreeUV. Its wrapper terminal must bind the original V1.2 archive,
activity terminal, safe sample map, D1/D2 aggregate tensors, and target/render
manifests while recording `new_forward_count=0`.
