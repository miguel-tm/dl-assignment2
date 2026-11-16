# STATUS

Progress log by phase. Read with [PLAN.md](PLAN.md) (what to do) and [DECISIONS.md](DECISIONS.md) (why).
New Copilot/chat sessions: read this file first to see what is done and what comes next.

## Current state

| Phase | Status | Tag |
|---|---|---|
| 1. Helper functions | Done, checkpoint passed | — |
| 2. Data preparation | Next | `m1-data` (planned) |
| 3. Model 1 (frozen base) | Not started | |
| 4. Model 2 (fine-tune block5) | Not started | |
| 5. Model 3 (fine-tune all) | Not started | |
| 6. Comparison | Not started | |
| 7. Conclusions | Not started | |

**Environment:** WSL2 Ubuntu, venv `~/venvs/dl-a2`, Python 3.12, TF 2.16.2 (CUDA), Keras 3.15.1, RTX A2000 (~5.6 GB VRAM).
Kernel: "Python 3.12 (dl-a2, WSL GPU)".

**Known quirks**
- Re-running the Setup cell alone fails (GPU memory growth can't be set after init) → restart kernel.
- `tf.function` retracing warnings and ptxas register-spill messages are harmless.
- Models containing augmentation layers can't use XLA; predictions differ from the XLA path by ~1e-3 (same argmax). Harmless.

---

## Phase 1: Helper functions

Cell numbers are as of the Phase 1 commit; they shift as cells are added. Function names are stable.

### Cell 4: Setup
- `TF_CPP_MIN_LOG_LEVEL=2` hides TF info logs (E07); `IN_COLAB` flag (P02).
- Prints library versions and asserts Keras 3.
- GPU memory growth on, so TF takes VRAM as needed (E08).
- `SEED = 42`; `set_global_seed()` seeds Python, NumPy and TF. No op determinism (E09).
- `%matplotlib inline` is required: with ipykernel 7.4, figures otherwise print as `<Figure ...>` text.

### Cell 6: `load_dataset()`
- Downloads `flower_photos` via `keras.utils.get_file` (cached in `~/.keras/datasets` on WSL and Colab; no hardcoded paths, P02).
- Keeps `.jpg` only, sorted paths, integer labels in alphabetical class order → same split on every machine.
- Keras 3.15 extracts one folder deeper (`flower_photos/flower_photos/<class>/`); both layouts are handled.
- Output: 3,670 images: daisy 633, dandelion 898, roses 641, sunflowers 699, tulips 799.

### Cell 7: `make_split()`
- Two-stage stratified split: 80/20 train/test, then 15% of train → val (D02).
- Stratified because classes are mildly imbalanced.
- Output: train 2,495 / val 441 / test 734. Test is only counted, never used until the end (Q1).

### Cell 8: `load_image()`, `make_tf_dataset()`
- read → decode JPEG → stretch to 224×224 (D01) → float32 **0–255** → batch → prefetch.
- No `preprocess_input` here: the model normalizes its own input (M01).
- Only train is shuffled, at the path level (cheap). Val, test and feature caching are never shuffled (M03).
- Output: batch `(32, 224, 224, 3)`, min 0, max 255. Batches per split: 78 / 14 / 23.

### Cell 9: VGG base, augmentation, feature extractor
- `build_vgg_base()`: VGG16, ImageNet weights, no top → `(7, 7, 512)`, 14,714,688 params.
- `build_augmentation()`: flip, rotation, zoom, contrast. Active only in training; expects 0–255. Strengths are placeholders (tune in Phase 3).
- `build_feature_extractor(base)`: Input → `preprocess_input` → base, no augmentation (M03).
- `extract_features()`: runs the extractor once, returns NumPy features + labels in RAM. Enables hybrid caching (Q4): head experiments train in seconds.

### Cell 10: `build_head()`
- Standalone model with input `(7, 7, 512)` (M02).
- Pooling (`flatten` | `gap`) → per hidden layer: Dense → [BN] → ReLU → [Dropout] → 5-way softmax. L2 on hidden kernels only.
- Same head object trains on cached features, then sits on the base for Model 1, and carries weights into Models 2–3 (Q2).
- Params: Flatten 6,424,069 vs GAP 132,613 (~48× fewer).

### Cell 11: `build_full_model()`
- Input (0–255) → [augmentation] → `preprocess_input` → base → head. Base and head are shared objects, not copies.
- Finding: full model vs extractor → head differ by ~1e-3 (same predicted class). Cause: augmentation layers block XLA, so a different GPU code path runs. Without augmentation the diff is exactly 0. No fix needed.

### Cell 12: `set_trainable()`, `count_params()`
- Modes `none` | `block5` | `all`: sets `base.trainable = True`, then freezes per layer by name (M06). Recompile afterwards.
- `count_params()` returns trainable / non-trainable / total (for Phase 6 table).
- Base trainable counts: 0 / 7,079,424 / 14,714,688. CPU test confirmed only block5 weights change in `block5` mode.

### Cell 13: `timed_fit()`, `plot_history()`
- `timed_fit()` returns `(history, seconds)` for Phase 6.
- `plot_history()`: loss and accuracy side by side, train vs val; accuracy axis fixed at 0–1 for visual comparison. Returns nothing (returning the figure displayed it twice).

### Cell 14: scratch tests
Replaced by cell 16; can be deleted.

### Cells 15–16: Phase 1 checkpoint (~200 images, `assert`-based)

| Check | Result |
|---|---|
| Split counts | 2,495 / 441 / 734, stratified |
| Subset | 160 train [28 39 28 30 35], 40 val [7 10 7 7 9] |
| Batch | (32, 224, 224, 3), 0–255 (M01) |
| Extractor | (32, 7, 7, 512) |
| Trainable params (base + head) | none 132,613 · block5 7,212,037 · all 14,847,301 |
| GAP head, 2 epochs on cached features | train acc 0.27 → 0.74, val acc 0.50 → 0.65, val loss 1.86 → 1.42, 1.9 s |

The pipeline works end to end and the head learns. First-epoch loss 4.8 is high because GAP features are unnormalized (values up to ~100); learning rate / BN experiments in Phase 3 may help. Accuracies mean little with 40 val images.

### Deviations from the Phase 1 handoff
- `set_seeds` → `set_global_seed`.
- `build_head(units=...)` default is a tuple, not a list.
- `plot_history` returns nothing.
- Save/load weights + `TRAIN` flag moved to Phase 3 (P04 decided there).
- Checkpoint trains 2 epochs (handoff said 1).

### Doc changes
- DECISIONS: added E09; D01 = stretch, D02 = 0.15 (both Decided).
- PLAN: rename to `set_global_seed`; reproducibility limitation added to Phase 7; weights helpers moved to Phase 3.

### Open items carried forward
- Phase 2: `[TODO Miguel]` markdown for D01 (stretch), Q3/M01 (`preprocess_input` is the normalization), split rationale.
- Phase 3: tune augmentation strengths; weights helpers + `TRAIN` flag, decide P04; optional `save_path` on `plot_history`; optional integer epoch ticks.
