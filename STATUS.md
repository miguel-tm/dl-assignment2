# STATUS

Progress log by phase. Read with [PLAN.md](PLAN.md) (what to do) and [DECISIONS.md](DECISIONS.md) (why).
New Copilot/chat sessions: read this file first to see what is done and what comes next.

## Current state

| Phase | Status | Tag |
|---|---|---|
| 1. Helper functions | Done, merged to `main` | — |
| 2. Data preparation | Done, merged to `main` | `m1-data` |
| 3. Model 1 (frozen base) | Next | |
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

Cell numbers count every cell, markdown included, from the top of the notebook. Function names are stable.

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

### Cells 14–15: Phase 1 checkpoint (~200 images, `assert`-based)

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
- `plot_history` returns nothing; Miguel later switched its epoch axis to integer ticks.
- Save/load weights + `TRAIN` flag moved to Phase 3 (P04 decided there).
- Checkpoint trains 5 epochs (handoff said 1; Miguel raised it from 2 to 5 after the run above).
- Scratch test cell deleted before the Phase 1 commit.

### Doc changes
- DECISIONS: added E09; D01 = stretch, D02 = 0.15 (both Decided).
- PLAN: rename to `set_global_seed`; reproducibility limitation added to Phase 7; weights helpers moved to Phase 3.

### Open items carried forward
- Phase 2: `[TODO Miguel]` markdown for D01 (stretch), Q3/M01 (`preprocess_input` is the normalization), split rationale (placeholders added in Phase 2).
- Phase 3: tune augmentation strengths; weights helpers + `TRAIN` flag, decide P04; optional `save_path` on `plot_history`.

---

## Phase 2: Data preparation

All cells sit under "## 1. Data Preparation" (cell 16). Branch `phase2-data`.
Phase 2 adds no new helpers; it applies the Phase 1 helpers to the full dataset and shows the results.

### Cell 17: split and class-proportion table
- `set_global_seed()`, `load_dataset()`, `make_split()` → module-level `paths`, `labels`, `class_names`, `splits` (reused by later phases).
- Re-runs the split instead of reusing the checkpoint's, so Phase 2+ works even if the checkpoint cell is skipped. Same SEED → identical split.
- Asserts: no image in two splits; split sizes sum to 3,670.
- Tables (pandas): counts per class × split (+ `all`, `total`), and class share (%) per split.
- Output: train 2,495 / val 441 / test 734 (68.0 / 12.0 / 20.0% of all images). Largest class-share gap vs. full dataset: **0.08 percentage points** (dandelion, test).
- Why gaps aren't zero: splits hold whole images; one image is 0.23 pp of val and 0.14 pp of test, so 0.08 pp is less than one image.

### Cell 18: class-proportion bar chart
- Grouped bars per class (train / val / test / all). Bars within each class are visually equal: dandelion ~24.5%, tulips ~21.8%, sunflowers ~19.0%, daisy and roses ~17.2-17.5%.

### Cell 20: sample grid
- 4 random training images per class (`np.random.default_rng(SEED)`), loaded with `load_image()`, so they show exactly what the model receives (stretched to 224×224, D01).
- Sets module-level `train_paths`, `train_labels` = full training split (overwrites the checkpoint's subset names).
- Titles give the original width×height. Sizes vary: mostly landscape (320×240, 500×333, 500×375), a few square (240×240) and portrait (180×240, 184×240). Landscape images get squeezed horizontally, portrait ones widened; many are smaller than 224 on one side, so they are also upscaled.
- Data-quality observation: some images are off-topic or noisy, e.g. a dog with angel wings labeled "roses", a person in front of sunflowers. Candidate for the limitations section (PLAN Phase 7).

### Cells 19 and 21: markdown placeholders (Miguel)
- Cell 19 "Split rationale": 80/20 stratified, fixed SEED, val = 15% of train (D02), test once (Q1).
- Cell 21 "Resizing (D01)": stretch vs. crop/pad, distortion visible in the grid.
- Cell 21 "Normalization (Q3, M01)": `preprocess_input` (RGB→BGR, subtract ImageNet means) inside the model; pipeline delivers 0-255.

### Doc changes
- PLAN: Phase 1 and Phase 2 code items ticked; label-noise note added to the Phase 7 limitations item.
- No new decisions in DECISIONS.md.

### Closed
- Miguel filled the three placeholders; merged to `main`, tagged `m1-data`.
