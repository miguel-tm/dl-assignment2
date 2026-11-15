# Prompt for VS Code Copilot — Phases 1 and 2

Read PLAN.md and DECISIONS.md in this repo first; they are the source of truth.
We are implementing **Phase 1 (helper functions)** and **Phase 2 (data preparation)**
only, in notebooks/SCS_3546_Assignment_02.ipynb. Do not start Phase 3.

## Context
- SCS 3546 Assignment 2: VGG16 (ImageNet, include_top=False) transfer learning on the
  5-class flower_photos dataset. Three models later: frozen base, unfreeze block5, unfreeze all.
- Environment: WSL2 Ubuntu, Python 3.12 venv, TF 2.16.2, Keras 3.15.1, NumPy 1.26.4,
  scikit-learn 1.5.1, matplotlib 3.8.4. GPU memory growth already enabled in the setup cell.
  Do not install or upgrade packages.

## Working rules
- I write the analysis and interpretation. In markdown cells, write only short headings
  and `[TODO Miguel: ...]` placeholders where explanation or interpretation is needed.
- Code: clear, commented, small functions with docstrings. No hardcoded Windows,
  /content/ or Drive paths (P02).
- Helpers go in notebook cells under a "Helper functions" heading, not a separate module (M04).
- Work in small steps. After each step, stop and tell me what to run and what output to expect.
  Do not commit; I commit myself.
- If something in PLAN.md or DECISIONS.md seems wrong or ambiguous, ask instead of guessing.

## Key design decisions to respect
- M01 (Option A): the model takes raw pixels 0–255. Inside the model:
  augmentation (training only) → keras.applications.vgg16.preprocess_input → VGG16 base → head.
  **make_tf_dataset must NOT call preprocess_input.**
- M02: build_head() returns a standalone keras.Model with input shape (7, 7, 512).
- M03: feature extractor = Input → preprocess_input → base (no augmentation);
  extraction never shuffles; features returned as NumPy.
- M05: save/load weights only; rebuild architecture in code.
- M06: set_trainable(base, mode) with modes "none" / "block5" / "all";
  set base.trainable = True first, then freeze per layer by name; caller recompiles.
- Q1: the test set is never used in Phases 1–2 except for split statistics.
- D01 (resize method) and D02 (validation share) are open: make them parameters
  (resize_mode, val_fraction) and ask me for the values.

## Phase 1 — implement, in this order
1. set_seeds(seed)
2. load_dataset() — keras.utils.get_file on the flower_photos URL (extract); keep only .jpg;
   sorted paths; integer labels; class_names sorted
3. make_split(paths, labels, test_size=0.2, val_fraction, seed) — two-stage stratified split
4. make_tf_dataset(paths, labels, batch_size=32, shuffle=False, resize_mode) —
   decode → resize 224×224 → float32 (0–255) → batch → prefetch
5. build_vgg_base(), build_augmentation(), build_feature_extractor(base),
   extract_features(extractor, ds)
6. build_head(pooling="flatten"|"gap", units=[256], dropout=0.0, l2=0.0, batch_norm=False)
7. build_full_model(base, head, augmentation=None)
8. set_trainable(base, mode), count_params(model)
9. plot_history(history, title), plus a simple way to record training time per fit

## Phase 1 checkpoint cell (subset of ~200 images)
Print/verify:
- class counts per split
- one batch: shape (32, 224, 224, 3), min ≈ 0, max ≈ 255
- feature extractor output (32, 7, 7, 512)
- trainable params for "none" / "block5" / "all" (expect ≈ 0 / ~7.1M / ~14.7M, plus head)
- head trains 1 epoch on cached subset features; plot_history shows curves

## Phase 2 — data preparation cells
- Full stratified 80/20 split with fixed SEED, then validation from train
- Table and bar chart: class proportions for train / val / test
- Grid of a few resized sample images per class
- Markdown placeholders for: resize method (D01), normalization = preprocess_input
  inside the model (Q3, M01), and the split rationale

Start with step 1 and the "Helper functions" heading. Show me the code before moving on.
