# PLAN — dl-assignment2 (SCS 3546, Assignment 2)

Transfer learning with VGG16 (ImageNet) on the 5-class flower_photos dataset.
Deliverable: one fully executed notebook (notebooks/SCS_3546_Assignment_02.ipynb)
with the write-up in markdown cells. Decisions are logged in DECISIONS.md.

## Ground rules
- The test set is used **once**, at the end (Phase 6). All tuning uses validation only (Q1).
- Every experiment gets a short markdown note: what changed, why, what happened.
- Develop on a small subset (a few hundred images, 1–2 epochs) before full runs.
- Commit after each phase; tag milestones (m1-data, m2-model1, ...).
- Copilot (chat or VS Code) writes code scaffolding and explains; the analysis,
  interpretation and write-up are mine.

## Architecture at a glance (M01, M02)

```
Final Model 1 / Model 2 / Model 3 (self-contained, raw pixels in):
  Input 224×224×3 (0–255) → [augmentation, training only] → preprocess_input → VGG16 base → head → 5 softmax

Feature extractor (Q4 caching, no augmentation):
  Input 224×224×3 (0–255) → preprocess_input → VGG16 base → 7×7×512  ──► cached to RAM

Head (standalone sub-model, M02):
  Input 7×7×512 → Flatten or GAP → Dense ... → Dense(5, softmax)
```

The same head object trains on cached features, is then stacked on the base for the
final Model 1, and its weights carry into Models 2 and 3 (Q2).

## Rubric map (15 pts)

| Criterion | Pts | Where |
|---|---|---|
| Data preparation | 1 | Phase 2 |
| Model 1 (frozen base, progression of experiments) | 5 | Phase 3 |
| Model 2 (unfreeze block5) | 3 | Phase 4 |
| Model 3 (unfreeze all) | 3 | Phase 5 |
| Comparison | 1 | Phase 6 |
| Explanations (comments, rationale, interpretation) | 2 | All phases |

Also required: name + student number at the top; submit the executed .ipynb.

## Phase 0 — Environment ✅
WSL2 + Ubuntu, Python 3.12 (uv), TF 2.16.2 [and-cuda], Keras 3.15.1, RTX A2000
visible. Scaffold committed. Setup cell passes.

## Phase 1 — Helper functions (cells inside the notebook, see M04) ✅
Utilities
- [x] set_global_seed(seed) — Python, NumPy, TensorFlow in one call (defined in the Setup cell; no op determinism, E09)
- [x] plot_history(history) — loss + accuracy, train vs. val; same axes across experiments
- [x] count_params(model) — trainable / non-trainable counts (needed for Phase 6 table)
- [x] training time captured for every fit (needed for Phase 6 table)

Data
- [x] load_dataset() — keras.utils.get_file; keep only .jpg (skip LICENSE.txt); **sort** paths; integer labels + class names
- [x] make_split() — two-stage stratified split: 80/20 train/test, then val from train (stratify both times)
- [x] make_tf_dataset(paths, labels, shuffle) — decode → resize 224×224 → float32 **0–255** → batch → prefetch.
      **No preprocess_input here** (M01). Shuffle the training split only; never shuffle for caching.

Model
- [x] build_vgg_base() — VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3)); verify output (None,7,7,512)
- [x] build_augmentation() — Keras augmentation layers (flip/rotation/zoom...); active only when training
- [x] build_feature_extractor(base) — Input → preprocess_input → base (no augmentation)
- [x] extract_features(extractor, ds) — returns (features, labels) as NumPy; ~370 MB for the full set, keep in RAM, not on GPU
- [x] build_head(config) — standalone keras.Model, input (7,7,512); configurable pooling (Flatten/GAP), units, layers, dropout, L2, batch norm
- [x] build_full_model(base, head, augmentation) — Input → augmentation → preprocess_input → base → head
- [x] set_trainable(base, mode) — "none" / "block5" / "all": set base.trainable=True first, then freeze per layer; always recompile after

Checkpoint (small subset, ~200 images)
- [x] Class counts per split printed
- [x] One batch: shape (32,224,224,3); min/max ≈ 0/255 (raw pixels, confirms M01)
- [x] Feature extractor on that batch → (32,7,7,512)
- [x] set_trainable in all three modes → trainable params ≈ 0 / ~7.1M / ~14.7M (+ head)
- [x] Head fits on cached subset features (5 epochs); plot_history works
- [x] Commit

## Phase 2 — Data preparation (1 pt) ✅
- [x] Stratified 80/20 train/test split (fixed SEED)
- [x] Validation split carved from train, also stratified (share: see D02)
- [x] Table/plot: class proportions match across train / val / test
- [x] Show a few resized sample images per class
- [x] Markdown: resize method (D01), and that preprocess_input **is** the normalization step (Q3), applied inside the model (M01)
- [x] Checkpoint: commit, tag m1-data

## Phase 3 — Model 1: frozen VGG16 (5 pts)
Hybrid approach (Q4):
- [ ] Run the feature extractor once; cache 7×7×512 features for train and val (un-augmented)
- [ ] Baseline head; then a justified progression, e.g.:
  - [ ] Flatten vs. GlobalAveragePooling
  - [ ] neurons / number of dense layers
  - [ ] optimizer and learning rate
  - [ ] dropout / L2 regularization
  - [ ] batch normalization
  - [ ] early stopping
- [ ] Learning curves for each experiment + a summary table of val accuracy
- [ ] Final Model 1: best head stacked on the frozen base, one end-to-end run **with data augmentation**
      (augmentation can only happen here — cached features are fixed)
- [ ] save/load weights helpers to outputs/ + TRAIN flag (moved from Phase 1); decide P04 (how Colab gets the weights) here
- [ ] Save weights to outputs/; report train and val accuracy
- [ ] Checkpoint: commit, tag m2-model1

## Phase 4 — Model 2: unfreeze block5 (3 pts)
- [ ] Rebuild full model, load final Model 1 weights (Q2)
- [ ] set_trainable(base, "block5"); **recompile**; low learning rate (e.g. 1e-5)
- [ ] Batch 32, fall back to 16 on OOM (E08)
- [ ] Learning curves; train and val accuracy; save weights
- [ ] Checkpoint: commit, tag m3-model2

## Phase 5 — Model 3: unfreeze all layers (3 pts)
- [ ] Rebuild full model, load final Model 1 weights (same start as Model 2, fair comparison)
- [ ] set_trainable(base, "all"); recompile; very low learning rate
- [ ] Watch for overfitting; early stopping
- [ ] Learning curves; train and val accuracy; save weights
- [ ] Checkpoint: commit, tag m4-model3

## Phase 6 — Final evaluation and comparison (1 pt)
- [ ] Evaluate the three final models on the test set **once**
- [ ] Table: train / val / test accuracy + trainable parameters + training time
- [ ] Optional: confusion matrix for the best model
- [ ] Markdown: which model is best and why (frozen vs. partial vs. full fine-tuning trade-off)

## Phase 7 — Conclusions and submission
- [ ] Conclusions and limitations (include: GPU runs are not bit-for-bit reproducible, E09; some dataset images are off-topic or mislabeled, seen in the Phase 2 sample grid)
- [ ] Re-run the whole notebook top to bottom (TRAIN = True); check outputs
- [ ] Name + student number in title cell
- [ ] Final commit, tag submitted; submit .ipynb with the late note
- [ ] Optional (P03): open in Colab with GPU runtime as a portability check
