# PLAN — dl-assignment2 (SCS 3546, Assignment 2)

Transfer learning with VGG16 (ImageNet) on the 5-class `flower_photos` dataset.
Deliverable: one fully executed notebook (`notebooks/SCS_3546_Assignment_02.ipynb`)
with the write-up in markdown cells. Decisions are logged in `DECISIONS.md`.

## Ground rules

- The test set is used **once**, at the end (Phase 6). All tuning uses validation only.
- Every experiment gets a short markdown note: what changed, why, what happened.
- Develop on a small subset (a few hundred images, 1–2 epochs) before full runs.
- Commit after each phase; tag milestones (`m1-data`, `m2-model1`, ...).

## Rubric map (15 pts)

| Criterion | Pts | Where |
|---|---|---|
| Data preparation | 1 | Phase 2 |
| Model 1 (frozen base, progression of experiments) | 5 | Phase 3 |
| Model 2 (unfreeze block5) | 3 | Phase 4 |
| Model 3 (unfreeze all) | 3 | Phase 5 |
| Comparison | 1 | Phase 6 |
| Explanations (comments, rationale, interpretation) | 2 | All phases |

Also required: name + student number at the top; submit the executed `.ipynb`.

## Phase 0 — Environment ✅

WSL2 + Ubuntu, Python 3.12 (uv), TF 2.16.2 `[and-cuda]`, Keras 3.15.1, RTX A2000
visible. Scaffold committed. Setup cell passes.

## Phase 1 — Helper functions

- [ ] `load_dataset()` — download via `keras.utils.get_file`, list files + labels
- [ ] `make_split()` — stratified train/val/test
- [ ] `make_tf_dataset()` — decode → resize 224×224 → `preprocess_input` → batch/prefetch
- [ ] `build_vgg_base()` — `VGG16(weights="imagenet", include_top=False)`
- [ ] `build_head()` — configurable dense head (units, layers, dropout, batch norm)
- [ ] `plot_history()` — loss + accuracy, train vs. validation
- [ ] `set_trainable()` — freeze all / unfreeze block5 / unfreeze all
- [ ] Checkpoint: functions run on a small subset; commit

## Phase 2 — Data preparation (1 pt)

- [ ] Stratified 80/20 train/test split (fixed `SEED`)
- [ ] Validation split (~15–20%) carved from train, also stratified
- [ ] Table/plot: class proportions match across train / val / test
- [ ] Show a few resized sample images per class
- [ ] Markdown: resizing method, and that `preprocess_input` **is** the normalization step
- [ ] Checkpoint: commit, tag `m1-data`

## Phase 3 — Model 1: frozen VGG16 (5 pts)

Hybrid approach (Q4):
- [ ] Run the frozen base once; cache the 7×7×512 features for train and val
- [ ] Baseline head; then a justified progression, e.g.:
  - [ ] neurons / number of dense layers
  - [ ] optimizer and learning rate
  - [ ] dropout / L2 regularization
  - [ ] batch normalization
  - [ ] early stopping
- [ ] Learning curves for each experiment + a summary table of val accuracy
- [ ] Final Model 1: one end-to-end run (frozen base) **with data augmentation**
- [ ] Save weights to `outputs/`; report train and val accuracy
- [ ] Checkpoint: commit, tag `m2-model1`

## Phase 4 — Model 2: unfreeze block5 (3 pts)

- [ ] Start from final Model 1 weights (Q2)
- [ ] Unfreeze `block5_conv1–3`; **recompile**; low learning rate (e.g. 1e-5)
- [ ] Batch 32, fall back to 16 on OOM
- [ ] Learning curves; train and val accuracy; save weights
- [ ] Checkpoint: commit, tag `m3-model2`

## Phase 5 — Model 3: unfreeze all layers (3 pts)

- [ ] Start from final Model 1 weights (same starting point as Model 2, for a fair comparison)
- [ ] Unfreeze all layers; recompile; very low learning rate
- [ ] Watch for overfitting; early stopping
- [ ] Learning curves; train and val accuracy; save weights
- [ ] Checkpoint: commit, tag `m4-model3`

## Phase 6 — Final evaluation and comparison (1 pt)

- [ ] Evaluate the three final models on the test set **once**
- [ ] Table: train / val / test accuracy + trainable parameters + training time
- [ ] Optional: confusion matrix for the best model
- [ ] Markdown: which model is best and why (frozen vs. partial vs. full fine-tuning trade-off)

## Phase 7 — Conclusions and submission

- [ ] Conclusions and limitations
- [ ] Re-run the whole notebook top to bottom (`TRAIN = True`); check outputs
- [ ] Name + student number in title cell
- [ ] Final commit, tag `submitted`; submit `.ipynb` with the late note
- [ ] Optional (P03): open in Colab with GPU runtime as a portability check
