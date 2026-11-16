# DECISIONS — dl-assignment2 (SCS 3546, Assignment 2)

Log of environment, process and modelling decisions. Status: Decided / Open / Superseded.

## Environment & repo

| ID | Decision | Rationale | Status |
|---|---|---|---|
| E01 | Train locally on the NVIDIA RTX A2000 Laptop GPU via WSL2 (Ubuntu) | Native Windows TensorFlow is CPU-only after 2.10; VGG16 fine-tuning (Models 2–3) is impractical on CPU. Also a reusable skill for the final project. Supersedes A1's "CPU is enough" choice. | Decided (verified: matmul ran on GPU:0) |
| E02 | Python 3.12 in a uv-managed venv at ~/venvs/dl-a2; never modify Ubuntu's system Python (3.14) | TensorFlow does not support Python 3.14; TF 2.16.2 supports 3.9–3.12 | Decided |
| E03 | Pin library versions to A1: TF 2.16.2 ([and-cuda]), Keras 3.15.1, NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.5.1, matplotlib 3.8.4; Pillow pinned at install | Known-good combination from A1; keeps the two assignments comparable | Decided |
| E04 | Repo lives in the Linux file system (~/repos/dl-assignment2); reach it from Windows via \\\\wsl.localhost\\Ubuntu\\... | /mnt/c I/O is much slower; matters when reading ~3,700 images per epoch | Decided |
| E05 | Edit files only through VS Code connected to WSL, not Windows apps | Avoid line-ending and permission problems | Decided |
| E06 | GitHub is the backup for this repo; commit and push regularly; repo is private | WSL files are outside Windows Documents backup | Decided |
| E07 | Keep TF_CPP_MIN_LOG_LEVEL=2; the cuFFT/cuDNN/cuBLAS "Unable to register" lines are known harmless noise | Level 3 would also hide real errors | Decided |
| E08 | Enable GPU memory growth; default batch size 32, fall back to 16 on OOM | About 5.6 GB usable VRAM; Models 2–3 store gradients for more layers | Decided |
| E09 | Seeds only (set_global_seed → keras.utils.set_random_seed); do **not** call tf.config.experimental.enable_op_determinism() | Seeds make splits and initial weights reproducible. Remaining GPU non-determinism moves accuracy by a fraction of a percent, unlikely to change conclusions; full determinism slows training and some ops can raise errors. Mentioned as a limitation in Phase 7 | Decided |

## Portability

| ID | Decision | Rationale | Status |
|---|---|---|---|
| P01 | Priority order: (1) train on local GPU + run in VS Code, (2) running/evaluating models also works in Colab, (3) training also works in Colab | The difference in effort is small, so build portability in from the start | Decided |
| P02 | Same code everywhere: only environment detection, paths and GPU checks differ; dataset via keras.utils.get_file; no hardcoded /content/, Drive or Windows paths | Notebook must run in both environments without edits | Decided |
| P03 | Colab testing is deferred to the end, as a final check | Local GPU is the main path | Decided |
| P04 | Separate training from evaluation: save trained weights to outputs/; a TRAIN flag chooses retraining vs. loading | Training is heavy, evaluation is light. Open: weights are git-ignored, so Colab must either retrain or load them from Drive | Open |
| P05 | In Colab, do not pip install -r requirements.txt | Can conflict with Colab's preinstalled CUDA/TF stack | Decided |

## Modelling — process

| ID | Decision | Rationale | Status |
|---|---|---|---|
| Q1 | Evaluate the test set only once, at the end, for all three final models | Tune on validation only; avoids leaking test information into choices | Decided |
| Q2 | Models 2–3 start from final Model 1's weights (head trained); recompile; low learning rate | Standard fine-tuning; a random head would send large gradients into pretrained layers and damage them | Decided |
| Q3 | VGG16 preprocess_input is the normalization step; stated explicitly in markdown | It reproduces the normalization VGG16 was trained with (RGB→BGR, subtract ImageNet channel means) | Decided |
| Q4 | Model 1 hybrid: cached 7×7×512 features for head experiments, then one end-to-end augmented run for the final Model 1 | Caching makes head experiments fast; augmentation needs live images, so it can only happen end-to-end | Decided |

## Modelling — architecture & data

| ID | Decision | Rationale | Status |
|---|---|---|---|
| M01 | **Option A — self-contained model.** The model takes raw pixels (0–255). Order inside the model: augmentation (training only) → preprocess_input → VGG16 base → head. The tf.data pipeline only decodes, resizes, batches and prefetches. | Augmentation layers (especially brightness/contrast) assume 0–255 input, so they must run before preprocess_input. One artifact defines its own input handling: clearer, portable (P02), nothing to keep in sync between pipeline and model. Option B (data-centric: preprocessing in tf.data) considered and rejected for this assignment — valid, but adds augmented / non-augmented / cached modes to keep consistent. | Decided |
| M02 | The head is a standalone keras.Model with input (7,7,512) | Same object trains on cached features, stacks on the base for the final Model 1, and carries weights into Models 2–3 (Q2) without shape/name mismatches | Decided |
| M03 | Feature caching uses a separate feature extractor: Input → preprocess_input → base (no augmentation); features kept in RAM as NumPy (~370 MB), never shuffled during extraction | Cached features must match exactly what the base sees at validation time; no shuffling keeps features aligned with labels | Decided |
| M04 | Helper functions live as cells in the notebook (under a "Helper functions" heading), not in a separate module | Deliverable is a single executed .ipynb; the marker must be able to run it as-is; also counts toward "code clearly commented" | Decided |
| M05 | Save/load **weights** (not full .keras models); rebuild the architecture in code, then load weights | Avoids serialization issues with preprocess_input inside the model; fits P04 | Decided |
| M06 | set_trainable(): set base.trainable = True first, then freeze per layer by name (block5_* for Model 2); always recompile after changing trainability | Keras nested-model trainability is easy to get silently wrong; the parameter-count check proves it worked. VGG16 has no BatchNorm, so the BN fine-tuning trap does not apply | Decided |

## Open — to settle before / during Phase 2

| ID | Question | Options | Status |
|---|---|---|---|
| D01 | Resize method | Stretch to 224×224 (simple; distorts aspect ratio) vs. crop/pad (keeps aspect ratio; may cut content). State choice in Phase 2 markdown | Decided: **stretch** (resize_mode="stretch"); short justification in Phase 2 markdown |
| D02 | Validation share | % of the training set (assignment wording: 10–20% of training set size) — pick exact value | Decided: **0.15** of the training set (val_fraction=0.15) |
