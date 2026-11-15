# DECISIONS — dl-assignment2 (SCS 3546, Assignment 2)

Log of environment, process and modelling decisions. Status: Decided / Open / Superseded.

## Environment & repo

| ID | Decision | Rationale | Status |
|---|---|---|---|
| E01 | Train locally on the NVIDIA RTX A2000 Laptop GPU via WSL2 (Ubuntu) | Native Windows TensorFlow is CPU-only after 2.10; VGG16 fine-tuning (Models 2–3) is impractical on CPU. Also a reusable skill for the final project. Supersedes A1's "CPU is enough" choice. | Decided (verified: matmul ran on GPU:0) |
| E02 | Python 3.12 in a uv-managed venv at `~/venvs/dl-a2`; never modify Ubuntu's system Python (3.14) | TensorFlow does not support Python 3.14; TF 2.16.2 supports 3.9–3.12 | Decided |
| E03 | Pin library versions to A1: TF 2.16.2 (`[and-cuda]`), Keras 3.15.1, NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.5.1, matplotlib 3.8.4; Pillow pinned at install | Known-good combination from A1; keeps the two assignments comparable | Decided |
| E04 | Repo lives in the Linux file system (`~/repos/dl-assignment2`); reach it from Windows via `\\wsl.localhost\Ubuntu\...` | `/mnt/c` I/O is much slower; matters when reading ~3,700 images per epoch | Decided |
| E05 | Edit files only through VS Code connected to WSL, not Windows apps | Avoid line-ending and permission problems | Decided |
| E06 | GitHub is the backup for this repo; commit and push regularly; repo is private | WSL files are outside Windows Documents backup | Decided |
| E07 | Keep `TF_CPP_MIN_LOG_LEVEL=2`; the cuFFT/cuDNN/cuBLAS "Unable to register" lines are known harmless noise | Level 3 would also hide real errors | Decided |
| E08 | Enable GPU memory growth; default batch size 32, fall back to 16 on OOM | About 5.6 GB usable VRAM; Models 2–3 store gradients for more layers | Decided |

## Portability

| ID | Decision | Rationale | Status |
|---|---|---|---|
| P01 | Priority order: (1) train on local GPU + run in VS Code, (2) running/evaluating models also works in Colab, (3) training also works in Colab | The difference in effort is small, so build portability in from the start | Decided |
| P02 | Same code everywhere: only environment detection, paths and GPU checks differ; dataset via `keras.utils.get_file`; no hardcoded `/content/`, Drive or Windows paths | Notebook must run in both environments without edits | Decided |
| P03 | Colab testing is deferred to the end, as a final check | Local GPU is the main path | Decided |
| P04 | Separate training from evaluation: save trained weights to `outputs/`; a `TRAIN` flag chooses retraining vs. loading | Training is heavy, evaluation is light. Open: weights are git-ignored, so Colab must either retrain or load them from Drive | Open |
| P05 | In Colab, do not `pip install -r requirements.txt` | Can conflict with Colab's preinstalled CUDA/TF stack | Decided |

## Modelling (open — to settle before PLAN.md)

| ID | Question | Leaning | Status |
|---|---|---|---|
| Q1 | Evaluate the test set only once, at the end, for all three final models | Yes: tune on validation only | Decided |
| Q2 | Models 2–3 continue from Model 1's trained head with a low learning rate | Yes: standard fine-tuning, protects pretrained weights | Decided |
| Q3 | Use VGG16 `preprocess_input` as the normalization step and say so explicitly in markdown | Yes | Decided |
| Q4 | Model 1: cached features, end-to-end training, or a hybrid (cached for head experiments plus one augmented run) | Hybrid | Decided |
