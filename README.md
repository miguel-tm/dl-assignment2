# dl-assignment2

SCS 3546 Assignment 2: transfer learning with VGG16 on the TensorFlow
`flower_photos` dataset (5 classes).

## Project structure

```
dl-assignment2/
├── data/                  # Dataset location (contents are git-ignored)
├── notebooks/
│   └── SCS_3546_Assignment_02.ipynb
├── outputs/               # Saved figures and tables (contents are git-ignored)
├── .gitignore
├── README.md
└── requirements.txt
```

## Local setup (WSL2 Ubuntu, NVIDIA GPU)

Tested with Python 3.12, TensorFlow 2.16.2 and an RTX A2000 (about 5.6 GB usable VRAM).

1. Install [uv](https://docs.astral.sh/uv/) if you do not have it yet.
2. Create and activate a Python 3.12 environment:

   ```bash
   uv venv ~/venvs/dl-a2 --python 3.12
   source ~/venvs/dl-a2/bin/activate
   ```

3. Install the dependencies:

   ```bash
   uv pip install -r requirements.txt
   ```

4. Register the Jupyter kernel:

   ```bash
   python -m ipykernel install --user --name dl-a2 --display-name "Python 3.12 (dl-a2, WSL GPU)"
   ```

5. Open `notebooks/SCS_3546_Assignment_02.ipynb` and select the kernel
   **Python 3.12 (dl-a2, WSL GPU)**.

## Google Colab

1. Upload or open the notebook in Colab.
2. Select a GPU runtime: **Runtime > Change runtime type > GPU**.
3. Do **not** run `pip install -r requirements.txt` in Colab. Colab already ships
   with a compatible TensorFlow, and reinstalling it can break the runtime.

The notebook detects Colab automatically, so no code changes are needed.

## Dataset

The `flower_photos` dataset downloads automatically through
`keras.utils.get_file`, so there is nothing to download by hand.
