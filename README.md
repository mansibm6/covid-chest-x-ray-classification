# COVID chest X-ray classification

Trains a binary convolutional classifier on 224×224 chest X-ray images.

## Setup and execution

Use Python 3.11 and a separate virtual environment for this project. Run commands from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_inputs.py
python -m jupyterlab
```

On Windows, activate with `.venv\Scripts\activate`. For notebooks, select this environment’s kernel and execute from the top in a fresh kernel. The first code cell exposes `INPUT_PATHS` where old machine-specific paths existed. `PROJECT_DATA_DIR` changes the base for those configured paths; remaining relative paths use the notebook working directory.

## Notebooks

- [Covidtest.ipynb](Covidtest.ipynb)

## Inputs

Supply these inputs yourself; no dataset is downloaded automatically:

- `data/train/`
- `data/test/`

## Outputs

An in-memory CNN, loss/accuracy plots, and evaluation metrics.

## Implementation and limitations

Use matching class subdirectories in both splits, including normal for the example image. The original test split is also used during training validation; its reported score is not an untouched final holdout.

Reusable model/analysis functions live in `experiment_covidtest.py`. Notebooks retain the experiment sequence, training, and visual narrative.

## Verification

The workspace regression suite checks source/notebook validity and selected behavior with bundled or synthetic inputs. Full experiment training and historical metric reproduction are not implied. See the workspace `REFACTOR_REPORT.md` for exact test coverage and unresolved runtime requirements.

## Original references

These links were retained from the original project notes; availability and licensing must be checked by the person obtaining the data.

- https://drive.google.com/drive/folders/1SSsnEUkZ-G96PTDyxVLvmz_GNWLcW-ih?usp=sharing
