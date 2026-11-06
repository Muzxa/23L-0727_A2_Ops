# Local setup and execution

Use Python 3.11 or 3.12 for broad TensorFlow wheel availability. A separate environment is required; do not install into system Python.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "from visionguard.training import environment; environment('outputs/environment')"
```

`venv` isolates packages. `pip install` installs dependencies. `environment` records Python, TensorFlow, Keras, GPU devices and a complete exact-version pip freeze. Dependency ranges bootstrap installation; `requirements-lock.txt` captures the actual run. TensorFlow CPU is sufficient for smoke checks but full training is better on Colab GPU. On Colab, keep its TensorFlow installation rather than replacing the CUDA stack.

Download/unzip the exact Kaggle dataset into `data/raw` (or use the Colab downloader). Inspect actual folder names; change the JSON mapping accordingly:

```bash
python -m visionguard.data --root data/raw --processed data/processed --out outputs/data --class-map '{"AiArtData":"ai","RealArt":"real"}'
python -m scripts.dvc_demo --destination outputs/dvc_toy
python -m scripts.run_training --root data/processed --splits outputs/data --out outputs/training
python -m unittest discover -s tests -v
```

The auditor raises an error for an image with an unrecognized label folder. It does not silently label unknown folders or silently ignore decoding failures. Nonimage sidecars are listed separately; corruption totals cover candidate image files.

If decoding/augmentation memory is tight, set `--batch 16`. Do not reduce trial count below eight for the graded run. Changing hyperparameters invalidates resume state: use a separate run directory BEFORE inspecting any test results. A fixed validation split can itself be overfit through many trials. Once test has been opened, keep it held out from further model decisions and report the results honestly.

Notebook rebuild after editing source/docs:

```bash
python scripts/build_notebook.py
```

The generated notebook embeds source so it runs independently of a Git repository. Rebuild whenever source changes.
