"""Build a self-contained Colab notebook embedding the exact project source files."""
import json
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb.metadata = {'colab': {'name': 'VisionGuard_Tasks_1_2.ipynb'},
               'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python'}, 'accelerator': 'GPU'}
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))
md('''# VisionGuard — Tasks 1 and 2
This notebook contains the project code; no GitHub upload is needed to start.

**Storage:** Drive mounting is optional. If it fails, the notebook falls back to local runtime storage. Set `USE_GOOGLE_DRIVE = False` to skip the mount attempt. Download outputs before the runtime ends; local files are temporary.

**Before running:** choose **Runtime → Change runtime type → Python 3 → T4 GPU** (or another GPU).
Run cells in order. Do not use Run all until you have checked the folder labels in the dataset section.
Replace your roll number below. Use the same `<RollNumber>_A2_Ops` name on GitHub and DagsHub.

This notebook creates actual evidence when run. It contains no claimed dataset statistics or model scores in advance.
Full eight-trial tuning is required; do not report a shortened smoke test as final results.
The optional Grad-CAM bonus is outside this notebook's required scope.''')
code('''ROLL_NUMBER = "23L-0727"
assert ROLL_NUMBER != "REPLACE_WITH_YOUR_ROLL_NUMBER", "Enter your roll number first"
PROJECT_NAME = f"{ROLL_NUMBER}_A2_Ops"
SEED = 42
from pathlib import Path
import os
USE_GOOGLE_DRIVE = False  # Drive was full during the submitted run; download local evidence.
DRIVE_MOUNTED = False
if USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive', timeout_ms=180000)
        DRIVE_MOUNTED = os.path.ismount('/content/drive') and Path('/content/drive/MyDrive').is_dir()
        if not DRIVE_MOUNTED:
            raise RuntimeError('Drive mount was not verified')
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'Google Drive unavailable: {exc}')
        print('Continuing with LOCAL runtime storage.')
WORK_DIR = Path('/content') / PROJECT_NAME
PROJECT_DIR = ((Path('/content/drive/MyDrive') / PROJECT_NAME) if DRIVE_MOUNTED
               else (Path('/content/visionguard_results') / PROJECT_NAME))
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)
print('Project:', PROJECT_NAME)
print('Evidence folder:', PROJECT_DIR)
if not DRIVE_MOUNTED:
    print('WARNING: Local files are temporary. Download evidence before the runtime is deleted.')''')
md('''## 1.1 — Isolated environment and package/GPU evidence
A fresh Colab runtime isolates the work from your laptop. The installation below keeps Colab's existing TensorFlow/CUDA stack, installs the other libraries, and records **exact installed versions**. For a separate local virtual environment, see `docs/LOCAL_SETUP.md` in the downloaded source bundle.
If Colab asks for a runtime restart, restart and rerun the setup cells before importing TensorFlow. GPU access depends on your Colab quota.''')
code('''%pip -q install "tensorboard>=2.16,<2.22" "keras-tuner>=1.4.7,<1.5" "kagglehub>=0.3,<1" "Pillow>=10,<13" "pandas>=2.2,<3.1" "matplotlib>=3.8,<4" "scikit-learn>=1.4,<2" "dvc[s3]>=3.50,<4"
import tensorflow as tf
print('TensorFlow:', tf.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))
assert tf.config.list_physical_devices('GPU'), 'Select a GPU runtime before the full training run.' ''')
md('## Write the bundled source files\nRerunning this cell restores the source; it does not delete datasets, checkpoints or results.')
files = {}
for folder in ['visionguard', 'scripts', 'docs']:
    for p in sorted((ROOT/folder).glob('*')):
        if p.is_file() and p.name != 'build_notebook.py':
            files[str(p.relative_to(ROOT))] = p.read_text()
for name in ['requirements.txt', '.gitignore', 'README.md']:
    if (ROOT/name).exists(): files[name] = (ROOT/name).read_text()
code('BUNDLED_FILES = '+repr(files)+'''\nfor relative, contents in BUNDLED_FILES.items():
    path = WORK_DIR / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
# Persist the source alongside evidence without persisting large decoded images.
for relative, contents in BUNDLED_FILES.items():
    path = PROJECT_DIR / 'source' / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
print('Wrote', len(BUNDLED_FILES), 'source files')''')
code('''from visionguard.training import environment, seed_everything
seed_everything(SEED)
environment(PROJECT_DIR / 'environment')
from datetime import datetime, timezone
print('Project:', PROJECT_NAME, 'UTC:', datetime.now(timezone.utc).isoformat())''')
md('''## Download the assignment dataset and inspect class folders
Uses the exact Kaggle dataset in the assignment. If authentication is required, run `kagglehub.login()` privately and retry. Do not paste a Kaggle token into a saved cell. If access or terms must be accepted, do that on Kaggle first.

Check the printed directory names. The mapping below is explicit: change **keys** to the actual folder names containing images, while keeping values `ai` and `real`. Do not guess labels from filenames. The nearest mapped ancestor determines the class.''')
code('''import kagglehub
DATA_ROOT = Path(kagglehub.dataset_download('cashbowman/ai-generated-images-vs-real-images'))
print('Dataset root:', DATA_ROOT)
from collections import Counter
folder_counts = Counter(str(p.parent.relative_to(DATA_ROOT)) for p in DATA_ROOT.rglob('*') if p.is_file())
for folder, count in folder_counts.most_common(40):
    print(f'{count:7d} files  {folder}')''')
code('''# EDIT these keys if Kaggle uses other folder names; verify visually after audit.
CLASS_FOLDERS = {'AiArtData': 'ai', 'RealArt': 'real'}
assert set(CLASS_FOLDERS.values()) == {'ai', 'real'}
CLASS_MAP_JSON = __import__('json').dumps(CLASS_FOLDERS)
DATA_EVIDENCE = PROJECT_DIR / 'data_evidence'
PROCESSED = WORK_DIR / 'data' / 'processed'
import subprocess, sys
def run_module(module, *args):
    from collections import deque
    tail = deque(maxlen=40)
    command = [sys.executable, '-u', '-m', module, *map(str, args)]
    with subprocess.Popen(command, cwd=WORK_DIR, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
            tail.append(line)
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f'{module} exited with code {returncode}. Last output:\\n' + ''.join(tail))
run_module('visionguard.data', '--root', DATA_ROOT, '--out', DATA_EVIDENCE,
           '--processed', PROCESSED, '--class-map', CLASS_MAP_JSON, '--seed', SEED)''')
md('''## 1.2–1.3 — Audit and EDA evidence
The audit describes **original** files. Training copies are lossless RGB PNGs with EXIF orientation corrected, and animated images use their first frame. Exact byte duplicates and identical decoded pixels are recorded separately. Conflicting labels for identical pixels are excluded. Deduplication precedes the seeded stratified 70/15/15 split; any original folder splits are deliberately repartitioned. Near duplicates and source overlap can still remain.

For **each** figure below, write 2–3 sentences based on the actual result: describe the visible pattern, compare the classes, and explain a modeling implication. Class-dependent dimensions/compression can become shortcuts. Inspect the samples to confirm that folder labels match the images. See the report worksheet for prompts.''')
code('''import pandas as pd
from IPython.display import display, Image
for name in ['class_counts.csv', 'summary_statistics.csv']:
    print(name)
    display(pd.read_csv(DATA_EVIDENCE / name))
print((DATA_EVIDENCE / 'audit_summary.json').read_text())
print((DATA_EVIDENCE / 'split_summary.json').read_text())
for name in ['samples.png', 'width_histogram.png', 'height_histogram.png', 'size_bytes_histogram.png']:
    display(Image(filename=str(DATA_EVIDENCE / name)))''')
md('''## 1.4 — DVC toy: run the actual round trip
This creates a new isolated Git repository, commits both CSV versions, pushes both to a local-folder DVC remote, restores the older version, deletes only the toy CSV and toy cache, then pulls and verifies the original content. Capture the command output and commit hashes. Run once; if rerunning, use a new destination name.

Local storage is explicitly allowed for the DVC remote. To satisfy the separate DagsHub commit evidence, push this toy repository's `main` branch to a **separate empty DagsHub toy repository** using the instructions in `docs/DVC_GUIDE.md`. The toy commits use a clearly labeled demo author; set your own identity for the actual project.''')
code('''TOY_DIR = PROJECT_DIR / 'dvc_toy'
run_module('scripts.dvc_demo', '--destination', TOY_DIR)''')
md('''## 1.4 — Version the actual dataset
Follow `docs/DVC_GUIDE.md` before claiming this part complete. DVC-track the original Kaggle dataset, the processed images and split manifests; push bytes to persistent Drive storage or DagsHub storage. Commit the `.dvc` pointers and `.dvc/config` in your named project repository. The dataset download itself does not count as DVC versioning.

DVC pointers hold a content checksum and path, plus metadata such as byte size and file count. Git stores these small text pointers; DVC storage contains the large image bytes. Large binaries inflate Git history and are difficult to diff. Your audit SHA-256 is separate from DVC's object checksum.''')
code('''print((WORK_DIR / 'docs/DVC_GUIDE.md').read_text())''')
md("""### Run actual dataset versioning with the selected storage
Enter your Git identity, then run once. This copies the raw dataset (extra disk space required), DVC-tracks raw/processed/splits, pushes data to the selected storage and backs up committed Git metadata there. With local fallback this backup is temporary: publish to DagsHub/storage before runtime deletion. The evidence ZIP excludes full DVC caches and Git repositories. It refuses to overwrite an existing Git repository; if interrupted, follow the guide to finish the existing repository. Publishing commits to your DagsHub account remains a separate authenticated step.""")
code('''GIT_NAME = "YOUR NAME"
GIT_EMAIL = "YOUR GIT EMAIL"
if not DRIVE_MOUNTED:
    print('Local DVC storage is temporary; use the guide to publish data and commits to DagsHub.')
assert GIT_NAME != "YOUR NAME" and GIT_EMAIL != "YOUR GIT EMAIL", "Enter your Git identity"
run_module('scripts.version_dataset', '--raw', DATA_ROOT, '--splits', DATA_EVIDENCE,
           '--persistent', PROJECT_DIR, '--git-name', GIT_NAME, '--git-email', GIT_EMAIL)''')

md("""## Validate training PNGs before fitting
This checks TensorFlow decoding and audited pixel hashes for every split image. Any incompatible or incomplete PNG is losslessly regenerated from the audited original using TensorFlow's encoder, validated, then atomically replaced. No images are dropped and split lists stay unchanged. This is file integrity checking, not test prediction/evaluation. Repairs are recorded under `png_integrity`. If DVC already tracks the processed files, update its pointer and push the repaired bytes afterward.""")
code('''run_module('scripts.repair_processed', '--raw', DATA_ROOT, '--processed', PROCESSED,
           '--splits', DATA_EVIDENCE, '--report', PROJECT_DIR / 'png_integrity')''')

md('''## 2.1–2.4 — Pipeline and models
`tf.data`: read saved split paths/labels → decode normalized RGB copies → resize with padding to 224×224 → shuffle training only → batch → prefetch. Padding preserves aspect ratio but introduces borders; augmentation can perturb those borders. Moderate flip/rotation/zoom/contrast augmentation runs inside each model **only when training**. Validation/test order and pixels are deterministic.

**Scratch CNN:** three Conv→BatchNorm→ReLU→Pool blocks learn local patterns while reducing spatial resolution. Global average pooling limits parameters; a 64-unit dense layer combines features and dropout regularizes the classifier. One sigmoid returns P(AI). Binary cross entropy matches the binary output; Adam is a practical adaptive optimizer. Accuracy/AUC/precision/recall are monitored. Training precision/recall use 0.5; final metrics use the validation-selected threshold.

**Transfer:** MobileNetV2 ImageNet weights, no classifier, global average pooling, dropout and sigmoid. Embedded rescaling maps [0,255] to [-1,1], exactly the MobileNetV2 preprocessing. Train a frozen backbone first, then unfreeze only weighted non-BatchNorm layers in `block_16` and its terminal `Conv_1` projection at 1e-5 (100× below the head LR). BatchNorm stays frozen/in inference mode, and the model is recompiled after changing trainability. The standalone frozen and fine-tuned stages each save curves and trainable-layer lists.

Fixed seeds and deterministic TensorFlow ops improve reproducibility; hardware/library changes can still alter results. Early stopping, validation-loss checkpointing and plateau LR reduction run in each main stage. Best epoch is the minimum validation-loss epoch, marked on the curves.''')
md('''## 2.7 — Keras Tuner (included in the run below)
The Fashion-MNIST toy tunes units and learning rate. VisionGuard runs **8 random-search trials** over learning rate, dropout, and the number of unfrozen weighted layers restricted to the last block. Each starts from the same frozen-head checkpoint. Saved marginal mean-loss tables help discuss which setting appeared most influential; eight confounded trials cannot prove causal importance. Repeated selection against validation may overfit that split, so it must never use test feedback.

## Run training, tuning and final evaluation
Defaults: 20 scratch epochs, 10 frozen-head epochs, 15 fine-tune epochs, up to 5 epochs per tuner trial. Early stopping may finish sooner. Runtime depends on the downloaded dataset and GPU; budget for multiple training runs and keep the tab connected. With local storage, download results as soon as training finishes. If GPU memory is exhausted, lower `BATCH_SIZE` before starting.

The script resumes completed stages/checkpoints and tuner state after a pre-test interruption. Use the **same settings and data** when resuming. If trials failed, fix the cause and restart the tuner in a fresh pre-test directory; eight completed trials are required. Once `TEST_EVALUATION_STARTED.json` exists, it refuses to rerun: reuse saved test probabilities/figures and do not retune based on test results.''')
code('''TRAINING_OUT = PROJECT_DIR / 'training'
BATCH_SIZE = 32
run_module('scripts.run_training', '--root', PROCESSED, '--splits', DATA_EVIDENCE,
           '--out', TRAINING_OUT, '--batch', BATCH_SIZE, '--seed', SEED)''')
md('''## 2.5 — Evaluation and decision costs
AI is the positive class (1), real is 0. For this product a false accusation of a real photograph can damage trust and reputation, so the predeclared example cost is 2× for real→AI versus AI→real. This is a design assumption, not an empirically established business cost. The threshold minimizes `2*FP + FN` using **validation only**, with conservative ties. Transfer candidates are chosen by validation loss before opening test.

Each final model makes one test inference pass. All metrics, confusion matrix, ROC curve and error images reuse those predictions. Accuracy, precision, recall, F1 and ROC-AUC are saved for both models. ROC-AUC uses continuous probabilities. A detector trained on this dataset can fail on new generators or image sources; do not present it as proof of authenticity.''')
code('''print('Project:', PROJECT_NAME, 'UTC:', datetime.now(timezone.utc).isoformat())
display(pd.read_csv(TRAINING_OUT / 'comparison.csv'))
print((TRAINING_OUT / 'selection_before_test.json').read_text())
for stage in ['baseline', 'head', 'finetune']:
    print(stage, (TRAINING_OUT / stage / 'best_epoch.json').read_text())
    display(Image(filename=str(TRAINING_OUT / stage / 'curves.png')))
display(Image(filename=str(TRAINING_OUT / 'augmented_batch.png')))
for model in ['baseline', 'transfer']:
    for figure in ['confusion_matrix.png', 'roc.png', 'misclassified.png']:
        display(Image(filename=str(TRAINING_OUT / 'evaluation' / model / figure)))''')
md('''## 2.6 — Complete the image-by-image error analysis
Open each model's `evaluation/<model>/error_analysis.csv`. For each of the 10 displayed errors, enter a specific visible observation and a testable hypothesis. Examples to investigate include blur, painting-like texture, watermarks, unusual crops or severe resizing. These are prompts, not claims about the actual images. If fewer than 10 errors exist, report that count honestly and do not duplicate errors.

Save the completed CSVs on Drive and include the grid and hypotheses in your report. The notebook deliberately leaves these observations blank until you inspect the images.''')
code('''display(pd.read_csv(TRAINING_OUT / 'tuner_toy' / 'trials.csv'))
display(pd.read_csv(TRAINING_OUT / 'tuner' / 'trials.csv'))
print((TRAINING_OUT / 'tuner' / 'best_hyperparameters.json').read_text())
for param in ['learning_rate', 'dropout', 'unfrozen_layers']:
    print(param)
    display(pd.read_csv(TRAINING_OUT / 'tuner' / f'{param}_association.csv'))''')
md('''## Download evidence and source
The archive includes CSVs, plots, model checkpoints, tuner state, exact package versions and source. It excludes the large dataset and DVC toy cache. Download size may be substantial due to model checkpoints. If Drive mounted successfully, it is also preserved in your Google Drive project folder. Otherwise it exists only on the runtime disk: download it before disconnecting/deleting the runtime.

Use `docs/REPORT_WORKSHEET.md` to assemble Tasks 1 and 2 in your own words, adding real screenshots with project name/date context, DagsHub evidence, interpretations and observed failures. Disclose AI assistance as required by the assignment. Tasks 3–8 are outside this deliverable.''')
code('''import zipfile
archive = Path('/content') / f'{PROJECT_NAME}_tasks_1_2_evidence.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for folder in [PROJECT_DIR / 'environment', DATA_EVIDENCE, TRAINING_OUT, PROJECT_DIR / 'source', PROJECT_DIR / 'png_integrity']:
        for path in folder.rglob('*'):
            if path.is_file():
                z.write(path, path.relative_to(PROJECT_DIR))
    for name in ['result.json', 'transcript.txt']:
        path = TOY_DIR / name
        if path.exists(): z.write(path, 'dvc_toy/' + name)
print(archive)
from google.colab import files
files.download(str(archive))''')
nb.cells = cells
nbf.validate(nb)
path = ROOT/'notebooks'/'VisionGuard_Tasks_1_2.ipynb'
nbf.write(nb, path)
print(path)
