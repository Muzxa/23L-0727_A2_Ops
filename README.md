# VisionGuard — Assignment Tasks 1 and 2

Start with **[notebooks/VisionGuard_Tasks_1_2.ipynb](notebooks/VisionGuard_Tasks_1_2.ipynb)**. It is self-contained: upload only this notebook to [Google Colab](https://colab.research.google.com/), select a GPU, enter your roll number, and run in order. No source ZIP upload is needed. Confirm the actual dataset folder names before assigning labels.

The notebook downloads the assignment's Kaggle dataset, audits it, produces EDA, creates deduplicated 70/15/15 splits, trains a scratch CNN and a two-stage MobileNetV2, runs the Fashion-MNIST tuner lab and eight VisionGuard trials, selects thresholds on validation, then evaluates the two final models on test once. Evidence and checkpoints persist to Google Drive when mounting succeeds. If mounting fails, the notebook continues on local runtime storage; download results before runtime deletion. Set `USE_GOOGLE_DRIVE = False` to skip mounting. It includes a runnable DVC toy and instructions to version the actual dataset and push commits to DagsHub.

## Files

- `visionguard/data.py`: original-file metadata/SHA-256 audit, corruption and duplicate detection, figures and split manifests.
- `visionguard/training.py`: tf.data, training-only augmentation, models, callbacks and layer evidence.
- `visionguard/tuning.py`: toy and applied Keras Tuner random searches.
- `visionguard/evaluation.py`: validation cost threshold, metrics, confusion matrix, ROC and error grids.
- `scripts/run_training.py`: main orchestration and single-use test guard.
- `scripts/repair_processed.py`: verifies TensorFlow PNG decoding/pixel hashes and losslessly repairs incompatible or partial files from audited originals.
- `scripts/version_dataset.py`: applies dataset DVC tracking, uploads objects and backs up Git on persistent storage.
- `scripts/dvc_demo.py`: real two-commit Git/DVC push/delete/cache-clear/pull verification.
- `docs/DVC_GUIDE.md`: dataset versioning and DagsHub publishing steps.
- `docs/REPORT_WORKSHEET.md`: exact evidence mapping and writing prompts.
- `docs/LOCAL_SETUP.md`: isolated Python setup and CLI commands.
- `tests/test_data.py`: synthetic-fixture checks for corruption, leakage and threshold optimization.

## Completed Tasks 1 and 2 - 6 November 2026

The full assignment dataset was audited and the actual experiment ran on Colab's Tesla T4 (Python 3.13.16, TensorFlow 2.21.0, Keras 3.13.2). There are 973 readable original images (538 AI, 435 real), zero unreadable candidates, and one duplicate removed. The disjoint stratified splits contain 680 train, 146 validation, and 146 test images. One incompatible processed PNG was repaired losslessly and all 972 retained pixel hashes were verified.

| Final model | Accuracy | Precision | Recall | F1 | ROC-AUC | Validation threshold |
|---|---:|---:|---:|---:|---:|---:|
| Scratch CNN | 0.4932 | 0.6667 | 0.1500 | 0.2449 | 0.6672 | 0.651028 |
| Tuned MobileNetV2 | 0.7055 | 0.7761 | 0.6500 | 0.7075 | 0.8377 | 0.680687 |

Eight Fashion-MNIST toy trials and eight applied trials completed. The applied winner used LR 0.0001, dropout 0.2, and two unfrozen weighted layers. Validation loss chose the tuned transfer model before test access. Each final model made one held-out inference pass; all figures and metrics reuse the saved predictions. Ten errors per model have image-specific observations and hypotheses in the evidence bundle.

- GitHub: https://github.com/Muzxa/23L-0727_A2_Ops
- DagsHub: https://dagshub.com/Muzxa/23L-0727_A2_Ops
- DVC toy commits: the `dvc-toy` branch on both platforms.
- Report: `submission/23L-0727_VisionGuard.pdf`.
- Local full evidence: `outputs/23L-0727_A2_Ops/colab/`.
- Local DVC object storage: `outputs/23L-0727_A2_Ops/versioned_storage/dvc_remote/`.

The DVC storage is a real local remote, an option permitted by Task 1.4. Data bytes are not uploaded to DagsHub Storage; DagsHub mirrors the Git source, pointers, and toy commits. A new machine needs the local remote or must recreate the dataset from Kaggle. The original dataset has artwork in its real folder, so labels must not be presented as verified photographic authenticity. Near-duplicate and common-source leakage remain limitations.

Google Drive quota prevented notebook saves. The completed run used runtime-local outputs, which were downloaded and verified locally. Deterministic resized inputs are cached before per-epoch shuffling to avoid repeated full-resolution decoding. Augmentation remains inside the model and is never cached.

Tasks 3-8, cloud/API deployment, remote MLflow tracking, and the optional Grad-CAM bonus are outside this Tasks 1-2 submission. Saved local experiments are not claimed as MLflow runs.

## Design choices

AI=1, real=0. Training inputs are RGB [0,255]; preprocessing is embedded in the saved model. Resize with padding preserves aspect ratio. Validation/test are never augmented. Byte hashes audit the source files; decoded-pixel hashes detect identical images saved with different metadata. Conflicting labels for identical pixels are excluded. Near-duplicate/source leakage remains a limitation; no source IDs are supplied here. Existing dataset folder splits are deliberately replaced with a global deduplicated split, so results should not be compared as if they used an official benchmark split.

The primary transfer experiment freezes the head's backbone, then unfreezes only the last MobileNetV2 block plus terminal convolution. Tuning varies the number of weighted layers within that region. BatchNorm statistics remain frozen. Transfer candidate selection uses validation loss; threshold selection minimizes the predeclared validation cost `2*FP+FN`. Real→AI is assumed twice as costly because false accusations can damage trust. This is an explicit product judgment, not a measured business fact.

The saved test guard blocks a second final evaluation in the same run directory. If interrupted after test probabilities are saved, reconstruct figures from them; do not train a new model against the exposed test. Resume only with unchanged data/settings. Checkpoints are not a substitute for a full experiment tracker.

## Authoritative references

- [Assignment dataset on Kaggle](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images)
- [TensorFlow transfer learning guide](https://www.tensorflow.org/guide/keras/transfer_learning)
- [Keras Tuner getting started](https://keras.io/keras_tuner/getting_started/)
- [DVC remote storage](https://dvc.org/doc/user-guide/data-management/remote-storage)
- [DagsHub DVC integration](https://dagshub.com/docs/integration_guide/dvc/)

AI-assistance disclosure: “OpenAI Codex helped prepare and repair the code, operated the Colab experiment, inspected outputs, and drafted the report. I am responsible for reviewing and understanding the submitted work.”
