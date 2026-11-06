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

## What still requires your execution

No original dataset, trained weights or user credentials were provided. Therefore **no dataset statistics, accuracy scores, training screenshots or DagsHub links are claimed**. Run the notebook on your data/GPU, verify folder labels, create your named GitHub/DagsHub projects, run dataset DVC versioning, push commits, capture genuine screenshots, interpret EDA, and fill the ten-error observation worksheet. The optional Grad-CAM bonus and Tasks 3–8 are not included. The assignment's overview asks for MLflow tracking of every experiment; this Tasks 1–2 package saves local evidence but does not implement Task 3's MLflow instrumentation. If your instructor expects remote tracking during the initial training, add it before your graded run rather than claiming it afterward.

A final report cannot honestly be completed before the dataset and training results exist. Use the worksheet to assemble your report after execution. Check any instructor-specific page limit.

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

AI-assistance disclosure suggestion: “I used OpenAI Codex to prepare the audit, data pipeline, training/evaluation scripts, Colab notebook and report checklist. I ran the experiments, inspected the outputs, and wrote/verified the reported interpretations.” Adjust this to describe what you actually did.
