# Verification performed locally

Date: 6 November 2026 (Asia/Karachi).

These checks use synthetic fixtures. They are software verification, not VisionGuard dataset statistics or model performance results.

- Unit checks passed for unreadable files, byte/pixel duplicate detection, conflicting-label removal, RGB conversion and disjoint stratified splits.
- Threshold optimization matched exhaustive weighted false-positive/false-negative costs on a known fixture.
- A local Git/DVC toy created two commits, pushed both data versions, restored the older commit, removed its CSV and local DVC cache, pulled from the configured remote and verified exact original content.
- The applied dataset-versioning helper was tested on synthetic data. It tracked the raw/processed/split pointers, pushed actual DVC storage objects, committed source/metadata and created a persistent bare Git backup.
- TensorFlow/Keras integration passed on CPU: scratch CNN training/save/reload; frozen MobileNetV2 head training; last-block fine-tuning; eight completed applied Keras Tuner trials; winning checkpoint reload; deterministic repeated inference; validation threshold; final evaluation metrics, confusion matrix, ROC and error grid.
- Synthetic smoke test used randomly initialized MobileNetV2 weights to avoid treating downloaded pretraining or fake images as a real assignment experiment. The actual notebook requests ImageNet weights.
- Local smoke environment: Python 3.11.15, TensorFlow 2.21.0, Keras 3.15.1, macOS CPU. Colab's own exact environment will be recorded when you run it.
- Notebook format validation and Python-cell syntax checks passed. Source modules compile.

A missing TensorBoard dependency found by integration testing was added to requirements and the notebook installation cell. A DVC pointer-discovery/ignore ordering issue found by testing was corrected before packaging.

Not executed here: download/audit of the full Kaggle dataset, pretrained-weight training on that dataset, Fashion-MNIST toy training, Google Colab GPU execution, account creation, DagsHub/GitHub publication or final report writing. Each has runnable code or instructions, but no evidence of execution is claimed.
