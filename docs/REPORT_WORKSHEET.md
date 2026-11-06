# Tasks 1 and 2 report worksheet

This is a scaffold, not a completed experiment report. Replace every prompt with observed outputs and your explanation. Do not claim a command ran remotely or a model achieved a score until you have evidence. Use your own screenshots; code blocks alone are not evidence of execution.

Front matter: name, roll number, project name `<RollNumber>_A2_Ops`, GitHub and DagsHub links, graded commit hash, scope “Tasks 1 and 2”, and AI-assistance disclosure. Add deployment URLs only when later tasks are done. Follow your instructor's PDF page limit.

For screenshots include readable project name plus username/date/time or browser context. Caption and reference every screenshot/figure in prose. Explain what you did, why, how, and what went wrong under each sub-task ID.

| ID | Evidence produced / action still required | Explain in your own words |
|---|---|---|
| 1.1 | `environment/environment.json`, `requirements-lock.txt`; screenshot versions and GPU output; local venv commands in setup guide | Isolation, exact versions, GPU detection and actual runtime |
| 1.2 | `data_evidence/audit.csv`, `class_counts.csv`, `summary_statistics.csv`, `audit_summary.json`, `corrupt.csv`, `exact_duplicates.csv`, `pixel_duplicates.csv`, `label_conflicts.csv` | Original-file walk, SHA-256, unreadability, byte vs pixel duplicates, raw counts vs retained split counts |
| 1.3 | `samples.png`, `width_histogram.png`, `height_histogram.png`, `size_bytes_histogram.png` | Each figure needs a caption and 2–3 sentences grounded in actual observations |
| 1.4 | Toy `transcript.txt`, `result.json`, `.dvc` pointer; actual dataset pointers; DVC push/status; DagsHub two toy commits and actual data commit screenshots | Git vs DVC storage, version change/rollback, remote recovery, pointer fields, why large bytes stay out of Git |
| 2.1 | Split manifests, `split_summary.json`, `training/augmented_batch.png` | Every tf.data stage, 224px compute/detail tradeoff, aspect ratio and borders, augmentation and leakage controls |
| 2.2 | `training/baseline/model_summary.txt`, `layers.csv` (include table in report), screenshot model summary | Purpose of each layer, sigmoid/P(AI), binary cross entropy, Adam, metrics and regularization |
| 2.3 | `head/curves.png`, `finetune/curves.png`, each `trainable_layers.json`; `transfer_validation_selection.csv` | ImageNet transfer, [-1,1] preprocessing, head stage, last block, frozen BatchNorm, recompile, 100× smaller LR |
| 2.4 | All main-stage `curves.png`, `best_epoch.json`, `epochs.csv`, `best.keras`, `config.json` | Seed, determinism limits, early stopping, minimum validation-loss epoch, checkpointing and LR reduction |
| 2.5 | `training/comparison.csv`, each `evaluation/<model>/metrics.json`, `confusion_matrix.png`, `roc.png`, validation threshold CSVs, `selection_before_test.json` | All six requested metrics, positive label, threshold cost assumption and actual selected thresholds, one test pass per model |
| 2.6 | Each model's `misclassified.png`, `error_analysis.csv` | Fill ten image-specific observations/hypotheses; if fewer errors exist, state that honestly |
| 2.7 | `tuner_toy/trials.csv`, `tuner/trials.csv`, `best_hyperparameters.json`, three `*_association.csv` files | Random search, search spaces, ≥8 completed applied trials, which setting appeared to matter, uncertainty/confounding and validation overfit |

## EDA writing prompts — do not treat these as observations

For the sample grid, describe subjects and visible variation per class. Identify any visible source artifact only after looking at the actual images, and state whether labels look plausible. Explain why a few samples cannot establish the dataset's full representativeness.

For each width/height/size histogram, describe the central range, spread and outliers using the actual table/figure. Compare class distributions and note the logarithmic x-axis. Explain how source resolution, compression or borders might become shortcuts rather than evidence of generation.

## Model rationale to adapt

The 224×224 choice is compatible with MobileNetV2 pretrained inputs and limits GPU memory. It can erase high-frequency forensic details. Letterboxing avoids stretching subjects but may introduce class-correlated borders; this is a limitation to investigate.

A convolution learns local features with shared weights. BatchNorm stabilizes activations. ReLU adds nonlinear behavior. Pooling reduces spatial size and computation. Global average pooling avoids a large flatten/dense stack; a small dense layer combines learned channels. Dropout regularizes, and a sigmoid provides a binary score. Scores need not be calibrated probabilities of authenticity.

Pretrained convolution features provide a useful starting point, but ImageNet classification and AI-image detection are different tasks. Frozen-head training preserves pretrained features initially. Last-block tuning allows adaptation at a small learning rate while protecting earlier features. Model selection and all thresholds must be locked before test inference.

## Evaluation discussion to complete after the run

Record the selected transfer candidate, each validation threshold, each best epoch, and the actual test metric table. Describe the confusion matrix in terms of real photos falsely accused and AI images missed. The default product judgment penalizes false accusations 2×; justify it for this use case and acknowledge it is an assumption. Discuss tradeoffs in precision/recall rather than selecting a winner from accuracy alone. If you name a preferred model, distinguish the pre-test validation choice from the retrospective test comparison.

## Error worksheet

Fill `visual_observation` with a visible fact (for example, “strong JPEG blocks around edges” only if actually present). Fill `hypothesis_to_verify` with a possible explanation and a check (“resize/compression may suppress artifacts; test on a separately held-out compression sweep”). A hypothesis is not a proven cause. Do not label an unseen image as blurry, watermarked or painting-like.

## Tuner discussion

Record the toy's best units/LR, all eight applied completed trials, and winning LR/dropout/unfrozen-layer count. Compare marginal means and individual trials. The widest marginal mean-loss range is an exploratory association, not reliable causal importance: trials vary multiple settings, the sample is small and training is stochastic. Explain that picking the best of many validation scores can select noise, making validation optimistic. A final untouched test checks generalization only once; further model iteration would require fresh holdout data.

## What went wrong

Record actual issues: authentication, folder mapping, corrupt inputs, GPU quota, out-of-memory, interrupted training, or unstable fine-tuning only if they occurred. Explain your fix and evidence. Do not invent a troubleshooting story to fill the section.

## Submission checks

Verify class labels and zero overlap of retained pixel hashes across splits. Verify eight applied tuner trials have `COMPLETED` status. Check all figures are readable, both transfer stages appear, ten available errors have individual hypotheses, and exact environment versions are included. Verify your DagsHub commits/storage and that tokens are absent from repository history and screenshots. Disclose AI assistance accurately. Export your authored report to the required `<RollNumber>_VisionGuard.pdf` filename once real evidence is inserted.
