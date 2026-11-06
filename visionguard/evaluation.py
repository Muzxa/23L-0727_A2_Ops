"""Validation-only threshold choice, a single held-out test pass, and error evidence."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from .data import rgb_image


def choose_threshold(y, probabilities, fp_cost=2., fn_cost=1.):
    """Minimize 2*false accusations + missed AI on validation; AI is positive."""
    y, probabilities = np.asarray(y).ravel(), np.asarray(probabilities).ravel()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Both classes are required to select a threshold')
    # Every attainable classification boundary, including all-real and all-AI.
    fpr, tpr, thresholds = roc_curve(y, probabilities, drop_intermediate=False)
    fp = np.rint(fpr * sum(y == 0)).astype(int)
    fn = np.rint((1-tpr) * sum(y == 1)).astype(int)
    costs = fp_cost*fp + fn_cost*fn
    tied = np.flatnonzero(costs == costs.min())
    # Prefer the conservative higher threshold for an exact cost tie.
    i = tied[0]
    t = float(thresholds[i])
    if not np.isfinite(t):
        t = float(np.nextafter(probabilities.max(), np.inf))
    table = pd.DataFrame(dict(threshold=thresholds, false_positive=fp, false_negative=fn, cost=costs))
    return t, table


def metrics(y, p, threshold):
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(threshold=threshold, accuracy=float(accuracy_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)), roc_auc=float(roc_auc_score(y, p)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def error_evidence(frame, p, threshold, root, out):
    out = Path(out)
    wrong = frame.copy()
    wrong['p_ai'] = p
    wrong['predicted_label'] = (p >= threshold).astype(int)
    wrong = wrong[wrong.label.ne(wrong.predicted_label)].head(10).copy()
    # Hypotheses must be written after inspecting the actual errors, not invented.
    wrong['visual_observation'] = ''
    wrong['hypothesis_to_verify'] = ''
    wrong.to_csv(out / 'error_analysis.csv', index=False)
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    for ax in axes.flat:
        ax.axis('off')
    for ax, rec in zip(axes.flat, wrong.itertuples()):
        ax.imshow(rgb_image(Path(root) / rec.path))
        ax.set_title(f'True: {"AI" if rec.label else "Real"}\nPred: {"AI" if rec.predicted_label else "Real"}; P(AI)={rec.p_ai:.3f}', fontsize=9)
    fig.suptitle(f'Test errors: {len(wrong)} shown (up to 10 available)')
    fig.tight_layout()
    fig.savefig(out / 'misclassified.png', dpi=160)
    plt.close(fig)
    return len(wrong)


def evaluate_predictions(frame, p, threshold, root, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    y = frame.label.to_numpy()
    result = metrics(y, p, threshold)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    predictions = frame[['path', 'source_path', 'label', 'sha256', 'pixel_sha256']].copy()
    predictions['p_ai'] = p
    predictions['predicted_label'] = (p >= threshold).astype(int)
    predictions.to_csv(out / 'predictions.csv', index=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y, p >= threshold, labels=[0, 1])
    ax.imshow(cm, cmap='Blues')
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha='center', va='center')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'],
           xlabel='Predicted', ylabel='True', title='Confusion matrix (AI is positive)')
    fig.tight_layout()
    fig.savefig(out / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f'AUC={result["roc_auc"]:.4f}')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Test ROC curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'roc.png', dpi=160)
    plt.close(fig)
    result['errors_shown'] = error_evidence(frame, p, threshold, root, out)
    return result
