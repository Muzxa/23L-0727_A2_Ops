"""Train, tune, lock selection on validation, then evaluate each model on test once."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow import keras
from visionguard.training import (seed_everything, environment, dataset, baseline, transfer,
    compile_model, unfreeze_last_block, fit_stage, augmented_batch, layer_evidence)
from visionguard.tuning import toy, visionguard
from visionguard.evaluation import choose_threshold, metrics, evaluate_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/processed')
    parser.add_argument('--splits', default='outputs/data')
    parser.add_argument('--out', default='outputs/training')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--baseline-epochs', type=int, default=20)
    parser.add_argument('--head-epochs', type=int, default=10)
    parser.add_argument('--fine-epochs', type=int, default=15)
    parser.add_argument('--tuner-epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    a = parser.parse_args()
    out, splits = Path(a.out), Path(a.splits)
    out.mkdir(parents=True, exist_ok=True)
    # Never silently rerun a test evaluation or overwrite final test evidence.
    marker = out / 'TEST_EVALUATION_STARTED.json'
    if marker.exists():
        raise RuntimeError('Final test was already opened here. Read saved results; do not retune against this test.')
    config_path = out/'config.json'
    current_config = vars(a)
    if config_path.exists() and json.loads(config_path.read_text()) != current_config:
        raise RuntimeError('Resume settings changed. Use the original settings or a new pre-test run directory.')
    split_identity = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (splits/'train.csv', splits/'val.csv', splits/'test.csv')}
    identity_path = out/'split_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != split_identity:
        raise RuntimeError('Split manifests changed; cannot reuse checkpoints or tuner state.')
    identity_path.write_text(json.dumps(split_identity, indent=2))
    seed_everything(a.seed)
    environment(out)
    (out / 'config.json').write_text(json.dumps(vars(a), indent=2))
    train = dataset(splits/'train.csv', a.root, a.size, a.batch, True, a.seed)
    val = dataset(splits/'val.csv', a.root, a.size, a.batch)
    augmented_batch(train, out/'augmented_batch.png', a.seed)
    stages = {}
    def stage(name, builder, epochs):
        path = out/name/'best.keras'
        if path.exists() and (out/name/'best_epoch.json').exists():
            return keras.models.load_model(path), json.loads((out/name/'best_epoch.json').read_text())
        return fit_stage(builder(), train, val, out/name, epochs)
    scratch, stages['baseline'] = stage('baseline', lambda: baseline(a.size, a.seed), a.baseline_epochs)
    del scratch
    keras.backend.clear_session()
    seed_everything(a.seed)
    head, stages['head'] = stage('head', lambda: transfer(a.size, seed=a.seed), a.head_epochs)
    def fine_builder():
        model = keras.models.load_model(out/'head'/'best.keras')
        unfreeze_last_block(model)
        return compile_model(model, 1e-5)
    fine, stages['finetune'] = stage('finetune', fine_builder, a.fine_epochs)
    del head, fine
    keras.backend.clear_session()
    toy(out/'tuner_toy', epochs=a.tuner_epochs, seed=a.seed)
    tuned, hp = visionguard(train, val, out/'head'/'best.keras', out/'tuner',
                           a.size, a.tuner_epochs, 8, a.seed)
    layer_evidence(tuned, out/'tuner')
    del tuned
    keras.backend.clear_session()
    # Compare transfer candidates on validation loss, never on held-out test.
    candidates = {'frozen_head': out/'head'/'best.keras',
                  'last_block': out/'finetune'/'best.keras', 'tuned': out/'tuner'/'best.keras'}
    scores = []
    for name, path in candidates.items():
        model = keras.models.load_model(path)
        score = model.evaluate(val, verbose=0, return_dict=True)
        scores.append(dict(candidate=name, path=str(path), **score))
        del model
    table = pd.DataFrame(scores).sort_values('loss')
    table.to_csv(out/'transfer_validation_selection.csv', index=False)
    chosen = table.iloc[0]
    final_models = {'baseline': out/'baseline'/'best.keras', 'transfer': Path(chosen['path'])}
    thresholds = {}
    val_y = pd.read_csv(splits/'val.csv').label.to_numpy()
    for name, path in final_models.items():
        model = keras.models.load_model(path)
        p = model.predict(val, verbose=0).ravel()
        t, choices = choose_threshold(val_y, p)
        thresholds[name] = t
        choices.to_csv(out/f'{name}_validation_thresholds.csv', index=False)
        (out/f'{name}_validation_metrics.json').write_text(json.dumps(metrics(val_y, p, t), indent=2))
        del model
    selection = dict(transfer_candidate=chosen['candidate'], thresholds=thresholds,
        threshold_rule='Minimize 2*FP+FN on validation. AI=1. False accusation of a real photo costs twice a missed AI image.',
        stages=stages, tuned_hyperparameters=hp,
        split_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in splits.glob('*.csv')})
    (out/'selection_before_test.json').write_text(json.dumps(selection, indent=2))
    # Exclusive create marks the test as opened even if a later plot/save fails.
    with marker.open('x') as f:
        json.dump(selection, f, indent=2)
    test_frame = pd.read_csv(splits/'test.csv')
    test = dataset(splits/'test.csv', a.root, a.size, a.batch)
    results = []
    for name, path in final_models.items():
        model = keras.models.load_model(path)
        # Exactly one test inference pass per final model; reuse predictions for all figures.
        p = model.predict(test, verbose=0).ravel()
        dest = out/'evaluation'/name
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest/'test_probabilities.npy', p)
        result = evaluate_predictions(test_frame, p, thresholds[name], a.root, dest)
        results.append(dict(model=name, **result))
        del model
    comparison = pd.DataFrame(results)
    comparison.to_csv(out/'comparison.csv', index=False)
    print(comparison.to_string(index=False))
    (out/'COMPLETE.json').write_text(json.dumps(dict(status='complete', models=list(final_models)), indent=2))

if __name__ == '__main__':
    main()
