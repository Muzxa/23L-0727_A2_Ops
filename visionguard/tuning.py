"""Task 2.7: Fashion-MNIST toy search and >=8 VisionGuard trials."""
import json
from pathlib import Path
import keras_tuner as kt
import numpy as np
import pandas as pd
from tensorflow import keras
from .training import transfer, compile_model, unfreeze_last_block, seed_everything


def export_trials(tuner, out):
    rows = []
    for trial in tuner.oracle.trials.values():
        rows.append(dict(trial_id=trial.trial_id, status=trial.status,
                         val_loss=trial.score, **trial.hyperparameters.values))
    df = pd.DataFrame(rows).sort_values('val_loss')
    df.to_csv(Path(out) / 'trials.csv', index=False)
    winner = tuner.oracle.get_best_trials(1)[0]
    (Path(out) / 'best_trial.json').write_text(json.dumps(dict(
        trial_id=winner.trial_id, best_epoch=int(winner.best_step)+1,
        val_loss=winner.score), indent=2))
    best = tuner.get_best_hyperparameters(1)[0].values
    (Path(out) / 'best_hyperparameters.json').write_text(json.dumps(best, indent=2))
    print(df.to_string(index=False))
    return df, best


def toy(out='outputs/tuner_toy', epochs=5, trials=8, seed=42):
    seed_everything(seed)
    (x, y), _ = keras.datasets.fashion_mnist.load_data()
    # Use a fixed stratified subset, leave the built-in test set unused.
    from sklearn.model_selection import train_test_split
    ix, iv = train_test_split(np.arange(len(y)), train_size=10000, test_size=2000,
                             stratify=y, random_state=seed)
    x = x.astype('float32')/255.
    def build(hp):
        model = keras.Sequential([keras.Input((28, 28)), keras.layers.Flatten(),
                 keras.layers.Dense(hp.Choice('units', [32, 64, 128]), activation='relu'),
                 keras.layers.Dense(10, activation='softmax')])
        model.compile(keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    tuner = kt.RandomSearch(build, objective='val_loss', max_trials=trials,
                            directory=out, project_name='search', seed=seed, overwrite=False)
    tuner.search(x[ix], y[ix], validation_data=(x[iv], y[iv]), epochs=epochs,
                 callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)], verbose=2)
    return export_trials(tuner, out)


def visionguard(train, val, frozen_checkpoint, out='outputs/tuner', size=224,
                epochs=5, trials=8, seed=42):
    if trials < 8:
        raise ValueError('Assignment requires at least eight trials')
    seed_everything(seed)
    def build(hp):
        # Every trial starts from the SAME trained frozen-head checkpoint.
        model = transfer(size, dropout=hp.Choice('dropout', [.2, .4, .5]), seed=seed, weights=None)
        model.load_weights(frozen_checkpoint)
        unfreeze_last_block(model, hp.Choice('unfrozen_layers', [1, 2, 4]))
        return compile_model(model, hp.Choice('learning_rate', [1e-5, 3e-5, 1e-4]))
    tuner = kt.RandomSearch(build, objective='val_loss', max_trials=trials,
                            directory=out, project_name='search', seed=seed, overwrite=False,
                            max_consecutive_failed_trials=2)
    tuner.search(train, validation_data=val, epochs=epochs, callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=2)], verbose=2)
    table, best = export_trials(tuner, out)
    if sum(table.status.eq('COMPLETED')) < trials:
        raise RuntimeError('Fewer than eight completed trials; inspect failed trials before submission.')
    # Descriptive marginal associations, not causal importance in a small random search.
    notes = {}
    for col in ['learning_rate', 'dropout', 'unfrozen_layers']:
        group = table[table.status.eq('COMPLETED')].groupby(col).val_loss.agg(['mean', 'count'])
        group.to_csv(Path(out) / f'{col}_association.csv')
        notes[col] = float(group['mean'].max() - group['mean'].min())
    (Path(out) / 'descriptive_effect_ranges.json').write_text(json.dumps(notes, indent=2))
    # Retain the actual winning trial checkpoint; no test data used.
    model = tuner.get_best_models(1)[0]
    model.save(Path(out) / 'best.keras')
    return model, best
