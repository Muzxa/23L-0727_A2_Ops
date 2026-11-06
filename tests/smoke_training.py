"""CPU smoke integration; synthetic pixels only, never assignment results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras
from visionguard.training import (seed_everything, baseline, transfer, unfreeze_last_block,
    compile_model, dataset, fit_stage, augmented_batch, environment)
from visionguard.tuning import visionguard
from visionguard.evaluation import choose_threshold, evaluate_predictions


def main():
    root = Path('outputs/synthetic_smoke_verified')
    root.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    environment(root)
    rng = np.random.default_rng(42)
    frames = {}
    for split in ['train', 'val', 'test']:
        rows = []
        for i in range(8):
            label = i % 2
            filename = f'{split}_{i}.png'
            pixels = rng.integers(0, 80, (48, 64, 3), dtype='uint8') + label*150
            Image.fromarray(pixels).save(root/filename)
            rows.append(dict(path=filename, source_path=filename, label=label,
                             sha256=f'synthetic-{split}-{i}', pixel_sha256=f'synthetic-{split}-{i}'))
        frames[split] = pd.DataFrame(rows)
        frames[split].to_csv(root/f'{split}.csv', index=False)
    train = dataset(root/'train.csv', root, size=64, batch=4, training=True)
    val = dataset(root/'val.csv', root, size=64, batch=4)
    augmented_batch(train, root/'augmented_batch.png')
    model, _ = fit_stage(baseline(64), train, val, root/'baseline', epochs=1)
    p1 = model.predict(val, verbose=0)
    p2 = model.predict(val, verbose=0)
    np.testing.assert_allclose(p1, p2)
    model, _ = fit_stage(transfer(64, weights=None), train, val, root/'head', epochs=1)
    names = unfreeze_last_block(model)
    assert names == ['block_16_expand', 'block_16_depthwise', 'block_16_project', 'Conv_1'], names
    compile_model(model, 1e-5)
    model, _ = fit_stage(model, train, val, root/'fine', epochs=1)
    tuned, best = visionguard(train, val, root/'head'/'best.keras', root/'tuner',
                             size=64, epochs=1, trials=8)
    assert sum(pd.read_csv(root/'tuner'/'trials.csv').status.eq('COMPLETED')) == 8
    test = dataset(root/'test.csv', root, size=64, batch=4)
    pval = tuned.predict(val, verbose=0).ravel()
    threshold, _ = choose_threshold(frames['val'].label, pval)
    p = tuned.predict(test, verbose=0).ravel()
    result = evaluate_predictions(frames['test'], p, threshold, root, root/'evaluation')
    (root/'VERIFIED.json').write_text(json.dumps(dict(
        fixture='synthetic images; backbone randomly initialized; NOT assignment results',
        checks=['baseline fit/save/reload', 'frozen head fit', 'last-block fine-tune',
                'eight Keras Tuner trials/checkpoint reload', 'deterministic inference',
                'validation threshold', 'test metrics and figures'], best_hyperparameters=best), indent=2))
    print('PASS: synthetic training integration. NOT assignment scores.')

if __name__ == '__main__':
    main()
