"""Exclude TensorFlow-undecodable PNG rows without deleting original images."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import pandas as pd
import tensorflow as tf


def drop_invalid(processed, splits, report, previous_training=None):
    processed, splits, report = map(Path, (processed, splits, report))
    if previous_training and (Path(previous_training)/'TEST_EVALUATION_STARTED.json').exists():
        raise RuntimeError('Test evaluation was already opened; do not change this test split.')
    frames, failures = {}, []
    for split in ['train', 'val', 'test']:
        frame = pd.read_csv(splits/f'{split}.csv')
        keep = []
        for rec in frame.itertuples():
            try:
                tf.io.decode_png(tf.io.read_file(str(processed/rec.path)), channels=3).numpy()
                keep.append(True)
            except (tf.errors.OpError, ValueError, OSError) as exc:
                keep.append(False)
                failures.append(dict(split=split, path=rec.path, label=rec.label, error=str(exc)))
        frames[split] = frame.loc[keep].copy()
        if set(frames[split].label) != {0, 1}:
            raise RuntimeError(f'{split} would lose an entire class; split lists left unchanged.')
        print(f'{split}: kept {sum(keep)}, excluded {len(keep)-sum(keep)}', flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = report/stamp
    backup.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(failures, columns=['split','path','label','error']).to_csv(backup/'excluded_images.csv', index=False)
    for split, frame in frames.items():
        dest = splits/f'{split}.csv'
        shutil.copy2(dest, backup/dest.name)
        if failures:
            temp = dest.with_suffix('.csv.tmp')
            frame.to_csv(temp, index=False)
            os.replace(temp, dest)
    result = dict(excluded=len(failures), backup=str(backup),
                  splits={s:dict(files=len(f), ai=int(sum(f.label==1)), real=int(sum(f.label==0))) for s,f in frames.items()},
                  note='File integrity exclusions before model test evaluation. Original files were not deleted.')
    summary_path = splits/'split_summary.json'
    if failures and summary_path.exists():
        shutil.copy2(summary_path, backup/summary_path.name)
        summary = json.loads(summary_path.read_text())
        summary['png_decode_filter'] = result
        summary_path.write_text(json.dumps(summary, indent=2))
    (backup/'filter_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result
