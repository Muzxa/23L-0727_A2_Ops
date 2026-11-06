"""Check every split PNG with TensorFlow and repair losslessly from audited sources."""
import argparse
import hashlib
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from visionguard.data import rgb_image, file_hash


def pixel_hash(pixels):
    h, w, channels = pixels.shape
    if channels != 3:
        raise ValueError('Expected three RGB channels')
    return hashlib.sha256(f'{w}x{h}:RGB:'.encode() + pixels.tobytes()).hexdigest()


def repair(raw, processed, splits, report):
    raw, processed, splits, report = map(Path, (raw, processed, splits, report))
    report.mkdir(parents=True, exist_ok=True)
    changes, checked = [], 0
    for split in ['train', 'val', 'test']:
        frame = pd.read_csv(splits/f'{split}.csv')
        for rec in frame.itertuples():
            dest = processed/rec.path
            reason = ''
            try:
                pixels = tf.io.decode_png(tf.io.read_file(str(dest)), channels=3).numpy()
                if pixel_hash(pixels) != rec.pixel_sha256:
                    raise ValueError('Processed pixels differ from audited pixel hash')
            except Exception as exc:
                reason = str(exc)
            if reason:
                print(f'Repairing {split}: {dest}\nReason: {reason}', flush=True)
                source = raw/rec.source_path
                if file_hash(source) != rec.sha256:
                    raise ValueError(f'Original file differs from audit: {source}')
                pixels = np.asarray(rgb_image(source), dtype=np.uint8)
                if pixel_hash(pixels) != rec.pixel_sha256:
                    raise ValueError(f'Original decoded pixels differ from audit: {source}')
                # TensorFlow's encoder writes fresh image data without inherited ICC/EXIF.
                encoded = tf.io.encode_png(pixels).numpy()
                decoded = tf.io.decode_png(encoded, channels=3).numpy()
                if pixel_hash(decoded) != rec.pixel_sha256:
                    raise ValueError(f'Re-encoded image failed lossless validation: {dest}')
                before = file_hash(dest) if dest.exists() else None
                dest.parent.mkdir(parents=True, exist_ok=True)
                temp = dest.with_name(dest.name+'.repair.tmp')
                try:
                    temp.write_bytes(encoded)
                    os.replace(temp, dest)
                finally:
                    temp.unlink(missing_ok=True)
                changes.append(dict(split=split, path=rec.path, source_path=rec.source_path,
                                    reason=reason, before_sha256=before, after_sha256=file_hash(dest)))
                (report/'png_repairs.json').write_text(json.dumps(changes, indent=2))
            checked += 1
            if checked % 100 == 0:
                print(f'Checked {checked} images; repaired {len(changes)}', flush=True)
    result = dict(checked=checked, repaired=len(changes), status='all split images TensorFlow-decodable and pixel hashes verified')
    (report/'png_integrity.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--processed', default='data/processed')
    p.add_argument('--splits', default='outputs/data')
    p.add_argument('--report', default='outputs/png_integrity')
    a = p.parse_args()
    repair(a.raw, a.processed, a.splits, a.report)
