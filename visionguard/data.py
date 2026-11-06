"""Audit original files, create RGB training copies, and write disjoint split lists."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

LABELS = {'real': 0, 'ai': 1}
ALIASES = {'aiartdata': 'ai', 'realart': 'real', 'real': 'real', 'realimages': 'real', 'realimage': 'real',
           'human': 'real', 'art': 'ai', 'ai': 'ai', 'fake': 'ai',
           'aiimages': 'ai', 'aigenerated': 'ai', 'aigeneratedimages': 'ai',
           'fakeimages': 'ai', 'synthetic': 'ai'}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tif', '.tiff', '.avif'}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def infer_class(path, mapping):
    # The nearest recognized ancestor is the class; split/outer folders are ignored.
    for part in reversed(path.parts[:-1]):
        key = ''.join(c for c in part.lower() if c.isalnum())
        if key in mapping:
            return mapping[key]
    raise ValueError(f'No class folder recognized for {path}. Supply --class-map JSON.')


def rgb_image(path):
    with Image.open(path) as im:
        im.seek(0)  # Animated files use their first frame only.
        return ImageOps.exif_transpose(im).convert('RGB')


def audit(root, out, mapping=None):
    root, out = Path(root), Path(out)
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset directory does not exist: {root}')
    # Validate recognized image paths before hashing/decoding the full dataset.
    effective_mapping = mapping or ALIASES
    for candidate in root.rglob('*'):
        if (candidate.is_file() and candidate.suffix.lower() in EXTENSIONS
                and not any(x.startswith('.') for x in candidate.relative_to(root).parts)):
            infer_class(candidate.relative_to(root), effective_mapping)
    out.mkdir(parents=True, exist_ok=True)
    mapping = mapping or ALIASES
    rows, skipped = [], []
    for p in sorted(root.rglob('*')):
        if not p.is_file() or any(x.startswith('.') for x in p.relative_to(root).parts):
            continue
        rel = p.relative_to(root)
        if p.suffix.lower() not in EXTENSIONS:
            # Try opening unknown extensions too; record sidecars separately.
            try:
                with Image.open(p):
                    pass
            except Exception:
                skipped.append(str(rel))
                continue
        cls = infer_class(rel, mapping)
        row = dict(path=str(rel), class_name=cls, label=LABELS[cls],
                   size_bytes=p.stat().st_size, sha256=file_hash(p), width=None,
                   height=None, mode=None, format=None, pixel_sha256=None, error='')
        try:
            with Image.open(p) as im:
                row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
                im.verify()
            im = rgb_image(p)
            im.load()
            row['pixel_sha256'] = hashlib.sha256(
                f'{im.width}x{im.height}:RGB:'.encode() + im.tobytes()).hexdigest()
        except Exception as e:
            row['error'] = f'{type(e).__name__}: {e}'
        rows.append(row)
    if not rows:
        raise ValueError('No labeled images found. Check DATA_ROOT and class folder mapping.')
    df = pd.DataFrame(rows)
    df.to_csv(out / 'audit.csv', index=False)
    valid = df[df.error.eq('')].copy()
    df[df.error.ne('')].to_csv(out / 'corrupt.csv', index=False)
    df[df.duplicated('sha256', keep=False)].to_csv(out / 'exact_duplicates.csv', index=False)
    valid[valid.duplicated('pixel_sha256', keep=False)].to_csv(out / 'pixel_duplicates.csv', index=False)
    stats = valid.groupby('class_name')[['width', 'height', 'size_bytes']].agg(['min', 'median', 'max'])
    stats.to_csv(out / 'summary_statistics.csv')
    counts = df.groupby('class_name').agg(files=('path', 'size'), readable=('error', lambda s: s.eq('').sum()))
    counts.to_csv(out / 'class_counts.csv')
    info = dict(total_files=len(df), readable_files=len(valid), corrupt_files=int(df.error.ne('').sum()),
                exact_duplicate_extra_files=int(df.duplicated('sha256').sum()),
                exact_duplicate_groups=int(df.loc[df.duplicated('sha256', False), 'sha256'].nunique()),
                decoded_pixel_duplicate_extra_files=int(valid.duplicated('pixel_sha256').sum()),
                skipped_nonimage_files=skipped, label_mapping=LABELS)
    (out / 'audit_summary.json').write_text(json.dumps(info, indent=2))
    print(counts.to_string())
    print(stats.to_string())
    print(json.dumps(info, indent=2))
    return df


def eda(df, root, out, seed=42):
    out = Path(out)
    valid = df[df.error.eq('')]
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), squeeze=False)
    for row, cls in enumerate(LABELS):
        sample = valid[valid.class_name.eq(cls)].sample(min(5, sum(valid.class_name.eq(cls))), random_state=seed)
        for ax in axes[row]:
            ax.axis('off')
        for ax, rec in zip(axes[row], sample.itertuples()):
            ax.imshow(rgb_image(Path(root) / rec.path))
            ax.set_title(f'{cls}: {rec.width} x {rec.height}', fontsize=9)
    fig.suptitle('Original images: deterministic sample from each class')
    fig.tight_layout()
    fig.savefig(out / 'samples.png', dpi=160)
    plt.close(fig)
    for col in ['width', 'height', 'size_bytes']:
        fig, ax = plt.subplots(figsize=(8, 4))
        values = valid[col].to_numpy(float)
        # Log-spaced bins retain visibility if there is a long tail.
        lo, hi = max(values.min(), 1), max(values.max(), values.min() + 1)
        bins = np.geomspace(lo, hi, 30)
        for cls in LABELS:
            ax.hist(valid.loc[valid.class_name.eq(cls), col], bins=bins, alpha=.5, label=cls)
        ax.set_xscale('log')
        ax.set(xlabel=col + ' (log scale)', ylabel='Image count', title=f'{col} by class')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'{col}_histogram.png', dpi=160)
        plt.close(fig)


def prepare(df, root, processed, out, seed=42):
    """Exclude corrupt/conflicting images and deduplicate BEFORE stratified splitting."""
    out, processed = Path(out), Path(processed)
    valid = df[df.error.eq('')].copy()
    conflicts = valid.groupby('pixel_sha256').label.nunique()
    bad_hashes = set(conflicts[conflicts.gt(1)].index)
    conflict_rows = valid[valid.pixel_sha256.isin(bad_hashes)]
    conflict_rows.to_csv(out / 'label_conflicts.csv', index=False)
    clean = valid[~valid.pixel_sha256.isin(bad_hashes)].drop_duplicates('pixel_sha256').copy()
    if set(clean.label) != {0, 1} or clean.groupby('label').size().min() < 10:
        raise ValueError('Need at least 10 distinct readable images per class for 70/15/15 splitting.')
    train, rest = train_test_split(clean, test_size=.30, random_state=seed, stratify=clean.label)
    val, test = train_test_split(rest, test_size=.50, random_state=seed, stratify=rest.label)
    processed.mkdir(parents=True, exist_ok=True)
    records = []
    for split, frame in [('train', train), ('val', val), ('test', test)]:
        frame = frame.sort_values('path').copy()
        frame['source_path'] = frame['path']
        names = []
        for rec in frame.itertuples():
            dest = processed / rec.class_name / f'{rec.pixel_sha256}.png'
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                im = rgb_image(Path(root) / rec.path)
                # Fresh pixels discard inherited ICC/EXIF metadata. Atomic replacement
                # prevents an interrupted conversion from leaving a partial final PNG.
                clean_im = Image.frombytes('RGB', im.size, im.tobytes())
                temporary = dest.with_name(dest.name + '.tmp')
                try:
                    clean_im.save(temporary, format='PNG')
                    os.replace(temporary, dest)
                finally:
                    temporary.unlink(missing_ok=True)
            names.append(str(dest.relative_to(processed)))
        frame['path'] = names
        frame.to_csv(out / f'{split}.csv', index=False)
        records.append(dict(split=split, files=len(frame), real=int(sum(frame.label == 0)), ai=int(sum(frame.label == 1))))
    summary = dict(seed=seed, proportions=[.7, .15, .15], splits=records,
                   excluded_label_conflicts=len(conflict_rows), retained_images=len(clean),
                   duplicates_removed=len(valid) - len(conflict_rows) - len(clean),
                   original_split_policy='Repartition globally after deduplication; do not mix with published-split results.',
                   limits='Exact decoded-pixel deduplication only; near duplicates and shared sources may remain.')
    (out / 'split_summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'class_mapping.json').write_text(json.dumps(LABELS, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', default='outputs/data')
    p.add_argument('--processed', default='data/processed')
    p.add_argument('--class-map', help='JSON mapping actual folder names to real or ai')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    mapping = None
    if a.class_map:
        raw = json.loads(a.class_map)
        if not set(raw.values()) <= set(LABELS):
            raise ValueError('Mapping values must be real or ai')
        mapping = {''.join(c for c in k.lower() if c.isalnum()): v for k, v in raw.items()}
    df = audit(a.root, a.out, mapping)
    eda(df, a.root, a.out, a.seed)
    prepare(df, a.root, a.processed, a.out, a.seed)

if __name__ == '__main__':
    main()
