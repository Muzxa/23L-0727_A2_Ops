"""Meaningful leakage, corruption and threshold checks using synthetic fixtures."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from visionguard.data import audit, prepare, eda, infer_class, ALIASES
from visionguard.evaluation import choose_threshold


class DataTests(unittest.TestCase):
    def test_assignment_dataset_nested_folders(self):
        self.assertEqual(infer_class(Path('AiArtData/AiArtData/example.jpg'), ALIASES), 'ai')
        self.assertEqual(infer_class(Path('RealArt/example.jpg'), ALIASES), 'real')
        with self.assertRaisesRegex(ValueError, 'No class folder recognized'):
            infer_class(Path('unknown/example.jpg'), ALIASES)

    def test_audit_dedup_conflict_and_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            raw, out, processed = p/'raw', p/'out', p/'processed'
            rng = np.random.default_rng(123)
            for cls in ['AI', 'Real']:
                (raw/cls).mkdir(parents=True)
                for i in range(30):
                    Image.fromarray(rng.integers(0, 256, (20+i, 30, 3), dtype=np.uint8)).save(raw/cls/f'{i}.png')
            (raw/'AI'/'duplicate.png').write_bytes((raw/'AI'/'0.png').read_bytes())
            (raw/'Real'/'conflict.png').write_bytes((raw/'AI'/'1.png').read_bytes())
            (raw/'AI'/'broken.jpg').write_bytes(b'broken image')
            df = audit(raw, out)
            self.assertEqual(len(df), 63)
            self.assertEqual(sum(df.error.ne('')), 1)
            eda(df, raw, out)
            prepare(df, raw, processed, out)
            sets = [pd.read_csv(out/f'{s}.csv') for s in ['train', 'val', 'test']]
            self.assertEqual(sum(map(len, sets)), 59)
            for i, left in enumerate(sets):
                self.assertEqual(set(left.label), {0, 1})
                for right in sets[i+1:]:
                    self.assertFalse(set(left.pixel_sha256) & set(right.pixel_sha256))
                for row in left.itertuples():
                    with Image.open(processed/row.path) as im:
                        self.assertEqual(im.mode, 'RGB')
            self.assertEqual(len(pd.read_csv(out/'label_conflicts.csv')), 2)

    def test_threshold_matches_exhaustive_weighted_cost(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([.2, .7, .8, .6, .7, .9])
        t, table = choose_threshold(y, p)
        predicted = p >= t
        cost = 2*sum((y == 0) & predicted) + sum((y == 1) & ~predicted)
        self.assertEqual(cost, table.cost.min())
        # All-real must be a candidate, and threshold selection must be deterministic.
        self.assertTrue(np.isinf(table.threshold.iloc[0]))
        self.assertEqual(t, choose_threshold(y, p)[0])

if __name__ == '__main__':
    unittest.main()
