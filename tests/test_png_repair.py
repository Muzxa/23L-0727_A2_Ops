"""Run explicitly in the TensorFlow environment; synthetic repair integration."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image

# Keep dependency-light data checks usable without installing TensorFlow locally.
try:
    import tensorflow
except ImportError:
    tensorflow = None

@unittest.skipIf(tensorflow is None, 'TensorFlow environment required')
class RepairTests(unittest.TestCase):
    def test_truncated_png_repaired_losslessly_without_split_change(self):
        from scripts.repair_processed import repair, pixel_hash
        from visionguard.data import file_hash
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ['raw','processed','splits','report']:
                (root/name).mkdir()
            rng=np.random.default_rng(3)
            for split in ['train','val','test']:
                pixels=rng.integers(0,256,(17,23,3),dtype=np.uint8)
                filename=split+'.png'
                Image.fromarray(pixels).save(root/'raw'/filename)
                (root/'processed'/filename).write_bytes(b'incomplete PNG')
                pd.DataFrame([dict(path=filename,source_path=filename,label=0,
                    sha256=file_hash(root/'raw'/filename),pixel_sha256=pixel_hash(pixels))]).to_csv(root/'splits'/f'{split}.csv',index=False)
            before={p.name:p.read_bytes() for p in (root/'splits').glob('*.csv')}
            result=repair(root/'raw',root/'processed',root/'splits',root/'report')
            self.assertEqual(result['repaired'],3)
            self.assertEqual(before,{p.name:p.read_bytes() for p in (root/'splits').glob('*.csv')})
            result=repair(root/'raw',root/'processed',root/'splits',root/'report2')
            self.assertEqual(result['repaired'],0)

if __name__=='__main__': unittest.main()
