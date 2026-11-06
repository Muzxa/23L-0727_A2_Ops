"""Apply real dataset DVC versioning in Colab; keep bytes and Git backup on Drive."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--splits', required=True)
    p.add_argument('--persistent', required=True)
    p.add_argument('--git-name', required=True)
    p.add_argument('--git-email', required=True)
    a = p.parse_args()
    root, persistent = Path.cwd(), Path(a.persistent).resolve()
    persistent.mkdir(parents=True, exist_ok=True)
    if (root/'.git').exists():
        raise RuntimeError('Repository already exists. Use docs/DVC_GUIDE.md to inspect and finish versioning safely.')
    env = {**os.environ, 'DVC_NO_ANALYTICS': 'true',
           'DVC_SITE_CACHE_DIR': str(root/'outputs'/'dvc-site-cache')}
    def run(*args):
        print('$ '+' '.join(map(str, args)), flush=True)
        subprocess.run(list(map(str, args)), check=True, env=env)
    run('git', 'init', '-b', 'main')
    run('git', 'config', 'user.name', a.git_name)
    run('git', 'config', 'user.email', a.git_email)
    run(sys.executable, '-m', 'dvc', 'init')
    # Persist this writable cache location so subsequent dvc commands work locally too.
    run(sys.executable, '-m', 'dvc', 'config', '--local', 'core.site_cache_dir', root/'outputs'/'dvc-site-cache')
    raw_dest = root/'data'/'raw'
    if Path(a.raw).resolve() != raw_dest.resolve():
        if raw_dest.exists():
            raise FileExistsError('data/raw already exists; inspect rather than overwrite it')
        shutil.copytree(a.raw, raw_dest)
    if not (root/'data'/'processed').is_dir():
        raise FileNotFoundError('Run the audit/preparation cell first to create data/processed')
    split_dest = root/'data'/'splits'
    split_dest.mkdir(parents=True, exist_ok=True)
    for name in ['train.csv', 'val.csv', 'test.csv', 'class_mapping.json', 'split_summary.json']:
        shutil.copy2(Path(a.splits)/name, split_dest/name)
    (root/'data'/'.gitignore').write_text('/raw\n/processed\n/splits\n')
    run(sys.executable, '-m', 'dvc', 'add', 'data/raw', 'data/processed', 'data/splits')
    run('git', 'add', 'data/raw.dvc', 'data/processed.dvc', 'data/splits.dvc', 'data/.gitignore')
    storage = persistent/'dvc_remote'
    storage.mkdir(exist_ok=True)
    run(sys.executable, '-m', 'dvc', 'remote', 'add', '-d', 'storage', storage)
    run(sys.executable, '-m', 'dvc', 'push')
    run(sys.executable, '-m', 'dvc', 'status', '-c')
    run('git', 'add', '.gitignore', '.dvc', '.dvcignore', 'data/raw.dvc', 'data/processed.dvc',
        'data/splits.dvc', 'data/.gitignore', 'visionguard', 'scripts', 'docs', 'requirements.txt', 'README.md')
    run('git', 'commit', '-m', 'Task 1: version original and processed dataset plus audited split lists')
    # A bare Git backup keeps the committed metadata after the Colab runtime is recycled.
    backup = persistent/'git_backup.git'
    if backup.exists():
        raise FileExistsError('Git backup already exists; inspect it before changing it')
    run('git', 'clone', '--bare', root, backup)
    run('git', 'rev-parse', 'HEAD')
    print('DVC dataset push and persistent Git backup complete. Publish commits to DagsHub per DVC_GUIDE.md.', flush=True)

if __name__ == '__main__':
    main()
