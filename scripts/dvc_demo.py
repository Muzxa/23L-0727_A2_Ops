"""Run an isolated two-version toy Git/DVC round trip; no user repo is modified."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def demo(destination):
    dest = Path(destination).resolve()
    if dest.exists():
        raise FileExistsError(f'Use a NEW directory; refusing to overwrite {dest}')
    repo, remote = dest/'repo', dest/'remote'
    repo.mkdir(parents=True)
    remote.mkdir()
    transcript = []
    def run(*args):
        shown = ' '.join(args)
        print(f'[{datetime.now(timezone.utc).isoformat()}] {repo.name}$ {shown}', flush=True)
        result = subprocess.run(args, cwd=repo, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False,
                                env={**os.environ, 'DVC_NO_ANALYTICS': 'true',
                                     'DVC_SITE_CACHE_DIR': str(dest/'site-cache'),
                                     'XDG_CONFIG_HOME': str(dest/'config'),
                                     'XDG_CACHE_HOME': str(dest/'cache')})
        print(result.stdout, flush=True)
        transcript.append('$ '+shown+'\n'+result.stdout)
        (dest/'transcript.txt').write_text('\n'.join(transcript))
        result.check_returncode()
        return result.stdout.strip()
    def git(*args):
        return run('git', '-c', 'user.name=VisionGuard Toy Demo', '-c',
                   'user.email=visionguard-demo@example.invalid', *args)
    def dvc(*args):
        return run(sys.executable, '-m', 'dvc', *args)
    git('init', '-b', 'main')
    dvc('init')
    dvc('remote', 'add', '-d', 'local', str(remote))
    csv = repo/'tiny.csv'
    v1 = 'id,label\n1,real\n2,ai\n'
    csv.write_text(v1)
    dvc('add', 'tiny.csv')
    git('add', '.dvc', '.dvcignore', '.gitignore', 'tiny.csv.dvc')
    git('commit', '-m', 'Version 1: two labeled rows')
    first = git('rev-parse', 'HEAD')
    dvc('push')
    csv.write_text(v1+'3,real\n')
    dvc('add', 'tiny.csv')
    git('add', 'tiny.csv.dvc')
    git('commit', '-m', 'Version 2: add third row')
    second = git('rev-parse', 'HEAD')
    dvc('push')
    git('log', '--oneline', '--all')
    git('checkout', first)
    dvc('checkout')
    assert csv.read_text() == v1
    print('Restored older version:\n'+csv.read_text(), flush=True)
    # Only this freshly created toy workspace/cache is removed, after both pushes.
    csv.unlink()
    shutil.rmtree(repo/'.dvc'/'cache')
    dvc('pull')
    assert csv.read_text() == v1
    print('PASS: recovered version 1 from remote after deleting local CSV AND cache.', flush=True)
    result = dict(first_commit=first, second_commit=second, restored_commit=first,
                  remote=str(remote), csv=csv.read_text(), verified=True,
                  note='Local round trip completed. Push both commits to your DagsHub toy repository separately.')
    (dest/'result.json').write_text(json.dumps(result, indent=2))
    (dest/'transcript.txt').write_text('\n'.join(transcript))
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--destination', default='outputs/dvc_toy')
    print(json.dumps(demo(p.parse_args().destination), indent=2))
