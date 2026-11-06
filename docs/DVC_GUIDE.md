# Task 1.4 — DVC and DagsHub

## Toy lab already scripted

Run `python -m scripts.dvc_demo --destination outputs/dvc_toy`. The destination must be new. The script prints every command and creates:

- `repo/`: an isolated Git+DVC repository with two committed CSV versions.
- `remote/`: actual persistent DVC objects for both versions.
- `transcript.txt`: command log.
- `result.json`: verified commit hashes and recovered content.

It checks out version 1, removes the toy CSV AND the toy cache, then pulls from the remote and verifies the exact CSV content. Clearing the toy cache proves retrieval from remote, rather than a local cache checkout. It never deletes any real dataset.

To put BOTH toy commits on DagsHub, create a separate empty DagsHub repository such as `<RollNumber>_A2_Ops_DVC_Toy`, then run from the toy's `repo` directory:

```bash
git remote add origin https://dagshub.com/YOUR_USERNAME/YOUR_TOY_REPOSITORY.git
git push -u origin main
```

The toy ends on the old commit in detached HEAD, but `main` still references version 2, so pushing `main` publishes both commits. Authenticate through a credential helper or a private prompt. Do not embed tokens in remote URLs or saved notebook cells. Capture DagsHub's commit view plus the terminal output. The local toy author is explicitly a demo identity; use your own Git identity for the actual project.

## Apply DVC to the actual dataset

Create GitHub and DagsHub repositories named exactly `<RollNumber>_A2_Ops`. Connect/import the GitHub repository in DagsHub if desired. Use your real roll number; the assistant has not created external accounts or repos.

Inside the actual project folder (in Colab, `WORK_DIR`), use the following commands once. In notebook Python cells use `subprocess.run([...], check=True, cwd=WORK_DIR)` or run these as a `%%bash` cell with `cd` to the project folder. Replace placeholder paths with the printed actual paths; `DATA_ROOT`, `PROJECT_DIR`, etc. are Python variables, not automatic shell variables.

```bash
git init -b main
git config user.name 'YOUR NAME'
git config user.email 'YOUR GIT EMAIL'
dvc init
```

Copy the downloaded original dataset to `data/raw` while preserving its class folders. This may require substantial extra disk space. Keep `data/processed` from the audit. Copy `train.csv`, `val.csv`, `test.csv`, `class_mapping.json`, `split_summary.json` from the audit evidence folder into `data/splits/`. Use this Python cell in Colab:

```python
import shutil
raw_copy = WORK_DIR / 'data/raw'
if not raw_copy.exists():
    shutil.copytree(DATA_ROOT, raw_copy)
split_copy = WORK_DIR / 'data/splits'
split_copy.mkdir(parents=True, exist_ok=True)
for name in ['train.csv', 'val.csv', 'test.csv', 'class_mapping.json', 'split_summary.json']:
    shutil.copy2(DATA_EVIDENCE / name, split_copy / name)
```

Track the immutable source, derived inputs, and split identity:

```bash
dvc add data/raw data/processed data/splits
git add data/*.dvc
```

`dvc add` creates `.dvc` pointer files and ignore rules if needed. Git must include the pointers and code, while ignoring the large image directories. This project's ignore rules ignore only the three image/manifest directories, allowing `data/*.dvc` to remain visible. Stage the pointers before pushing so DVC discovers them reliably.

### Option A — persistent local-folder remote (allowed in assignment)

In Colab use a folder on your mounted Google Drive, for example the following Python cell:

```python
remote_path = PROJECT_DIR / 'dvc_remote'
remote_path.mkdir(parents=True, exist_ok=True)
subprocess.run(['dvc', 'remote', 'add', '-d', 'storage', str(remote_path)], cwd=WORK_DIR, check=True)
subprocess.run(['dvc', 'push'], cwd=WORK_DIR, check=True)
subprocess.run(['dvc', 'status', '-c'], cwd=WORK_DIR, check=True)
```

A remote under `/content` alone disappears with the runtime; Drive storage persists. Drive quota must accommodate raw and processed images plus DVC objects. This remote path is specific to your mounted filesystem; another user must mount/access the shared remote or reconfigure their own location.

### Option B — DagsHub object storage

Install `dvc[s3]` (included in Colab setup), then copy current commands from your repository's **Remote → Data → DVC** panel. The documented pattern is:

```bash
dvc remote add -d storage s3://dvc
dvc remote modify storage endpointurl https://dagshub.com/YOUR_USERNAME/YOUR_PROJECT.s3
```

Set credentials privately from a Python cell (input is hidden):

```python
from getpass import getpass
token = getpass('DagsHub token: ')
for field in ['access_key_id', 'secret_access_key']:
    subprocess.run(['dvc', 'remote', 'modify', '--local', 'storage', field, token],
                   cwd=WORK_DIR, check=True, stdout=subprocess.DEVNULL)
del token
subprocess.run(['dvc', 'push'], cwd=WORK_DIR, check=True)
```

`--local` stores secrets in ignored `.dvc/config.local`. Never print it or include it in screenshots/ZIPs. Consult the repo's current remote panel if the endpoint or credential method differs.

### Commit and publish pointers and source

```bash
git add .gitignore .dvc .dvcignore data/*.dvc visionguard scripts docs requirements.txt README.md
git commit -m 'Task 1: version audited source, processed images and split lists'
git remote add origin https://github.com/YOUR_USERNAME/YOUR_PROJECT.git
git remote add dagshub https://dagshub.com/YOUR_USERNAME/YOUR_PROJECT.git
git push -u origin main
git push dagshub main
```

Run a remote command only once; if it already exists, inspect `git remote -v` and use `git remote set-url`. If DagsHub is connected as an automatic mirror, use its sync mechanism instead of pushing to a read-only mirror. Add the completed notebook and selected report evidence deliberately; the full model/dataset archives do not belong in Git. Preserve the repo metadata by pushing it before the Colab runtime expires.

For recovery in a new clone: configure remote access, then `dvc pull`. For a prior version: `git checkout <commit> -- data/raw.dvc data/processed.dvc data/splits.dvc`, then `dvc checkout` (or `dvc pull` if not cached). This changes tracked data to that snapshot; keep uncommitted work safe first. Demonstrate destructive deletion/recovery only in the isolated toy.

## Explain in your report

A `.dvc` file is a small text manifest with the data path, checksum and metadata, typically size and file count for directories. The directory checksum references a DVC manifest of its contents. DVC's checksum/object identity is not necessarily SHA-256; the separate audit explicitly computes SHA-256. Git versions the small pointer and code; DVC caches and transfers the large bytes. Git history retains previous versions, so committing large image datasets directly bloats clones and history without useful text diffs.

Explain `git init`, `dvc init`, `dvc add`, `git commit`, `dvc remote add`, `dvc push`, `git checkout`, `dvc checkout` and `dvc pull` in your own words, tied to your actual outputs.

References: [DVC remote storage](https://dvc.org/doc/user-guide/data-management/remote-storage), [DagsHub DVC integration](https://dagshub.com/docs/integration_guide/dvc/).
