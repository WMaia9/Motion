# Research environments

The experiment host used Linux, Python 3.10.13, PyTorch 2.11.0+cu128 and CUDA 12.8.
The pins below reflect observed working environments; full installed-distribution
snapshots are supplied as JSON. They are not universal cross-platform locks, and
fresh installation was not used to rerun the scientific experiments in this
repository-preparation pass.

| Profile | Install file | Purpose |
|---|---|---|
| Main | `main.txt` | Readers, local denoisers, layers, analyses and plotting |
| Replay | `replay.txt` | Public CFM/retrieval replay; separate Transformers version |

Do not combine the two profiles or automatically upgrade an existing environment.
The original producers include compatibility handling for the native data release;
upstream vendor environment files may specify different NumPy/PyTorch versions.

From a fresh Python 3.10 environment, the recorded main dependencies can be installed
with `python -m pip install -r requirements/main.txt`. Use a separate environment
and `requirements/replay.txt` for the replay. Install only missing external code
at the locked revisions, without using its dependency installer to override these
pins. Dataset/model acquisition is described in `docs/data-and-artifacts.md`.

Check the existing environments without importing GPU models:

```bash
venv_mska/bin/python tools/repository.py env
.venv_signspark_confirmation/bin/python tools/repository.py env --profile replay
```

Lightweight repository checks and result inspection use only the standard library.

The public motion-generator audit uses an additional isolated import target:

```bash
venv_mska/bin/python -m pip install --no-deps --target external/motion_lcm_python -r requirements/motion-release-extra.txt
```

MLD uses CLIP ViT-L/14; MotionLCM-v1 uses Sentence-T5-large, including its
pooling, dense projection and normalization modules. Their inference adapters
insert the isolated target into `sys.path`. Each method runs in a separate
process because upstream packages share incompatible `mld`, `models` and
`utils` names. The original research producers and environments are preserved.
