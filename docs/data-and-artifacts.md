# Data and local artifacts

## What belongs in Git

Scientific Python source, pinned external revisions, aggregate metadata, LaTeX,
figures and reviewed PDFs. `results/index.json` connects each exported metadata
file with its original full-archive manifest.

## What stays local

| Path in the original workspace | Purpose |
|---|---|
| `annotations/`, `features/`, `evaluation/`, `models/` | Original PHOENIX dataset resources |
| `keypoints/`, `mp_models/` | Pose extraction data and resources |
| `reports/<study>/` | Full frozen inputs, fitted checkpoints, arrays, decoded strings and source closures |
| `external/` | External Git checkouts at pinned revisions |
| `venv_mska/`, `.venv_signspark_confirmation/` | Separate original Python environments |
| `paper/cvpr2027_*/`, `paper/archive/` | Earlier manuscript versions, compiler and retired manuscript files |
| `maintenance/` | Local inventories, integrity checks, archived editorial scripts and internal review notes (`docs_history/`) |
| `.build/` | Temporary manuscript compilation output |

The preparation pass moved only one-off editorial/cleanup scripts into
`maintenance/repository_prepare_20261012/editorial_scripts/`, checking copied
bytes before removing them from the active script directory. It did not delete
datasets, models, scientific outcomes or any frozen report.

## Required scientific inputs

The core experiments require native re-optimized motion data and language-specific
SignSparK-BT checkpoints. Local DDPM/layer experiments also require the shared
pretraining checkpoint, TRAIN-only normalization, cached generated motion and
frozen layer/reader checkpoints listed in their protocols. Public CFM replay uses
released body/hand EMA weights, the M-CLIP text encoder, released predicted glosses
and a TRAIN-only retrieval dictionary.

The existing download scripts document public artifact revisions, but project-
fitted weights and full frozen reports are not supplied by a Git clone. No remote
archive URL is fabricated here. Restore the original full archive with its
relative paths to repeat its verification/inference, or regenerate inputs using
new experiment directories and protocols. Historical absolute paths identify
recorded inputs; they are not portable download URLs.

Get datasets/models under their original access terms and follow their native
feature layout, frame cap, vocabulary and normalization. HumanML3D native RIC
and forward-kinematics recovery verify positions and rotations without a body
model. A compatible licensed body model is needed for mesh/avatar rendering;
no such rendering is claimed here.

The compact metadata's complete SHA256 inventories deliberately include filenames
whose contents are not exported. Use `make check` for the exported subset and
`make verify-archive` only when the full local `reports/` tree is present.
