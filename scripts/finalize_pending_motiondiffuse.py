"""Finish verification/build after both registered MotionDiffuse workers complete.

Does not retrain, change seeds, select outcomes, publish, or replace reviewed PDFs.
The latest build still requires a rendered-page review before repository export.
"""
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'reports/cvpr_motiondiffuse_compiled_20261013'
DELIVERY = ROOT / 'reports/cvpr_generalization_review_20261013'
STATUS = DELIVERY / 'pending_motiondiffuse_status.json'


def write_status(stage, **values):
    assert not (DELIVERY / 'manifest_sha256.json').is_file(), 'Frozen delivery cannot be changed'
    data = {'stage': stage, 'updated_utc': datetime.datetime.now(
        datetime.timezone.utc).isoformat(), **values}
    temporary = STATUS.with_suffix('.pending')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(STATUS)
    print(json.dumps(data), flush=True)


def count_results():
    return {dataset: len(list((STUDY / dataset / 'motiondiffuse').glob(
        'rep_*/metrics.json'))) for dataset in ['humanml', 'kit']}


def check_registered_workers():
    record = json.loads((STUDY / 'resume_execution_20261013.json').read_text())
    for dataset, process in record['processes'].items():
        if (STUDY / dataset / 'motiondiffuse/complete.json').is_file():
            continue
        command_file = Path('/proc') / str(process['pid']) / 'cmdline'
        if not command_file.is_file() or b'run_motiondiffuse_compiled_audit.py' not in command_file.read_bytes():
            raise RuntimeError(f'{dataset} worker exited before completion; '
                               'resume the same registered protocol, preserving saved repetitions')


def run(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *arguments],
                   cwd=ROOT, check=True, env={**os.environ, 'OPENBLAS_NUM_THREADS': '4'})


def main():
    os.chdir(ROOT)
    if (DELIVERY / 'manifest_sha256.json').is_file():
        print('Delivery already frozen; no files changed', flush=True)
        return
    # Completion files are produced only after all twenty cells are persisted.
    deadline = time.monotonic() + 24 * 3600
    previous = None
    while not all((STUDY / dataset / 'motiondiffuse/complete.json').is_file()
                  for dataset in ['humanml', 'kit']):
        counts = count_results()
        check_registered_workers()
        if counts != previous:
            write_status('waiting_for_registered_workers', completed_repetitions=counts,
                         required_repetitions_each=20, reviewed_PDFs_preserved=True)
            previous = counts
        if time.monotonic() > deadline:
            raise TimeoutError('Workers did not complete within 24 hours; no final claims exported')
        time.sleep(30)
    assert count_results() == {'humanml': 20, 'kit': 20}
    write_status('verifying_native_geometry_and_metrics', completed_repetitions=count_results())
    if not (STUDY / 'manifest_sha256.json').is_file():
        run('verify_additional_motion_audits.py', 'motiondiffuse')
        run('freeze_generalization_studies.py', 'motiondiffuse')
    else:
        # Idempotent continuation accepts a frozen study only after checking it.
        from freeze_motion_release_studies import sha
        for relative, expected in json.loads((STUDY / 'manifest_sha256.json').read_text()).items():
            assert sha(STUDY / relative) == expected, relative
    write_status('assembling_all_declared_endpoints')
    run('export_generalization_evidence.py')
    run('build_generalization_presentation.py')
    subprocess.run(['make', 'paper'], cwd=ROOT, check=True)
    write_status('build_ready_for_visual_review', completed_repetitions=count_results(),
                 all_declared_experiments_complete=True, reviewed_PDFs_preserved=True,
                 final_delivery_exported=False, publication_or_submission=False)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write_status('failed', error_type=type(error).__name__, error=str(error),
                     completed_repetitions=count_results(), final_delivery_exported=False)
        raise
