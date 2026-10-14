"""Repository maintenance commands, usable on a lightweight clone without a GPU."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper/current'
ALLOWED_DIRS = {'.github', 'configs', 'docs', 'requirements', 'results', 'scripts', 'tools'}
ALLOWED_FILES = {
    '.gitignore', '.gitattributes', '.editorconfig', 'Makefile', 'README.md',
    'CURRENT_RESEARCH.md', 'RESEARCH_STATUS_AND_NEXT_EXPERIMENTS.md',
    'CONTRIBUTING.md', 'NOTICE.md',
}
FORBIDDEN_SUFFIXES = {
    '.pt', '.pth', '.ckpt', '.npy', '.npz', '.pkl', '.pickle', '.lmdb',
    '.mdb', '.safetensors', '.onnx', '.zip', '.tar',
}
TEXT_SUFFIXES = {'.py', '.md', '.json', '.yaml', '.yml', '.txt', '.tex', '.bib', '.sty', '.bst'}
SECRET_PATTERN = re.compile(
    rb'(?m)^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----$|'
    rb'github_pat_[A-Za-z0-9_]{70,}|gh[pousr]_[A-Za-z0-9]{36}'
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(
            hashlib, 'file_digest') else _digest_stream(stream)


def _digest_stream(stream) -> str:
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def run(command: list[str], cwd: Path = ROOT) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def candidates() -> list[Path]:
    names = subprocess.check_output([
        'git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z',
    ], cwd=ROOT).split(b'\0')
    return sorted({ROOT / name.decode() for name in names if name})


def inside(path: Path, parent: Path) -> bool:
    return path.resolve().is_relative_to(parent.resolve())


def verify_exported_results() -> int:
    index = read(ROOT / 'results/index.json')
    for item in index['files']:
        path = ROOT / 'results' / item['path']
        if not inside(path, ROOT / 'results') or path.is_symlink():
            raise ValueError(f'Invalid evidence path: {item["path"]}')
        if digest(path) != item['sha256']:
            raise ValueError(f'Evidence fingerprint changed: {item["path"]}')
    return len(index['files'])


def verify_paper() -> int:
    record = read(PAPER / 'provenance.json')
    for relative, expected in record['files_sha256'].items():
        path = PAPER / relative
        if not inside(path, PAPER) or path.is_symlink():
            raise ValueError(f'Invalid manuscript path: {relative}')
        if digest(path) != expected:
            raise ValueError(f'Reviewed manuscript fingerprint changed: {relative}')
    keys = set(re.findall(r'@\w+\s*\{([^,]+),', (PAPER / 'references.bib').read_text()))
    cited = set()
    for path in PAPER.glob('*.tex'):
        text = path.read_text()
        for group in re.findall(r'\\cite\w*\{([^}]+)\}', text):
            cited.update(group.split(','))
        for item in re.findall(r'\\input\{([^}]+)\}', text):
            if not (PAPER / (item + '.tex')).is_file():
                raise ValueError(f'Missing LaTeX input: {item}')
        for item in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text):
            if not inside(PAPER / item, PAPER) or not (PAPER / item).is_file():
                raise ValueError(f'Missing/invalid figure: {item}')
    if not cited <= keys:
        raise ValueError(f'Missing references: {sorted(cited - keys)}')
    return len(keys)


def verify_document_links(files: list[Path]) -> int:
    checked = 0
    for path in files:
        if path.suffix != '.md':
            continue
        for target in re.findall(r'\[[^\]\n]+\]\(([^\s)]+)\)', path.read_text()):
            if target.startswith(('http://', 'https://', 'mailto:', '#')):
                continue
            target = target.split('#', 1)[0]
            linked = path.parent / target
            if not inside(linked, ROOT) or not linked.exists():
                raise ValueError(f'Broken local documentation link in {path.relative_to(ROOT)}: {target}')
            checked += 1
    return checked


def check() -> None:
    files = candidates()
    total = 0
    scripts = 0
    for path in files:
        rel = path.relative_to(ROOT)
        permitted = rel.parts[0] in ALLOWED_DIRS or str(rel) in ALLOWED_FILES
        permitted |= rel.parts[:2] == ('paper', 'current')
        if not permitted:
            raise ValueError(f'Local-only file would enter Git: {rel}')
        if path.is_symlink() or not path.is_file() or not inside(path, ROOT):
            raise ValueError(f'Invalid repository entry: {rel}')
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or '.env' == path.name:
            raise ValueError(f'Experimental binary or local secret would enter Git: {rel}')
        size = path.stat().st_size
        if size > 5 * 1024 * 1024:
            raise ValueError(f'File exceeds source-repository size limit: {rel}')
        total += size
        if path.suffix == '.py':
            compile(path.read_bytes(), str(rel), 'exec')
            scripts += 1
        if path.suffix == '.json':
            json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f'Nonfinite JSON constant in {rel}: {value}')))
        if path.suffix in TEXT_SUFFIXES and SECRET_PATTERN.search(path.read_bytes()):
            raise ValueError(f'Credential-like content in {rel}; value suppressed')
    if total > 25 * 1024 * 1024:
        raise ValueError('Git source tree exceeds25MiB; inspect generated assets.')
    metadata = verify_exported_results()
    references = verify_paper()
    links = verify_document_links(files)
    expected = [
        'features/private.png', 'annotations/private.csv', 'reports/model.pt',
        'external/SignSparK/sample.py', 'venv_mska/bin/python',
        '.venv_signspark_confirmation/bin/python', 'maintenance/local.json',
        '.claude/settings.local.json', 'paper/cvpr2027/tools/tectonic',
        'paper/current/build/main.pdf', '.env',
    ]
    result = subprocess.run(['git', 'check-ignore', '--stdin'], cwd=ROOT, input='\n'.join(
        expected) + '\n', text=True, capture_output=True, check=False)
    if set(result.stdout.splitlines()) != set(expected):
        raise ValueError('Git ignore policy does not exclude every local-only category')
    print(f'PASS: {len(files)} Git files, {total / 1024 / 1024:.2f}MiB; '
          f'{scripts} Python files; {metadata} evidence fingerprints; '
          f'{references} references; {links} local links.')


def summary() -> None:
    stability_path = ROOT / 'results/cvpr_sign_stability_20261014/analysis.json'
    if stability_path.is_file():
        print('Real-sign stability: exact raw/canonical replay; matched per-clip '
              'Gaussian fields and gauge-preserving physical changes.')
        for name, row in read(stability_path)['results'].items():
            q = row['primary_summary']
            for condition in ['canonical', 'gaussian_support_mean', 'physical_matched']:
                v = q[condition]
                print(f'{name}/{condition}: changed '
                      f'{100*v["token_changed_fraction"]:.1f}%; '
                      f'normalized edit {v["mean_normalized_edit_distance"]:.3f}')
        ratio = read(ROOT / 'results/cvpr_sign_stability_20261014/rotation_ratio.json')
        print('Same-sign fixed fiber / physical FID changes: '
              f'{ratio["directions"]["plus"]["ratio_of_mean_FID_changes"]:.3f}/'
              f'{ratio["directions"]["minus"]["ratio_of_mean_FID_changes"]:.3f}; '
              'direction-specific, no universal sensitivity claim.')
    coherent_path = ROOT / 'results/cvpr_coherent_blocks_20261014/guo/analysis.json'
    if coherent_path.is_file():
        controls = read(coherent_path)['conditions']
        print('Fixed-direction controls: twenty paired draws; numerical L2 matched; '
              'coherence is temporal, not cross-block physical consistency.')
        for block in ['rotation', 'velocity', 'contact', 'position']:
            values = [controls[f'{block}_fixed_{sign}_r3']['FID']['mean_change']
                      for sign in ['plus', 'minus']]
            print(f'Guo/{block}, kappa3 fixed +/−: FID '
                  f'{values[0]:+.5f}/{values[1]:+.5f}')
        natural = read(ROOT / 'results/cvpr_sign_natural_restoration_20261014/analysis.json')
        for name, row in natural['results'].items():
            print(f'Natural sign/{name}: {row["changed_hypotheses"]}/'
                  f'{row["clips"]} hypotheses change without attack; BLEU '
                  f'{row["BLEU4"]["raw"]:.4f}/{row["BLEU4"]["canonical"]:.4f}')
    full_tmr = ROOT / 'results/cvpr_tmr1024_controls_20261013/analysis.json'
    generalization = full_tmr if full_tmr.is_file() else ROOT / 'results/cvpr_tmr_attack_20261013/analysis.json'
    tmr_count = 1024 if full_tmr.is_file() else 128
    if generalization.is_file():
        for source, values in read(generalization).items():
            before = values['identity']['R_precision'][0]
            after = values['actual']['actual']['R_precision'][0]
            closed = values['actual']['canonical']['R_precision'][0]
            print(f'TMR/{source}: target-aware R1 {before:.4f} -> {after:.4f}; '
                  f'canonical attack {closed:.4f}; fixed {tmr_count} clips, native cosine')
        additional = read(ROOT / 'results/cvpr_additional_motion_audit_20261013/analysis.json')
        for source, values in additional['completed_methods'].items():
            metric = values['FID']
            print(f'Additional/{source}: FID {metric["raw"]:.5f}/'
                  f'{metric["canonical"]:.5f}; raw-C {metric["delta"]:+.5f}; 20 draws')
        motiondiffuse = read(ROOT / 'results/cvpr_motiondiffuse_compiled_20261013/analysis.json')
        for source, values in motiondiffuse['completed_methods'].items():
            metric = values['FID']
            print(f'Additional/{source}: FID {metric["raw"]:.5f}/'
                  f'{metric["canonical"]:.5f}; raw-C {metric["delta"]:+.5f}; 20 draws')
        qualified = read(ROOT / 'results/cvpr_reward_dev_search_20261013/selection.json')
        print(f'Reward DEV: {qualified["qualified"]} qualifying configurations; '
              'failed qualification cannot test successful reward optimization.')
    mechanism_path = ROOT / 'results/cvpr_rotation_mechanism_20261013/analysis.json'
    if mechanism_path.is_file():
        mechanism = read(mechanism_path)
        print('Rotation-only mechanism: eight release means; descriptive, not family causality.')
        for family, row in mechanism['family_descriptive_means'].items():
            print(f'{family}: mean signed FID gap {row["mean_signed_gap"]:+.5f}')
        essential = read(ROOT / 'results/cvpr_mardm_readers_20261013/analysis.json')
        print(f'MARDM essential readers: Kendall versus canonical Guo '
              f'{essential["Kendall_vs_canonical_Guo"]}; invariance by feature exclusion.')
        for reader in ['guo', 'tmr']:
            controls = read(ROOT / f'results/cvpr_matched_blocks_20261013/{reader}/analysis.json')
            print(f'Equal-L2 {reader}: 20 draws, both radii, all14 conditions; '
                  'paired real-reference FID-zero baseline is by construction.')
    releases = read(ROOT / 'results/cvpr_motion_prevalence_20261012/analysis.json')
    print('HumanML3D released outputs: 20 native repetitions, same motion raw/C.')
    for name, result in releases['methods'].items():
        metric = result['FID']
        interval = ', '.join(f'{v:+.5f}' for v in metric['paired_family_CI95'])
        print(f'{name}: FID {metric["raw_mean"]:.5f}/{metric["canonical_mean"]:.5f}; '
              f'raw-C {metric["paired_raw_minus_canonical"]:+.5f}, '
              f'ten-endpoint CI [{interval}]')
    print('Primary mean FID ranking unchanged; second official weights share architecture.')
    repetitions = read(ROOT / 'results/cvpr_humanml_blind_repetitions_20261012/analysis.json')
    for cohort in repetitions['cohorts']:
        values = cohort['FID']
        print(f'Blind generation seed{cohort["generation_seed"]}: '
              f'FID {values["identity"]:.3f} -> {values["blind_OT_bound010"]:.3f} '
              f'(bound0.1); {values["blind_OT_bound035"]:.3f} (bound0.35)')
    print('Blind cohorts overlap in source groups; no pooled independent-clip CI.')
    pilot = read(ROOT / 'results/cvpr_humanml_reward_pilot_20261012/summary.json')
    metric = pilot['conditions']['raw_reward']['R1']
    print(f'Ordinary MDM reward pilot: raw/C R1 '
              f'{metric["raw_mean"]:.3f}/{metric["canonical_mean"]:.3f}; '
              'retrieval optimization failed; does not test a successful reward fine-tune.')
    reward = read(ROOT / 'results/cvpr_motion_reward_release_20261013/analysis.json')
    for name, result in reward['readers'].items():
        metric = result['reward_finetuned']['R1']
        print(f'Released MotionRFT/{name}: raw/C R1 '
              f'{metric["raw_mean"]:.3f}/{metric["canonical_mean"]:.3f}; '
              'gains over linked base survive canonicalization.')
    data = read(ROOT / 'results/cvpr_second_review_confirmation_20261012/analysis.json')
    print('Fixed TEST confirmation; native BLEU units differ by language.')
    for corpus, result in data['real_motion_layers'].items():
        arm = result['arms']['text_raw']
        interval = ', '.join(f'{value:.2f}' for value in arm['paired_ci95'])
        print(f'{corpus}: {result["baseline_raw"]:.2f} -> {arm["raw_mean"]:.2f}; '
              f'gain {arm["gain"]:.2f}, paired 95% CI [{interval}]')
    adaptive = read(ROOT / 'results/cvpr_adaptive_distance_confirmation_20261012/analysis.json')
    for name in ['circle_correct_step200', 'circle_wrong_step200']:
        result = adaptive['conditions'][name]
        interval = ', '.join(f'{value:.2f}' for value in
                             result['target_gain_family_ci95_four_endpoints'])
        print(f'Adaptive/{name}: {result["baseline_target_BLEU"]:.2f} -> '
              f'{result["attacked_target_BLEU"]:.2f}; gain {result["target_gain"]:.2f}, '
              f'four-comparison-adjusted CI [{interval}]')
    study = read(ROOT / 'results/cvpr_seed_dose_response_20261012/analysis.json')
    for name in ['primary_seven_new_seeds', 'historical_plus_new_ten_seeds']:
        result = study['results']['test'][name]
        interval = ', '.join(f'{value:.2f}' for value in result['seed_and_clip_ci95'])
        print(f'Ordinary/{name}: gap {result["gap"]:.2f}, seed+clip CI [{interval}]')
    human = read(ROOT / 'results/cvpr_humanml3d_audit_20261012/analysis.json')
    for source in ['real', 'mdm']:
        result = human['TEST']['results'][source]['target_caption']
        print(f'HumanML3D/{source}: R1 {result["R1_before"]:.3f} -> '
              f'{result["R1_after"]:.3f}; target-caption access, fixed 1024-clip subset')
    blind = read(ROOT / 'results/cvpr_humanml3d_blind_fid_20261012/analysis.json')
    for name in ['blind_OT_bound010', 'blind_OT_bound035']:
        result = blind['conditions'][name]
        baseline = blind['conditions']['identity']['raw']['FID']
        print(f'Caption-blind/{name}: FID {baseline:.3f} -> '
              f'{result["raw"]["FID"]:.3f}; TRAIN-only targets, fixed 960-clip cohort')
    print('Natural representation dependence confirmed; deliberate training exploitation, '
          'general stealth and better motion are not established. Explicit attack '
          'access and evaluator-family limits remain separate.')


def environment(profile: str) -> None:
    observed = read(ROOT / 'requirements' / (profile + '.observed.json'))
    errors = []
    if sys.version_info[:2] != (3, 10):
        errors.append('Research environment requires Python3.10; lightweight checks do not.')
    for item in observed['direct_dependencies']:
        try:
            version = importlib.metadata.version(item['name'])
        except importlib.metadata.PackageNotFoundError:
            version = 'not installed'
        if version != item['version']:
            errors.append(f'{item["name"]}: observed {item["version"]}, found {version}')
    if errors:
        raise ValueError('\n'.join(errors))
    print(f'PASS: {profile} dependency versions match the observed research environment.')


def vendor(check_only: bool) -> None:
    for entry in read(ROOT / 'configs/external_sources.json')['sources']:
        target = ROOT / entry['path']
        if not inside(target, ROOT / 'external') or target.is_symlink():
            raise ValueError(f'Invalid external checkout path: {entry["path"]}')
        if not re.fullmatch(r'[a-f0-9]{40}', entry['revision']):
            raise ValueError(f'External revision must be a full commit SHA: {entry["name"]}')
        if not target.is_dir():
            if check_only:
                raise ValueError(f'Missing external checkout: {entry["path"]}; run make vendor')
            target.parent.mkdir(exist_ok=True)
            subprocess.run(['git', 'clone', '--no-checkout', entry['url'], str(target)], check=True)
            subprocess.run(['git', '-C', str(target), 'checkout', '--detach', entry['revision']], check=True)
        revision = run(['git', 'rev-parse', 'HEAD'], target)
        if revision != entry['revision']:
            raise ValueError(f'External revision differs: {entry["path"]}; existing checkout preserved')
        subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=target, check=True)
        print(f'PASS: {entry["name"]} at {revision}')


def build_paper() -> None:
    compiler = shutil.which('tectonic')
    local = ROOT / '.tools/tectonic'
    if not compiler and local.is_file():
        compiler = str(local)
    if not compiler:
        raise ValueError('Install Tectonic on PATH; see docs/reproduction.md')
    output = ROOT / '.build/paper'
    output.mkdir(parents=True, exist_ok=True)
    for name in ['main', 'supplement']:
        subprocess.run([compiler, '--keep-logs', '--outdir', str(output),
                        str(PAPER / (name + '.tex'))], cwd=ROOT, check=True)
    print('Built PDFs in .build/paper; reviewed PDFs and frozen evidence were preserved.')


def verify_archive() -> None:
    if not list((ROOT / 'reports').glob('*/manifest_sha256.json')):
        raise ValueError('Full local archive absent; make check verifies the compact Git export.')
    subprocess.run([sys.executable, 'scripts/verify_frozen_research.py'], cwd=ROOT, check=True,
                   env=_no_bytecode_environment())


def _no_bytecode_environment() -> dict[str, str]:
    import os
    return {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}


def help_text() -> None:
    print('make check          Validate the lightweight Git tree and evidence hashes\n'
          'make summary        Show aggregate TEST results, no data/GPU needed\n'
          'make env            Check observed main dependency versions\n'
          'make vendor-check   Check pinned external repositories\n'
          'make vendor         Clone missing pinned external repositories\n'
          'make paper          Build current LaTeX into ignored .build/paper\n'
          'make test-canonicalize Test portable NumPy rotation consistency\n'
          'make test-matched-blocks Check equal-norm geometry/protected frames\n'
          'make test-coherent-blocks Check time-fixed directions and equal norms\n'
          'make test-sign-stability Check norm/Gram-preserving physical controls\n'
          'make verify-archive Verify full local archive (requires reports/)')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=[
        'help', 'check', 'summary', 'env', 'vendor', 'paper', 'verify-archive',
    ])
    parser.add_argument('--profile', choices=['main', 'replay'], default='main')
    parser.add_argument('--check', action='store_true', help='Check vendor paths without cloning')
    args = parser.parse_args()
    try:
        if args.command == 'env':
            environment(args.profile)
        elif args.command == 'vendor':
            vendor(args.check)
        else:
            {'help': help_text, 'check': check, 'summary': summary,
             'paper': build_paper, 'verify-archive': verify_archive}[args.command]()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
