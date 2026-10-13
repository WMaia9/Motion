"""Export reviewed sources and compact evidence from a configured frozen delivery.

Never writes to reports, changes scientific producers, or exports model/data arrays.
The explicit configuration prevents reverting a newer manuscript to an old round.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METADATA = (
    'analysis.json', 'summary.json', 'verification.json', 'protocol.json',
    'plan.json', 'complete.json', 'baseline_complete.json',
    'transfer_protocol.json', 'manifest_sha256.json',
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def local(name: str) -> Path:
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Export path outside repository: {name}')
    return path


def copy_checked(source: Path, target: Path, report: Path, manifest: dict) -> None:
    relative = str(source.relative_to(report))
    if relative not in manifest or sha(source) != manifest[relative]:
        raise ValueError(f'Frozen input fingerprint mismatch: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    config = read(ROOT / 'configs/reviewed_delivery.json')
    delivery = local(config['report'])
    paper = local(config['paper_source'])
    manifest_path = delivery / 'manifest_sha256.json'
    if not manifest_path.is_file():
        raise SystemExit('Export requires the configured frozen local delivery.')
    manifest = read(manifest_path)
    layout = read(delivery / 'layout_verification.json')
    output = ROOT / 'paper/current'
    sources = [p for pattern in ['*.tex', '*.bib', '*.sty', '*.bst']
               for p in paper.glob(pattern)]
    sources += list((paper / 'figures').glob('*.pdf'))
    sources += list((paper / 'figures/source_frames').glob('*.png'))
    hashes = {}
    for source in sorted(sources):
        relative = source.relative_to(paper)
        target = output / relative
        copy_checked(source, target, delivery, manifest)
        hashes[str(relative)] = sha(target)
    for name in ['main', 'supplement']:
        target = output / 'reviewed' / (name + '.pdf')
        copy_checked(paper / 'build' / target.name, target, delivery, manifest)
        hashes[str(target.relative_to(output))] = sha(target)
    write_json(output / 'provenance.json', {
        'review_date': config['review_date'],
        'original_manuscript': config['paper_source'],
        'frozen_delivery_manifest_sha256': sha(manifest_path),
        'files_sha256': hashes,
        'style': 'unmodified official CVPR kit, upstream2026',
        'style_commit': '291758547e923160eb4d37079b7b9f0dfce82355',
        'target_year': 2027, 'format_provisional': True,
        'main_content_pages': layout['main']['content_pages'],
        'supplement_content_pages': layout['supplement']['content_pages'],
        'source_frames': config.get('figure_scope', 'see figure captions and frozen evidence maps'),
        'scope': 'reviewed snapshot; anonymous authors and DRAFT submission ID',
    })
    entries = []
    extra = config.get('additional_metadata', {})
    for record_path in sorted((ROOT / 'reports').glob('*/manifest_sha256.json')):
        report = record_path.parent
        record = read(record_path)
        for name in dict.fromkeys(METADATA + tuple(extra.get(report.name, []))):
            source = report / name
            if not source.is_file():
                continue
            if source.stat().st_size > 1024 * 1024:
                raise ValueError(f'Metadata too large for compact export: {source}')
            relative = Path(report.name) / name
            target = ROOT / 'results' / relative
            if name == 'manifest_sha256.json':
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            else:
                copy_checked(source, target, report, record)
            entries.append({
                'path': str(relative), 'sha256': sha(source),
                'original_path': str(source.relative_to(ROOT)),
                'original_manifest_sha256': sha(record_path),
            })
    write_json(ROOT / 'results/index.json', {
        'export_date': config['review_date'],
        'scope': 'aggregate metadata only; full archive data and weights are not in Git',
        'full_manifests_are_inventories_not_complete_exports': True,
        'report_count': len({Path(item['path']).parts[0] for item in entries}),
        'files': entries,
    })
    for original, name in config.get('documents', {}).items():
        copy_checked(delivery / original, ROOT / 'docs' / name, delivery, manifest)
    print(f'Exported reviewed manuscript and {len(entries)} verified metadata files.')


if __name__ == '__main__':
    main()
