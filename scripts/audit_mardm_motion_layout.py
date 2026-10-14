"""Run the full layout audit with lossless bounding boxes for TeX math glyphs.

Poppler can export CMEX delimiter glyph codes as XML-forbidden control characters.
Only the parser copy of word text is normalized; every word and bounding box is
retained. The original PDF, text and bbox exports remain untouched. Rendering,
missing-glyph, font embedding, integrity and margin checks are still mandatory.
"""
from pathlib import Path
import io
import re
import json
from xml.etree import ElementTree as ET
import audit_current_motion_layout as native
from motion_release_audit_common import sha

R = Path('reports/cvpr_mardm_followup_20261013')


def main():
    assert not (R / 'manifest_sha256.json').exists()
    original_parse = ET.parse
    diagnostics = {}
    forbidden = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

    def parse(path, *args, **kwargs):
        if not str(path).endswith('_bbox.html'):
            return original_parse(path, *args, **kwargs)
        text = Path(path).read_text(); changes = []

        def word(match):
            prefix, value, suffix = match.groups()
            codes = [ord(c) for c in value if forbidden.fullmatch(c)]
            if codes:changes.append({'word_tag':prefix, 'Poppler_control_codes':codes, 'bounding_box_retained':True})
            return prefix + forbidden.sub('\ufffd', value) + suffix

        mapped = re.sub(r'(<word\b[^>]*>)([^<]*)(</word>)', word, text)
        assert not forbidden.search(mapped), 'Control character outside word text'
        assert mapped.count('<word ') == text.count('<word ')
        diagnostics[Path(path).name] = {'control_word_count':len(changes), 'control_words':changes, 'no_word_removed':True, 'original_export_sha256':sha(path)}
        return original_parse(io.StringIO(mapped), *args, **kwargs)

    ET.parse = parse
    native.ROOT = R
    try:native.main()
    finally:ET.parse = original_parse
    result = json.loads((R / 'layout_verification.json').read_text())
    result['Poppler_math_XML_extraction'] = diagnostics
    result['all_original_word_bounding_boxes_checked'] = True
    result['PDF_or_font_substitution_for_parser_issue'] = False
    result['layout_verifier_sha256'] = sha(__file__)
    (R / 'layout_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('MARDM_FULL_LAYOUT_VERIFIED', {k:v['control_word_count'] for k,v in diagnostics.items()}, flush=True)


if __name__ == '__main__':main()
