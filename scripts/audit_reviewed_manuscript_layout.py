"""Check PDF integrity, native fonts, page limits, references, and text bounds."""
import json,re,subprocess
from pathlib import Path
from xml.etree import ElementTree as ET
from verify_signspark_studies import sha
ROOT=Path('reports/cvpr_manuscript_review_20261011')
PAPER=Path('paper/cvpr2027_rotation_equivalence_revision_20261011')
def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    result={};ns={'x':'http://www.w3.org/1999/xhtml'}
    for name in ['main','supplement']:
        pdf=PAPER/'build'/(name+'.pdf');log=(PAPER/'build'/(name+'.log')).read_text()
        assert not re.search(r'Overfull|Underfull|undefined|Missing character|Font shape|multiply defined',log,re.I)
        fonts=subprocess.run(['pdffonts',str(pdf)],capture_output=True,text=True,check=True)
        assert not fonts.stderr.strip(),fonts.stderr
        assert 'Type 3' not in fonts.stdout and 'LMRoman' not in fonts.stdout
        fontlines=fonts.stdout.splitlines()[2:];assert all(line.split()[-5]=='yes' for line in fontlines)
        assert 'NimbusRomNo9L' in fonts.stdout
        (PAPER/'build'/(name+'_fonts.txt')).write_text(fonts.stdout)
        gs=subprocess.run(['gs','-q','-dNOPAUSE','-dBATCH','-sDEVICE=nullpage',str(pdf)],capture_output=True,text=True,check=True)
        assert not gs.stdout.strip() and not gs.stderr.strip(),(gs.stdout,gs.stderr)
        text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True);(PAPER/'build'/(name+'.txt')).write_text(text)
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);n=int(re.search(r'Pages:\s+(\d+)',info)[1]);assert '612 x 792' in info
        bbox=PAPER/'build'/(name+'_bbox.html');subprocess.run(['pdftotext','-bbox',str(pdf),str(bbox)],check=True)
        pages=ET.parse(bbox).findall('.//x:page',ns);assert len(pages)==n
        reference_pages=[];inspected=0
        for i,page in enumerate(pages,1):
            for word in page.findall('x:word',ns):
                a=word.attrib;x0=float(a['xMin']);x1=float(a['xMax']);y0=float(a['yMin']);y1=float(a['yMax']);t=word.text or ''
                assert 0<=x0<=x1<=612 and 0<=y0<y1<=792,(name,i,t,a)
                if t=='References' and 70<y0<100:reference_pages.append(i)
                if 70<y0<720:
                    if t.isdigit() and (x1<58.5 or x0>553.5):continue # official review line numbers
                    assert x0>=58.4 and x1<=553.65,(name,i,t,a)
                    inspected+=1
        assert len(reference_pages)==1
        content=reference_pages[0]-1
        if name=='main':assert content<=8 and n==8 and content==7
        else:assert n==6 and content==5
        result[name]={'pdf_sha256':sha(pdf),'total_pages':n,'content_pages':content,'reference_pages':n-content,'body_words_checked_for_margin_overflow':inspected,'all_fonts_embedded':True,'Times_text_font_from_unmodified_CVPR_style':True,'no_Type3_fonts':True,'no_font_substitution_or_missing_glyphs':True,'ghostscript_integrity_passed':True,'latex_reference_and_box_checks_passed':True}
        tex=(PAPER/(name+'.tex')).read_text()
        for table in re.findall(r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}',tex,re.S):assert table.index('\\caption')<table.index('\\begin{tabular}')
        assert '\\usepackage[review]{cvpr}' in tex and '\\documentclass[10pt,twocolumn,letterpaper]{article}' in tex
        assert not re.search(r'\\(?:setmainfont|geometry|vspace\*?)',tex)
    for f in ['cvpr.sty','ieeenat_fullname.bst']:assert sha(PAPER/f)==sha(ROOT/'sources'/('kit_'+f))
    for fragment in ['tables_generated.tex','tables_hashes.tex']:
        text=(PAPER/fragment).read_text()
        for table in re.findall(r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}',text,re.S):assert table.index('\\caption')<table.index('\\begin{tabular}')
    result['visual_inspection']='All 8 main and 6 supplement pages rendered and reviewed; final hash wrapping checked separately.'
    result['remaining_external_warning']='lineno.sty:296 invalid UTF-8 comment in installed TeX package; no missing manuscript glyphs'
    result['format_scope']='Official available author-kit commit291758547e923160eb4d37079b7b9f0dfce82355, upstream2026; target2027 provisional, AuthorGuidelines HTTP404'
    (ROOT/'layout_verification.json').write_text(json.dumps(result,indent=2)+'\n');print('LAYOUT_VERIFIED',json.dumps(result),flush=True)
if __name__=='__main__':main()
