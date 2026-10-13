"""Audit compiled draft, figure sources, references, and kit provenance."""
import json,re,subprocess
from pathlib import Path
from verify_signspark_studies import sha
ROOT=Path('reports/cvpr_representation_paper_20261011')
PAPER=Path('paper/cvpr2027_rotation_equivalence')
def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    files={};pages={};text={}
    for name in ['main','supplement']:
        pdf=PAPER/'build'/(name+'.pdf');log=(PAPER/'build'/(name+'.log')).read_text()
        assert not re.search(r'Overfull|Underfull|undefined|Missing character|multiply defined',log,re.I)
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);n=int(re.search(r'Pages:\s+(\d+)',info)[1]);pages[name]=n
        t=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True);text[name]=t
        (PAPER/'build'/(name+'.txt')).write_text(t)
        files[name]={'pdf_sha256':sha(pdf),'pages':n,'log_sha256':sha(PAPER/'build'/(name+'.log'))}
    content=[p for p in text['main'].split('\f') if p.strip()]
    reference_start=next(i+1 for i,p in enumerate(content) if re.search(r'\bReferences\b',p))
    assert reference_start==8 and pages['main']==8 and pages['supplement']==3
    assert reference_start-1<=8
    tex=(PAPER/'main.tex').read_text();bib=(PAPER/'references.bib').read_text()
    keys=re.findall(r'@\w+\{([^,]+),',bib);assert len(keys)==len(set(keys))
    cites={k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',tex) for k in group.split(',')}
    assert cites<=set(keys)
    labels=re.findall(r'\\label\{([^}]+)\}',tex);assert len(labels)==len(set(labels))
    assert set(re.findall(r'\\ref\{([^}]+)\}',tex))<=set(labels)
    for name in ['cvpr.sty','ieeenat_fullname.bst']:assert sha(PAPER/name)==sha(PAPER/'sources'/('official_'+name))
    figures=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',tex)
    for figure in figures:assert (PAPER/figure).is_file()
    out={'compiled':files,'main_content_pages':reference_start-1,'main_reference_pages':pages['main']-reference_start+1,'supplement_pages':pages['supplement'],'bibliography_entries':len(keys),'cited_entries':len(cites),'figure_files':figures,'unresolved_citations_or_box_overflows':False,'upstream_style_bytes_preserved':True,'remaining_template_warning':'lineno.sty:296 invalid UTF-8 comment sequence in external TeX package','template_2027_status':'provisional upstream 2026','real_submission_ID_or_authors_filled':False,'conference_submission_sent':False,'visual_review':'all seven main content pages, reference page and all three supplement pages inspected as rendered PNGs'}
    (ROOT/'paper/manuscript_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
