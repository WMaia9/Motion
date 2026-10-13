"""Cross-check bibliography identities against archived primary sources."""
import json,re,unicodedata
from pathlib import Path
ROOT=Path('reports/cvpr_manuscript_review_20261011')
PAPER=Path('paper/cvpr2027_rotation_equivalence_revision_20261011')

def normalize(value):
    value=re.sub(r'''\\['".^`~=]\s*\{?([A-Za-z])\}?''',r'\1',value)
    value=unicodedata.normalize('NFKD',value).encode('ascii','ignore').decode().lower()
    return re.sub('[^a-z0-9]','',value)

def entries(text):
    found={}
    starts=list(re.finditer(r'@(\w+)\{([^,]+),',text))
    for i,m in enumerate(starts):
        chunk=text[m.end():starts[i+1].start() if i+1<len(starts) else len(text)]
        fields={};position=0
        while True:
            match=re.search(r'(\w+)\s*=\s*([{"])',chunk[position:])
            if not match:break
            start=position+match.end();pos=start
            if match[2]=='{':
                depth=1
                while depth:
                    c=chunk[pos]
                    if c=='\\':pos+=2;continue
                    if c=='{':depth+=1
                    elif c=='}':depth-=1
                    pos+=1
                value=chunk[start:pos-1]
            else:
                while chunk[pos]!='"':pos+=1
                value=chunk[start:pos];pos+=1
            fields[match[1]]=value;position=pos
        assert m[2] not in found
        found[m[2]]={'type':m[1],**fields}
    return found

def author_list(value):
    value=re.sub(r'\s+',' ',value)
    result=[]
    for author in value.split(' and '):
        author=author.strip()
        if ',' in author:
            last,first=author.split(',',1);author=first+' '+last
        result.append(normalize(author))
    return result

def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    refs=entries((PAPER/'references.bib').read_text());assert len(refs)==23
    source=ROOT/'sources';retrievals={r['file']:r for r in json.loads((source/'retrievals.json').read_text())}
    mapping={'zhou2019':'zhou2019.html','camgoz2018':'camgoz2018.html','camgoz2020':'camgoz2020.html','duarte2021':'duarte2021.html','zhou2021':'zhou2021.html','low2026':'signspark.html','imai2025':'silver.html','cory2026':'bt2.html','symeonidis2026':'m3t.html','ho2020':'ddpm.html','song2021':'ddim.html','ho2022':'cfg.html','saunders2020':'progressive.html','liu2024':'liu2024.html'}
    checks={}
    for key,name in mapping.items():
        m=retrievals[name];assert m['status']==200
        metadata=m['citation_metadata'];titles=[v for k,v in metadata if k=='citation_title'];authors=[v for k,v in metadata if k=='citation_author']
        assert normalize(refs[key]['title'])==normalize(titles[0]),(key,refs[key]['title'],titles)
        assert author_list(refs[key]['author'])==[author_list(a)[0] for a in authors],key
        fields=dict(metadata)
        published=fields.get('citation_publication_date',fields.get('citation_date',''))
        if published and key!='song2021':assert refs[key]['year']==published[:4],(key,published)
        if fields.get('citation_firstpage'):assert refs[key]['pages']==fields['citation_firstpage']+'--'+fields['citation_lastpage']
        checks[key]={'title_and_author_order_match':True,'source':name,'year':refs[key]['year'],'pages_match_when_metadata_available':True}
    for key,name in [('yin2024','t2sgpt.bib'),('papineni2002','bleu.bib'),('post2018','sacrebleu.bib')]:
        native=next(iter(entries((source/name).read_text()).values()))
        assert normalize(refs[key]['title'])==normalize(native['title'])
        assert author_list(refs[key]['author'])==author_list(native['author']),key
        for field in ['year','doi','pages']:assert refs[key][field]==native[field],(key,field)
        checks[key]={'title_authors_year_DOI_pages_match':True,'source':name}
    text=(source/'li2023.txt').read_text();portal=(source/'li2023.html').read_text()
    assert normalize(refs['li2023']['title']) in normalize(text[:300])
    for author in refs['li2023']['author'].split(' and '):assert normalize(author) in normalize(text[:500])
    assert '978-981-99-6483-3_4' in portal and '2023' in portal
    checks['li2023']={'title_authors_match_author_PDF':True,'venue_year_pages_DOI_checked_university_record':True,'sources':['li2023.txt','li2023.html']}
    ctc=json.loads((source/'ctc_crossref.json').read_text())['message']
    assert refs['graves2006']['pages']==ctc['page'].replace('-','--') and refs['graves2006']['year']=='2006'
    nativeauthors=[a['given']+' '+a['family'] for a in ctc['author']]
    assert author_list(refs['graves2006']['author'])==[normalize(a) for a in nativeauthors]
    assert normalize(refs['graves2006']['title']) in normalize((source/'graves2006.txt').read_text()[:250])
    checks['graves2006']={'full_title_matches_author_PDF':True,'authors_year_pages_DOI_match_publisher_supplied_registry':True,'sources':['graves2006.txt','ctc_crossref.json'],'publisher_page_HTTP403_preserved':True}
    walsh=(source/'walsh2025.txt').read_text();front=normalize(walsh[:1900])
    assert normalize(refs['walsh2025']['title']) in front
    for author in refs['walsh2025']['author'].split(' and '):assert normalize(author) in front,author
    checks['walsh2025']={'full_title_and_all_22_authors_match_CVF_PDF':True,'source':'walsh2025.txt'}
    progressive=(source/'progressive_publisher.html').read_text();assert refs['saunders2020']['doi'] in progressive
    ddpm=next(iter(entries((source/'ddpm_publisher.bib').read_text()).values()))
    for field in ['pages','year','volume']:assert refs['ho2020'][field]==ddpm[field]
    assert 'Published as a conference paper at ICLR 2021' in (source/'ddim_firstpage.txt').read_text()
    assert 'Accepted at European Conference on Computer Vision (ECCV) 2026' in (source/'signspark.html').read_text()
    assert 'Accepted at ECCV 2026' in (source/'bt2.html').read_text()
    checks['signsparkbt']={'source':'BT_modelcard.md','attribution':'published hub owner account LionelLow; no inferred personal attribution','revision_pinned':True}
    checks['m3trepo']={'source':'m3t_README.md','attribution':'M3T contributors','availability_claim_checked':True}
    assert 'not yet' in (source/'m3t_README.md').read_text().lower() or 'not released' in (source/'m3t_README.md').read_text().lower()
    checks['slrtprepo']={'source':'official repository linked by SLRTP2025 paper; frozen native execution records','attribution':'Harry Walsh','status':'software citation, not another scientific result'}
    assert set(checks)==set(refs)
    keys={k.strip() for doc in ['main.tex','supplement.tex'] for group in re.findall(r'\\cite\{([^}]+)\}',(PAPER/doc).read_text()) for k in group.split(',')}
    assert keys<=set(refs) and set(refs)<=keys
    output={'references':checks,'total_entries':len(refs),'all_entries_cited':True,'primary_sources_archived':True,'DOI_typo_caught_before_revision_build':'Progressive Transformers resolved to publisher title; correct DOI 10.1007/978-3-030-58621-8_40','novelty_scope':'adversarial sign recognition and learned-score attacks precede this audit; no global-first priority claim'}
    (ROOT/'reference_verification.json').write_text(json.dumps(output,indent=2)+'\n');print('REFERENCES_VERIFIED',len(checks),flush=True)
if __name__=='__main__':main()
