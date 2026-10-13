"""Archive the reviewed manuscript and checks; preserve all prior reports."""
import ast,json,shutil,subprocess,sys,zipfile
from pathlib import Path
from verify_signspark_studies import sha,freeze
ROOT=Path('reports/cvpr_manuscript_review_20261011')
PAPER=Path('paper/cvpr2027_rotation_equivalence_revision_20261011')
SCRIPTS=['fetch_manuscript_review_sources','build_reviewed_manuscript_tables','plot_reviewed_manuscript','audit_reviewed_references','audit_reviewed_manuscript_evidence','audit_reviewed_manuscript_layout','finalize_manuscript_review']
def read(p):return json.loads(Path(p).read_text())
def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    layout=read(ROOT/'layout_verification.json');evidence=read(ROOT/'evidence_verification.json');refs=read(ROOT/'reference_verification.json')
    for name in ['main','supplement']:assert sha(PAPER/'build'/(name+'.pdf'))==layout[name]['pdf_sha256']
    assert refs['total_entries']==23 and evidence['canonical_attack_pairs']==1024
    prefix='''<!-- cvpr_manuscript_review_20261011 -->
## Artigo revisado em formato CVPR — 2026-10-11

Nova versão **Same Rotations, Different Scores**, preservando o artigo/relatórios anteriores. [PDF principal](paper/cvpr2027_rotation_equivalence_revision_20261011/build/main.pdf): **7 páginas de conteúdo + 1 de referências**. [Suplemento](paper/cvpr2027_rotation_equivalence_revision_20261011/build/supplement.pdf): **5 de conteúdo + 1 de referências**. [Revisão detalhada](reports/cvpr_manuscript_review_20261011/REVISAO_E_ENTREGA.md), [mapa de evidências](reports/cvpr_manuscript_review_20261011/EVIDENCE_MAP.md).

Redação/prova/protocolo/resultados revisados. Inclui a conferência CSL posterior: 768 previsões feature-only reproduzidas e BLEU independente 7,3955→38,2706, com alvo conhecido na otimização. O domínio dos ataques é explicitamente não canônico, mas não degenerado e aceito pela API crua; converter mantém rotações. A tese é falha de equivalência de representação e fechamento de canal. Não é nova geração SOTA, cifra/anonimização, semântica humana ou matemática nova.

**23 referências** conferidas em fontes primárias. Adicionados antecedentes de ataques a reconhecimento de sinais por esqueleto (ICIRA 2023) e modelos de qualidade (CVPR 2024), Progressive Transformers como origem de BT em produção, e referências BLEU/SacreBLEU/CTC/DDPM/DDIM/CFG. Autores, metadados e DOI revisados. Nenhuma garantia de prioridade mundial ou comparação executada com SiLVERScore/BT2.

**Formato e integridade:** kit oficial fixado, sem mudar sty/BST ou margens. Corpo Times 10pt/T1, fontes embutidas e escaláveis, sem Type 3, legendas acima das tabelas, hashes com quebra correta. Todas as 14 páginas renderizadas e revisadas; fontes, margens, referências, caixas e Ghostscript passaram. Kit disponível ainda 2026 e guidelines 2027 HTTP404: preparação para 2027 provisória. Autoria anônima/ID DRAFT; nenhuma submissão/registro/publicação.

**Evidências:** 33.126 cálculos de sentenças BLEU e 3.336 WER reavaliados nos outputs salvos, incluindo os 20.544 decodes gerados anteriores; nove checkpoints hash-verificados. Quatro reports anteriores, 532 arquivos, integralmente SHA-verificados. Nenhum novo treino/inferência de modelo nesta revisão. Relatórios antigos não foram alterados. Código/fontes/diffs/auditorias/PDFs/LaTeX/snapshots arquivados e manifesto por último.

**Lacunas científicas preservadas:** três BTs da mesma família, CTC local, nenhum BT publicado independente compatível atacado; quatro variantes locais/uma seed/duração real; sem humanos/meshes; custo How2Sign da defesa e gaps ASL/CSL dos cards; nenhuma inversão de vencedores apoiada por ambos os intervalos. Revisão editorial não substitui esses experimentos. [Reprodução](reports/cvpr_manuscript_review_20261011/REPRODUCAO.md).

---

'''
    for name in ['CURRENT_RESEARCH.md','RESEARCH_STATUS_AND_NEXT_EXPERIMENTS.md']:
        path=Path(name);assert '<!-- cvpr_manuscript_review_20261011 -->' not in path.read_text()
        path.write_text(prefix+path.read_text());shutil.copy2(path,ROOT/'sources'/('after_'+name))
    pending=SCRIPTS.copy();found=set()
    while pending:
        name=pending.pop()
        if name in found:continue
        found.add(name);path=Path('scripts')/(name+'.py');shutil.copy2(path,ROOT/'scripts'/path.name)
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[v.name for v in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in modules:
                n=module.split('.')[0]
                if (Path('scripts')/(n+'.py')).exists() and n not in found:pending.append(n)
    (ROOT/'sources/script_provenance.json').write_text(json.dumps({'producers':SCRIPTS,'all_archived_dependencies':sorted(found),'old_mains_not_executed':True},indent=2)+'\n')
    (ROOT/'sources/pip_list.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','list','--format=freeze'],text=True))
    for src in Path('/tmp').glob('cvpr_manuscript_*20261011.log'):shutil.copy2(src,ROOT/'logs'/src.name)
    # Source ZIP includes only manuscript assets, not checkpoints/dataset outputs.
    archive=PAPER/'manuscript_source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PAPER.rglob('*')):
            if p.is_file() and p!=archive and p.suffix in ['.tex','.bib','.sty','.bst','.pdf','.md','.json'] and 'sources' not in p.relative_to(PAPER).parts:
                z.write(p,str(p.relative_to(PAPER)))
    shutil.copytree(PAPER,ROOT/'paper/manuscript',dirs_exist_ok=True)
    shutil.copy2(archive,ROOT/'paper/manuscript_source.zip')
    assert sha(PAPER/'cvpr.sty')==sha(ROOT/'sources/kit_cvpr.sty')
    freeze(ROOT,{'layout':layout,'references':refs,'evidence':evidence,'source_zip_sha256':sha(archive),'scripts_archived':len(found),'prior_reports_preserved':True,'new_training_or_model_inference':False,'no_submission_or_publication':True,'all_writers_closed_before_manifest':True})
if __name__=='__main__':main()
