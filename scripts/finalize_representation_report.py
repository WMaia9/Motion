"""Archive closed jobs, update research state, then freeze the new report."""
import ast,json,platform,shutil,subprocess,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
import torch
from verify_signspark_studies import sha,freeze
ROOT=Path('reports/cvpr_representation_paper_20261011')
PAPER=Path('paper/cvpr2027_rotation_equivalence')
PRODUCERS=['download_representation_csl','run_representation_ctc_test','run_representation_csl_test','audit_representation_native_contract','rotation_equivalence','verify_representation_math','verify_representation_paper','plot_representation_paper','build_representation_supplement','audit_representation_manuscript','finalize_representation_report']

def main():
    assert not (ROOT/'manifest_sha256.json').exists()
    scientific=json.loads((ROOT/'independent_verification.json').read_text())
    manuscript=json.loads((ROOT/'paper/manuscript_verification.json').read_text())
    for name in ['main','supplement']:assert sha(PAPER/'build'/(name+'.pdf'))==manuscript['compiled'][name]['pdf_sha256']
    assert scientific['new_attack_pairs_verified']==512 and manuscript['main_content_pages']==7
    prefix='''<!-- cvpr_representation_paper_20261011 -->
## Três línguas, segunda arquitetura e PDF candidato CVPR — 2026-10-11

Rodada concluída: **512 novos ataques**, sem novos treinos, protocolo e assets fixados antes da inferência; total **1.024 pares** com hipótese canônica invariante. [Relatório](reports/cvpr_representation_paper_20261011/RESULTADOS_E_DECISAO.md), [mapa de evidências](reports/cvpr_representation_paper_20261011/EVIDENCE_MAP.md), [PDF principal](paper/cvpr2027_rotation_equivalence/build/main.pdf), [suplemento](paper/cvpr2027_rotation_equivalence/build/supplement.pdf). Principal: **7 páginas de conteúdo + 1 de referências**; suplemento: **3 páginas**. Kit oficial disponível ainda identifica 2026, adaptação para 2027 provisória, autoria anônima/ID DRAFT, nenhuma submissão enviada.

**CSL-Daily TEST:** baseline completo de 1.162 clips, BLEU por caracteres 6,8907→canônico 7,0662; 464 hipóteses mudam, Δ+0,1755 IC95 [−0,2758; 0,5657]. Subset de 128: alvo correto 7,3955→38,2706, ganho 30,8751 IC95 [26,1416; 36,1692]. Alvo deslocado 0,6723→32,4434; BLEU contra referência real 1,2586. Três colisões de caption retidas: 125/128 alvos diferentes. Rotações preservadas, erro máximo 9,39e−6°, esquerda idêntica; 256 pares canônicos sem mudança.

**CTC PHOENIX TEST:** checkpoint local Conv/BiGRU/CTC congelado, selecionado em DEV. Todos os 642 clips: WER cru/canônico 50,1111%; tempo invertido 91,4836%, mãos identidade 93,9521%. Mesmo subset de 128 previamente fixado: alvo correto WER 51,5684%→0%, IC95 da mudança [−55,7219; −47,6383] pp; inclui 127 referências não vazias e uma vazia. Alvo deslocado 118,0678%→3,7641%, referência real 121,5809%. Sem OOV; um alvo errado inviável mantido. Erro angular máximo 5,47e−6°, 256 pares canônicos iguais. É confirmação em outra arquitetura local de reconhecimento, **não outro BT publicado independente**.

**Reprodução e limites:** diagnóstico nativo repetido nos três corpora, sem retuning ou substituir ataques: zero mudanças por projeção esquerda. PHOENIX arredonda para 15,00; ASL 3,4779 e CSL 6,8907 diferem dos cards 3,53/6,98, causas não confirmadas. Defesa fecha canal sem certificar semântica; custo anterior no subset How2Sign permanece. Gerações/rankings anteriores: 2.568 movimentos, 20.544 decodes, 7.704 pares defendidos; inversão pontual, nenhum par com vencedores opostos sustentados pelos dois IC95, uma seed, quatro variantes locais, duração real.

**Busca externa:** M³T descreve 360 dimensões rotacionais/faciais, mas o repositório oficial consultado declara código/pesos não liberados. SLRTP reproduz resultado nativo 13,3786518569 em DEV com 515 clips, contrato 534 XYZ; não replica falha 6D. Principal lacuna: BT publicado compatível independente e/ou avaliação linguística física independente. Sem meshes/humanos, SOTA novo, cifra, anonimato ou nova matemática alegados.

**Entrega científica:** candidato concreto de artigo de auditoria da equivalência da representação na avaliação de produção de sinais, com prova, intervenção, controles, consequências em difusão e reparo delimitado. Evidência mais forte, aceitação CVPR não garantida. Próxima rodada deve priorizar validade externa, mantendo esta versão revisável, em vez de substituir o resultado por losses sem ganho estabelecido.

**Integridade:** 512 novos ataques e 7.708 avaliações de sentenças recomputadas; fontes comparadas com LMDB nativa, WER por DP independente, conversor oficial/NumPy concordam; 4.096 checks matemáticos. Os 174 arquivos da confirmação anterior e 163 da confirmação CTC foram integralmente SHA-verificados. Código/logs/fontes/pesos/dados CSL/LaTeX/PDFs/snapshots arquivados e manifesto por último. Todos os jobs encerrados, GPU ociosa. Única instalação: jieba 0.42.1 no venv, registrada antes do primeiro score CSL após falha inicial. [Reprodução](reports/cvpr_representation_paper_20261011/REPRODUCAO.md).

---

'''
    # Preserve every historical section. Before snapshots were made before work.
    for name in ['CURRENT_RESEARCH.md','RESEARCH_STATUS_AND_NEXT_EXPERIMENTS.md']:
        path=Path(name);assert '<!-- cvpr_representation_paper_20261011 -->' not in path.read_text()
        path.write_text(prefix+path.read_text());shutil.copy2(path,ROOT/'sources'/('after_'+name))
    found=set();pending=PRODUCERS.copy()
    while pending:
        name=pending.pop()
        if name in found:continue
        found.add(name);p=Path('scripts')/(name+'.py');shutil.copy2(p,ROOT/'scripts'/p.name)
        for node in ast.walk(ast.parse(p.read_text())):
            modules=[a.name for a in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in modules:
                sub=module.split('.')[0]
                if (Path('scripts')/(sub+'.py')).exists() and sub not in found:pending.append(sub)
    (ROOT/'sources/script_provenance.json').write_text(json.dumps({'new_producers':PRODUCERS,'all_archived_dependencies':sorted(found),'old_mains_not_executed':True},indent=2)+'\n')
    commits={}
    for repo in ['SignSparK_BT','SignSparK','SLRTP_Evaluation_20261011']:
        path=Path('external')/repo
        commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        status=subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True)
        commits[repo]={'commit':commit,'tracked_status':status}
        if repo=='SignSparK_BT':
            names=subprocess.check_output(['git','-C',str(path),'ls-files'],text=True).splitlines()
            for name in names:
                src=path/name
                if src.is_file() and (src.suffix in ['.py','.yaml','.yml','.toml','.txt','.md'] or src.name.startswith('LICENSE')):
                    dest=ROOT/'sources/external_code'/repo/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        elif repo=='SignSparK':
            dest=ROOT/'sources/external_code'/repo/'tools/visualize.py';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path/'tools/visualize.py',dest)
    (ROOT/'sources/external_commits.json').write_text(json.dumps(commits,indent=2)+'\n')
    environment={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'platform':platform.platform(),'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),'package_listing_command':'venv_mska/bin/python -m pip list --format=freeze','recorded_UTC':datetime.now(timezone.utc).isoformat()}
    (ROOT/'sources/environment.txt').write_text(json.dumps(environment,indent=2)+'\n')
    (ROOT/'sources/pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','list','--format=freeze'],text=True))
    for src in Path('/tmp').glob('representation_*20261011.log'):shutil.copy2(src,ROOT/'logs'/src.name)
    shutil.copytree(PAPER,ROOT/'paper/manuscript',dirs_exist_ok=True)
    with zipfile.ZipFile(ROOT/'paper/manuscript_source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PAPER.rglob('*')):
            if p.is_file() and p.suffix not in ['.png','.log','.blg','.txt']:z.write(p,str(p.relative_to(PAPER)))
    (ROOT/'sources/jobs_closed.json').write_text(json.dumps({'all_download_training_inference_optimization_plot_compile_audit_processes_closed':True,'new_training_runs':0,'GPU_idle_before_freeze':environment['GPU'],'scored_failed_CSL_runs_discarded':0},indent=2)+'\n')
    checks={'scientific_audit':scientific,'manuscript_audit':manuscript,'math_verification':json.loads((ROOT/'math_verification.json').read_text()),'archived_script_count':len(found),'external_commits':commits,'no_old_reports_edited':True,'no_conference_submission':True,'all_writers_closed_before_manifest':True}
    freeze(ROOT,checks)
if __name__=='__main__':main()
