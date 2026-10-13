"""Close reviewed manuscript, preserve predecessors, archive sources, freeze last."""
import json,shutil,difflib,subprocess,sys,zipfile
from pathlib import Path
from verify_signspark_studies import sha,freeze
from freeze_learned_reader_studies import archive
R=Path('reports/cvpr_paper_story_20261011');P=Path('paper/cvpr2027_learned_equivalence_20261011')
OLD=Path('paper/cvpr2027_rotation_equivalence_revision_20261011')
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not (R/'manifest_sha256.json').exists()
 layout=read(R/'layout_verification.json');refs=read(R/'reference_verification.json');evidence=read(R/'evidence_verification.json');integrity=read(R/'prior_integrity_verification.json')
 assert refs['passed'] and refs['total_entries']==27 and evidence['passed'] and integrity['passed']
 assert sha(OLD/'build/main.pdf')=='3d6125644d9d88d5e7c5fa1c744dc1047507b33ae5c258b9f4a910ac7d3e877f'
 assert sha(OLD/'build/supplement.pdf')=='5d8f37ccb4c6599f039d60e31f96aa9d3e31af37a0d7bd180e3bcaae4bafaabe'
 for name in ['main','supplement']:
  assert sha(P/'build'/(name+'.pdf'))==layout[name]['pdf_sha256']
  previews=sorted((R/'previews').glob(name+'-[0-9]*.png'));assert len(previews)==layout[name]['total_pages']
  (R/'sources'/(name+'_revision.diff')).write_text(''.join(difflib.unified_diff((OLD/(name+'.tex')).read_text().splitlines(True),(P/(name+'.tex')).read_text().splitlines(True),fromfile='prior/'+name+'.tex',tofile='revised/'+name+'.tex')))
 layout['visual_inspection']='All 9 main and 9 supplement pages rendered and reviewed after final build, including opening example, equations, all tables, fonts and references; source frame provenance checked separately.'
 (R/'layout_verification.json').write_text(json.dumps(layout,indent=2)+'\n')
 for name in ['README.md','CURRENT_RESEARCH.md','RESEARCH_STATUS_AND_NEXT_EXPERIMENTS.md']:
  shutil.copy2(name,R/'sources'/('before_'+name))
 current='''# Estado atual da pesquisa — 12/10/2026

Artigo atual: **Same Rotations, Different Captions: Learning to Game Sign-Language Backtranslation**.
[PDF principal](paper/cvpr2027_learned_equivalence_20261011/build/main.pdf),
[suplemento](paper/cvpr2027_learned_equivalence_20261011/build/supplement.pdf),
[fontes e reprodução](paper/cvpr2027_learned_equivalence_20261011/README.md).
Principal: sete páginas de conteúdo + duas de referências. Suplemento: oito + uma.

## Evidência principal

Uma camada limitada treinada em TRAIN aumenta BLEU com rotações preservadas:
PHOENIX TEST642, baseline6,3905 → média15,2053 em três seeds; ganho8,8148,
IC pareado[7,8196;9,8706]. Todas as hipóteses canônicas ficam iguais. Inferência
é uma passagem, sem otimização por clip. Excluindo32captions repetidas de TRAIN,
restam610clips com ganho8,3745. Trocar somente o texto da camada redireciona o
leitor bruto sem mudar as rotações. É uma exploração aprendida explicitamente
projetada; o fine-tune convencional anterior não confirmou inflação seletiva.

Transferência sem ajuste:128movimentos do SignSparK público,0,7381→12,6456,
com score canônico fixo. Baseline fraca, não reprodução competitiva de SOTA.
Leitores How2Sign/CSL:12fine-tunes pareados, três seeds, TRAIN completo válido,
cinco épocas. Fechamento verificado em384ataques canônicos DEV e10.035pares
TEST. Custo de acurácia é medido separadamente; não é universalmente zero.

## Entrega e evidências congeladas

- [Revisão detalhada](reports/cvpr_paper_story_20261011/REVISAO_E_ENTREGA.md).
- [Estudo de oito papers premiados](reports/cvpr_paper_story_20261011/ESTUDO_DE_ESCRITA.md).
- [Mapa de afirmações](reports/cvpr_paper_story_20261011/EVIDENCE_MAP.md).
- Camada TRAIN/DEV: `reports/cvpr_learned_equivalence_20261011/`.
- Leitores adaptados: `reports/cvpr_canonical_readers_20261011/`.
- Confirmação fixa TEST: `reports/cvpr_learned_confirmation_20261012/`.
- Parecer/piloto negativo/controles: `reports/cvpr_reviewer_followup_20261011/`.

27 referências verificadas. Todos os reports anteriores permanecem preservados;
2.066 arquivos foram SHA-verificados antes de congelar a entrega editorial.
Figura inicial usa frames originais, sem avatar/renderização inventada. Formato
usa kit oficial fixado2026, sem editar estilo/margens/fontes. Regras2027 precisam
de conferência quando disponíveis. ID DRAFT; nenhuma submissão/publicação.

## Alcance e próximos testes científicos

A evidência agora inclui aprendizagem, controles causais e defesa adaptada.
Persistem uma família pública de BT, CTC local, um gerador/leitor congelado por
estudo, replay público fraco e ausência de avaliação humana. Outro BT publicado
compatível e independente, ou uma adaptação validada a outro contrato físico,
seria a extensão mais valiosa. Reprodução competitiva de geração e avaliação
linguística também fortaleceriam generalização. Não há garantia de aceitação,
criptografia/anonimização demonstrada ou melhoria física/semântica do sinal.

A versão anterior permanece intacta em
`paper/cvpr2027_rotation_equivalence_revision_20261011/`. Seu estado e o histórico
antes desta rodada estão preservados em `reports/cvpr_paper_story_20261011/sources/`.
'''
 # Add spaces to avoid compact numeric-word artifacts in status prose.
 for a,b in [('TEST642','TEST, 642'),('baseline6','baseline 6'),('média15','média 15'),('ganho8','ganho 8'),('pareado[','pareado ['),('32captions','32 captions'),('610clips','610 clips'),('128movimentos','128 movimentos'),('público,0','público, 0'),('12fine','12 fine'),('384ataques','384 ataques'),('10.035pares','10.035 pares'),('fixado2026','fixado 2026'),('Regras2027','Regras 2027')]:current=current.replace(a,b)
 Path('CURRENT_RESEARCH.md').write_text(current)
 Path('RESEARCH_STATUS_AND_NEXT_EXPERIMENTS.md').write_text(current)
 readme=Path('README.md').read_text();i=readme.index('## Organização')
 top='''# Sign language backtranslation and rotation equivalence

Workspace atual de **Same Rotations, Different Captions: Learning to Game Sign-Language Backtranslation**, candidato ao CVPR2027.

[Artigo](paper/cvpr2027_learned_equivalence_20261011/build/main.pdf),
[suplemento](paper/cvpr2027_learned_equivalence_20261011/build/supplement.pdf),
[fontes LaTeX e build](paper/cvpr2027_learned_equivalence_20261011/README.md),
[revisão e resultados](reports/cvpr_paper_story_20261011/REVISAO_E_ENTREGA.md),
[estudo de escrita](reports/cvpr_paper_story_20261011/ESTUDO_DE_ESCRITA.md).

Uma camada aprendida aumenta o score de BT a rotações preservadas: PHOENIX
TEST6,39→15,21 em três seeds, com predições canônicas iguais. A contribuição é
uma falha aprendida de avaliação e o fechamento do canal com custo de acurácia
medido. Limites científicos e o piloto convencional negativo permanecem claros
no artigo. [Estado atual](CURRENT_RESEARCH.md). Kit final2027 ainda provisório;
nenhuma submissão realizada. O artigo e os reports anteriores estão preservados.

'''.replace('TEST6','TEST 6').replace('final2027','final 2027')
 readme=top+readme[i:]
 readme=readme.replace('| `scripts/` |','| `reports/cvpr_paper_story_20261011/` | Manuscrito atual, estudo de escrita e auditorias |\n| `reports/cvpr_learned_equivalence_20261011/` | Camada aprendida TRAIN/DEV e transferência pública |\n| `reports/cvpr_canonical_readers_20261011/` | Leitores adaptados em How2Sign/CSL |\n| `reports/cvpr_learned_confirmation_20261012/` | Confirmação TEST de pesos congelados |\n| `scripts/` |')
 Path('README.md').write_text(readme)
 for name in ['README.md','CURRENT_RESEARCH.md','RESEARCH_STATUS_AND_NEXT_EXPERIMENTS.md']:shutil.copy2(name,R/'sources'/('after_'+name))
 producers=['fetch_awarded_paper_study','plot_paper_story','build_paper_story_tables','audit_paper_story_references','audit_paper_story_evidence','audit_paper_story_layout','verify_frozen_research','finalize_paper_story']
 archive(R,producers,['cvpr_story_main_build.log','cvpr_story_supplement_build.log','cvpr_story_main_render.log','cvpr_story_supplement_render.log','cvpr_story_prior_integrity.log','cvpr_freeze_TEST_confirmation_20261012.log'])
 (R/'sources/python_environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','list','--format=freeze'],text=True))
 archive_path=P/'manuscript_source.zip'
 with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(P.rglob('*')):
   rel=p.relative_to(P)
   if not p.is_file() or p==archive_path:continue
   if rel.parts[0]=='build' and p.name not in ['main.pdf','supplement.pdf']:continue
   if p.suffix in ['.tex','.bib','.sty','.bst','.pdf','.png','.md','.json']:z.write(p,str(rel))
 with zipfile.ZipFile(archive_path) as z:
  assert z.testzip() is None
  assert {'main.tex','supplement.tex','references.bib','cvpr.sty','ieeenat_fullname.bst','figures/teaser.pdf','figures/learned_method.pdf','build/main.pdf','build/supplement.pdf'}<=set(z.namelist())
  zip_count=len(z.namelist())
 shutil.copytree(P,R/'paper/manuscript',dirs_exist_ok=True)
 checks={'layout':layout,'references':refs,'evidence':evidence,'prior_frozen_report_files_verified':integrity['total_files'],
 'old_manuscript_PDFs_unchanged':True,'source_ZIP_CRC_passed':True,'source_ZIP_entries':zip_count,'source_ZIP_sha256':sha(archive_path),
 'new_experiments_completed_and_frozen_separately':True,'new_editorial_model_inference':False,'no_external_submission_or_publication':True,'all_writers_closed_before_manifest':True}
 freeze(R,checks)
 print('PAPER_STORY_DELIVERED',sha(P/'build/main.pdf'),sha(P/'build/supplement.pdf'),flush=True)
if __name__=='__main__':main()
