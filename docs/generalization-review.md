# Generalização e revisão do paper — 13/10/2026

A [revisão editorial posterior](editorial-review.md) reorganiza esta evidência no
mesmo `paper/current/`. As contagens de páginas abaixo descrevem a entrega
científica original desta rodada; os PDFs atuais incorporam a revisão editorial.

A rodada acrescentou outra arquitetura de avaliador, calibração física coerente,
transferência sem gradientes do leitor de destino e um segundo dataset de movimento.
A afirmação é **inconsistência de representação**. Exploração espontânea em treino
bem-sucedido permanece sem confirmação.

O manuscrito único é [paper/current/](../paper/current/README.md). A rodada completou todos os endpoints, incluindo MotionDiffuse nos dois
datasets. A entrega revisada fica em `reviewed/`; compilações locais ficam
em `.build/paper/`. Snapshots congelados são arquivos de evidência, sem criar
outra pasta de trabalho.

| Pedido do parecer | Evidência concluída | Limite ou pendência |
|---|---|---|
| Piloto de recompensa válido | Oito fits DEV: dois learning rates, MSE e três pesos por taxa; seleção verificada independentemente | Há pequenos ganhos abaixo do limiar; nenhum satisfaz o critério conjunto ou avança para TEST. Não demonstra ausência de exploração em um treino bem-sucedido. |
| Outra arquitetura | TMR ACTOR/Transformer, pesos oficiais/DistilBERT; 20 draws de seis fontes e sete condições | Coortes compatíveis com nossa auditoria; não é o protocolo completo de anotações/filtros TMR |
| Calibração física | 20 draws completos, seis doses angulares, eixos fixos/independentes por frame; FK, posições, velocidades e contatos atualizados | Referência reconstruída pareada, com seu custo explícito; sem conversão universal FID–graus |
| Mais releases e KIT | MoMask/DiP/MotionDiffuse HumanML e MoMask/MotionDiffuse KIT: 20 repetições cada | DiP recebe prefixo de 20 frames. MotionDiffuse completou 20 por dataset e passou na verificação. Checkpoints aparentados não contam como arquiteturas independentes. |
| Ataque cego e transferência | MLD além de MDM; outputs selados antes da pontuação por outro checkpoint Guo, sem seus gradientes | Os checkpoints compartilham arquitetura; cohorts MDM têm sobreposição de grupos |
| Custo da defesa | 20 draws reais nos dois Guo, TMR e KIT: R1 bruto/canônico idêntico | Custo observado nessa distribuição; sem teorema de acurácia grátis |

## Resultados e interpretação

No TMR, o ataque direcionado leva R1 de **0,6328 para 0,9922** no MDM e de
**0,6953 para 1,0000** no movimento real, em 128 clipes fixos por fonte.
Canonicalizar o ataque recupera os R1 originais. Alvos trocados e gradiente
canônico são controles separados. Um ataque genérico com orçamento L2 igual
também funciona, mas altera posições/rotações e sobrevive à canonicalização:
a defesa fecha o canal 6D, não todos os ataques.

Na auditoria natural TMR, apenas MotionLCM e T2M têm intervalos do gap R1 que
excluem zero na família de doze endpoints. Os efeitos são pequenos. Cosseno é
outro endpoint, não confirmação de toda diferença de R1.

O ataque cego MLD, bound 0,1, muda FID **0,7064 → 0,2886** no leitor de origem
e **0,6594 → 0,5786** no destino. Bound 0,35 piora FID em ambos. Nos três cohorts
MDM, o destino apresenta reduções de 18,2–25,1% no orçamento menor; o maior
também piora todos. Todas as condições ficam no suplemento.

| Release/dataset, 20 draws | FID bruto | Canônico | Bruto − canônico | IC ajustado, dez endpoints |
|---|---:|---:|---:|---|
| MoMask / HumanML3D | 0,067702 | 0,064398 | +0,003304 | [+0,002789; +0,003820] |
| DiP com prefixo / HumanML3D | 0,366507 | 0,364970 | +0,001537 | [+0,000044; +0,003031] |
| MotionDiffuse / HumanML3D | 0,976604 | 0,980670 | −0,004067 | [−0,007371; −0,000762] |
| MoMask / KIT | 0,351156 | 0,364873 | −0,013717 | [−0,024730; −0,002704] |
| MotionDiffuse / KIT | 1,779444 | 1,815863 | −0,036419 | [−0,042817; −0,030020] |

No MotionDiffuse KIT, as 20 repetições dão FID 1,779444 bruto e 1,815863
canônico: gap −0,036419, IC ajustado [−0,042817; −0,030020]. O gap R1
é inconclusivo. A verificação cobre 14.080 pares geométricos e 40 casos FK;
resíduos KIT permanecem em unidades nativas e relativos à altura. O estudo completo está congelado, incluindo o endpoint HumanML.

Os dois endpoints MotionDiffuse integram essa família prospectiva de dez FID/R1
comparações. Todas as 20 repetições de cada endpoint estão incluídas; tentativas incompletas
e logs de implementação são preservados sem entrar nas médias científicas. A inferência mantém mil
passos DDPM nativos, pesos e seeds. Compilação float32 passa pela paridade
registrada antes dos resultados. Reiniciar preserva repetições salvas.

A calibração pontua o custo original → reconstrução zero: FID **0,0201**, R1
**0,4510 → 0,4474**. Na referência reconstruída, recodificações ±0,35 dão FID
1,972/3,574 sem mudança física. Esses níveis ficam entre doses de 1–3 graus com
ruído independente por frame e 10–30 graus com eixos fixos. São brackets dos
pontos medidos, sem interpolação nova ou afirmação perceptual.

## Antecedentes e escala

[MARDM, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Meng_Rethinking_Diffusion_for_Text-Driven_Human_Motion_Generation_Redundant_Representations_Evaluation_CVPR_2025_paper.pdf)
já diagnostica viés por features redundantes e propõe leitores com features
essenciais. Não reivindicamos o primeiro diagnóstico. Nossa distinção é mudar
a codificação preservando **posições e rotações**, e fechar o canal mantendo
informação rotacional. O texto credita essa relação em todas as seções pertinentes.

[TMR, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Petrovich_TMR_Text-to-Motion_Retrieval_Using_Contrastive_3D_Human_Motion_Synthesis_ICCV_2023_paper.pdf)
fornece a arquitetura diferente. Seu KIT AMASS de 263 features não é compatível
com KIT MMM de 251 features; não foi aplicado por adaptação artificial. A escala
de coordenadas KIT é outra: resíduos FK são reportados em unidades nativas e
relativos à altura, sem presumir metros nem reescalar inputs antes do leitor.

Margens publicadas usam as linhas corretas: T2M estimando comprimento
(1,087 ± 0,021), T2M-GPT com corrupção uniforme (0,141 ± 0,005), MLD
(0,473 ± 0,013), MotionLCM um passo (0,467 ± 0,012) e MDM50 do
[resultado corrigido pelos autores](https://github.com/GuyTevet/motion-diffusion-model/blob/main/assets/fixed_results.tex)
(0,489 ± 0,047). Fontes, revisão e fingerprints constam dos artefatos.
Barras de erro entre papers não são usadas como teste pareado de significância.

## Fechamento

Os nove estudos concluídos passaram por verificações independentes de métricas,
geometria, inputs e fontes. Estão congelados com inventários SHA256. Relatórios
antigos permanecem imutáveis; logs e revisões de verificadores ficam nos artefatos.

O estado local dos workers está em
`reports/cvpr_generalization_review_20261013/pending_motiondiffuse_status.json`.
O coordenador completou a verificação/congelamento do estudo, a inclusão de
todos os resultados nas tabelas e a compilação. A entrega revisada contém sete
páginas de conteúdo e duas de referências; o suplemento tem dezenove mais duas.
Todas as 30 páginas passaram pela inspeção visual, com verificação de layout,
referências, fontes incorporadas e fingerprints. Não houve publicação/submissão.

Autores, perfis e leitores humanos ficam para depois, conforme solicitado.
[Datas oficiais CVPR 2027](https://cvpr.thecvf.com/Conferences/2027/Dates): registro
10/11, paper 16/11, suplemento 23/11, AoE. O kit local continua o oficial de 2026;
a conformidade final será rechecada quando as instruções completas 2027 saírem.
