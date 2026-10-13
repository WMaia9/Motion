# Resultado da nova rodada

Executamos **79 novos ajustes de modelos**, 28 condições de confirmação TEST
com pesos congelados, diagnóstico de transplante de coordenadas e replay público
com recuperação de sinais de TRAIN. As versões anteriores e seus resultados
foram preservados. A pesquisa leu e arquivou quatorze fontes primárias; o paper
passou a citar 36 referências pertinentes.

A crítica do professor foi tratada experimentalmente. A cobertura em outras
línguas e o orçamento da camada melhoraram. A objeção ao treino convencional
**foi parcialmente respondida**, mas não devemos anunciar exploração espontânea
inequívoca: dois controles importantes ficaram inconclusivos no TEST.

## O teste decisivo: denoiser comum, sem camada de ataque

42 fits cruzam três seeds, MSE/geodésica, CE bruta/canônica, quatro pesos e controle
sem CE. Todas as condições chegam a cinco épocas; geodésica sem CE, CE bruta1 e
CE canônica1 chegam ao endpoint declarado de vinte. Estes nove modelos foram
congelados antes da confirmação. Entre modelos, a rotação física pode mudar;
a comparação bruto/canônico de uma mesma saída preserva a rotação.

| PHOENIX TEST, 642 clips | BLEU bruto | BLEU canônico | Bruto − canônico |
|---|---:|---:|---:|
| Geodésica sem CE | 2,64 | 5,10 | −2,45 |
| Geodésica + CE bruta, peso1 | 5,13 | 4,79 | +0,35 |
| Geodésica + CE canônica, peso1 | 3,41 | 3,55 | −0,14 |

São médias de três seeds, não um checkpoint escolhido pelo melhor BLEU.
Excesso de ganho bruto sobre o controle sem CE: **+2,80**, IC pareado
**[1,85;3,79]**, IC seeds+clips [1,63;3,94]. O ganho bruto é +2,49 e o canônico
−0,31. Entretanto:

- gap bruto–canônico absoluto: +0,35, IC [−0,29;0,96];
- contraste de gaps contra CE canônica: +0,48, IC [−0,31;1,26];
- duas seeds não superam seu próprio canônico;
- transplante das coordenadas redundantes de outra legenda, mantendo rotações,
  ganha só +0,1644 no acordo com a legenda errada, IC [−0,0524;0,4994].

DEV foi mais favorável: excesso +2,95, IC [1,95;4,01]; os três contrastes
primários passaram os intervalos ajustados em família. Não trocamos a conclusão
do TEST pela de DEV. Parte do contraste positivo pode vir do desvio de
normalização do controle sem CE. Também há diferença de unidades/gradientes:
na primeira minibatch da seed42, a razão das normas dos gradientes na saída
CE/movimento em peso1 é 214,4 na geodésica contra 27,5 na MSE. O sweep completo
não prova causalidade exclusiva dos graus redundantes versus força da CE.

**Interpretação sustentada:** mudar a supervisão de movimento afeta a avaliação
dependente da representação em treino comum. **Não sustentado:** transmissão
semântica robusta ou exploração habitual por geradores competitivos.

Fontes: [DEV](../results/cvpr_quotient_training_20261012/analysis.json),
[TEST](../results/cvpr_second_review_confirmation_20261012/analysis.json),
[transplante](../results/cvpr_ordinary_transplant_20261012/analysis.json).

## Camada aprendida sobre movimento real: dois novos corpora

| TEST completo | Baseline bruto | Camada bruta, média3seeds | Canônico fixo | Ganho e IC pareado |
|---|---:|---:|---:|---|
| How2Sign, 2.183 clips /149 VIDEO_ID | 3,48 | 6,11 | 3,48 | +2,64 [2,27;3,03] |
| CSL-Daily, 1.162 clips | 6,89 | 7,43 | 7,07 | +0,54 [0,14;0,97] |

Os intervalos com seeds+unidades também permanecem positivos. Predições
canônicas idênticas em todos os modelos. A camada recebe texto-fonte; isto
é uma réplica do ataque amortizado, não exploração espontânea. O encoder textual
foi adaptado ao vocabulário nativo, com embeddings do leitor congelados e
769.784 parâmetros treináveis. Controles de movimento/loss canônica usam uma
seed cada, explicitamente identificada. Não houve transferência entre línguas.

## Orçamento menor e detector simples

| Bound | Ganho TEST | IC pareado | IC seeds+clips | RMS ao canônico | AUC detector RMS |
|---|---:|---|---|---:|---|
| 0,01 | +0,05 | [−0,20;0,29] | [−0,22;0,32] | 0,0056 | 1,00 |
| 0,05 | +0,85 | [0,48;1,23] | [−0,0055;1,68] | 0,0274 | 1,00 |
| 0,10 | +1,65 | [1,15;2,16] | [0,32;2,81] | 0,0548 | 0,917–0,941 |

O gerador bruto natural tem RMS médio0,1018. Menor distância não significa
que a saída fique escondida: o detector pode separar valores abaixo do natural.
A seed777 é mais fraca. A sensibilidade de seeds em bound0,05 inclui zero.
Não há alegação de anonimização, criptografia ou segurança.

## Matemática aplicada: controles de distância e norma constantes

Derivamos controles auxiliares sobre s=(ρ−1,σ−1,τ): ||s||² é a distância
quadrática à representação canônica. Rodar s preserva a distância sem rodar
as juntas. Um eixo (1,1,0) também preserva a norma6D total. Registramos,
implementamos e treinamos ambos com três seeds e controles.

- Círculo, distância+norma fixas: ganho DEV +0,0649, IC [−0,3860;0,5297].
- Esfera, distância fixa: ganho DEV −0,0525, IC [−0,4955;0,3909].

RMS detector AUC≈0,50, mas **nenhum ataque aprendido positivo confiável**.
São limites experimentais, não uma novidade criptográfica validada. A extensão
esférica foi declaradamente exploratória após o negativo do círculo. Todas as
condições estão no suplemento; nenhuma foi descartada.

## Transfer público e comparação com movimento real

Corrigimos o contrato do replay para CSL: glosses preditas públicas, dicionário
somente TRAIN e keyframes de sinais recuperados de TRAIN. BLEU bruto/canônico
**9,32/9,12** em128clips DEV. Camadas CSL transferidas sem fitting dão ganho
**+0,15**, IC [−0,86;1,15]: transferência inconclusiva. Diferenças do pipeline
publicado e seed efetiva102 estão documentadas. A tabela do paper original
não oferece um número BT diretamente comparável para esta tarefa.

O replay PHOENIX de0,74 saiu do corpo principal e ficou no suplemento,
com a limitação explícita de que o checkpoint divulgado não foi treinado
em PHOENIX. Não anunciamos reprodução SOTA.

Também verificamos a frase sugerida sobre movimento real: 15,2053 versus
14,9995 é diferença +0,2058, **IC [−0,8454;1,2357]**. O abstract destaca
os valores como comparáveis em escala, sem afirmar superioridade estatística
ou qualidade de sinais melhor que a real.

## Apresentação e integridade

- Abstract/introduction/conclusion refletem os testes e seus limites.
- Antecedentes de ataques universais, redes de perturbação e reward hacking foram adicionados.
- Figura de barras sem anotação sobre barra/whisker; orçamento aprendido e por clip no corpo.
- Suplemento reescrito como documento único, com protocolos, provas e negativos.
- Resíduos editoriais de histórico de auditoria/submissão foram removidos; o uso anterior dos splits públicos TEST continua honestamente declarado.
- Rotação, métricas, pesos registrados e fontes foram conferidos independentemente.
- 28 condições TEST têm28.281pares de rotação verificados, erro máximo0,0001335grau e zero mudanças canônicas nas camadas.
- O verificador global conferiu3.183arquivos congelados antes de congelar esta entrega.

## Situação científica para CVPR

A demonstração de manipulação aprendida agora tem confirmação em três corpora,
orçamento/detecção e uma análise de treino comum bem mais informativa. Isso
melhora a submissão. Ainda não equivale a um accept claro: a família de leitores
continua única, dois leitores são fracos, o treino comum não passa todos os
controles TEST e o transfer nativo não é positivo. Uma revisão honesta precisa
preservar estes limites. A alegação mais forte continua sendo o ataque
condicionado em uma classe de equivalência física e a sua prevenção pelo
contrato do avaliador.

A próxima evidência de maior valor é um avaliador forte e independente que
consuma uma representação compatível, ou uma validação humana da utilização
da métrica. Repetir seeds do mesmo leitor não resolveria essa objeção. A pesquisa
sobre difusão intrínseca/quocientes oferece outro projeto metodológico, mas
não foi implementada nesta rodada e não será vendida como resultado.
