# Fechamento do parecer pós-MARDM

Os quatro itens da lista fechada foram concluídos. O manuscrito permanece em
[paper/current](../paper/current/README.md), com [paper](../paper/current/reviewed/main.pdf)
e [suplemento](../paper/current/reviewed/supplement.pdf) no mesmo local.
Não houve treino, ajuste de recompensa, novas seeds de modelos ou novos geradores.

## O reparo principal muda a interpretação

A crítica à comparação entre direção fixa e ruído aleatório procede. Orçamento
igual não controla a estrutura temporal ou direção. Os testes adicionais usam
as mesmas vinte coortes reais, 4.640 entradas por repetição, em Guo e TMR.
Uma direção local fixa é repetida em todas as articulações e frames válidos,
com ambos os sinais e os dois orçamentos. O escalar depende somente da norma
na entrada; não há busca por scores ou gradientes do leitor.

| Bloco, kappa=3; Guo ΔFID | Aleatório | Fixo + | Fixo − |
|---|---:|---:|---:|
| Fibra de rotação | 0,00238 | 0,08230 | 0,11688 |
| Rotações alteradas | 0,00349 | 0,01061 | 0,01512 |
| Velocidades | 0,01586 | 0,06033 | 0,47604 |
| Contatos | 0 | 0 | 0 |
| Posições | 0,01826 | 0,34160 | 1,18896 |

Os deslocamentos fixos de posições e velocidades superam o maior efeito nulo.
A Figura 3 mostra os três tipos de direção em cada bloco. O texto agora inclui
explicitamente que ruído aleatório de posições/velocidades também supera a
fibra aleatória. **Não se atribui uma preferência geral por redundância ao
avaliador. A particularidade da fibra é preservar rotações e todos os demais
canais, enquanto muda a codificação.**

Constância temporal não implica um vetor completo fisicamente consistente:
nesses controles, alterar posições não atualiza velocidades, e alterar rotações
não atualiza posições. A calibração angular coerente anterior é separada.
As direções são controles específicos, sem provar a maior sensibilidade possível
em qualquer bloco ou direção. Não se faz inferência de importância semântica.

O TMR confirma sensibilidade dependente do bloco e direção. O offset fixo
negativo de contatos reduz R1 em 2,316 pontos percentuais, com IC primário ajustado
[-2,640; -1,991]. Contatos podem sair do domínio binário; Guo não os lê, com
embeddings bitwise idênticos. A fibra fixa negativa muda R1 TMR em cerca de
-0,232 ponto percentual na família histórica. Essa comparação é descritiva,
sem teste prospectivo entre contato fixo e fibra fixa.

Os dois leitores têm normalizações próprias. O orçamento igual por clipe
é anterior à seleção de features e não é uma unidade física intercambiável
entre leitores. As tolerâncias numéricas float32 são explicitadas; o maior erro
relativo observado foi 7,84e-6, no controle de contatos do Guo.

Cada leitor mantém dezesseis condições por repetição. Suas famílias primárias
ajustam 32 endpoints; os contrastes pareados fixo menos aleatório usam uma
família secundária de 32 endpoints. A família original de 28 endpoints permanece
intacta. Ambos os sinais, ambos os raios e todos os contrastes estão nas tabelas
completas do suplemento e em [Guo](../results/cvpr_coherent_blocks_20261014/guo/analysis.json)
e [TMR](../results/cvpr_coherent_blocks_20261014/tmr/analysis.json).

## A dependência natural em sinais voltou ao corpo

Recontamos todos os 3.987 pares de hipóteses salvas e recomputamos os seis BLEUs
com contagem independente de n-gramas e penalidade de brevidade. Não houve novo
decode, treino, camada ou seleção de subconjunto.

| Entrada real | Hipóteses alteradas | BLEU bruto | BLEU canônico |
|---|---:|---:|---:|
| PHOENIX | 0/642 | 14,9995 | 14,9995 |
| How2Sign | 1.438/2.183 (65,9%) | 3,4779 | 3,4840 |
| CSL | 464/1.162 (39,9%) | 6,8907 | 7,0662 |

Esses resultados aparecem no abstract, motivação, Seção 5.5, última linha da
Tabela 3 e numa tabela própria no suplemento. Eles sustentam dependência natural
num domínio só de rotações, sem acesso de uma camada à legenda. Pequena mudança
no BLEU agregado não implica invariância das frases decodificadas. As mudanças
não demonstram melhora ou piora semântica, nem custo zero de canonicalização.

Há uma precisão importante: ambos os caminhos compartilham o pré-processamento
nativo original da mão esquerda antes do decode. “Sem ataque” descreve os inputs
reais desse protocolo; não significa bytes intocados da release. O CSL usa
caracteres e os demais corpora usam palavras, preservando seus protocolos.
O [registro verificado](../results/cvpr_sign_natural_restoration_20261014/analysis.json)
guarda as estatísticas suficientes e os hashes dos arquivos originais.

## Notação, escrita e validade

`rho` permanece a escala positiva da fibra; o orçamento se chama `kappa` no
corpo, suplemento, tabelas e figuras. O abstract agora inclui a evidência real
sem alvo, separada dos ataques source-conditioned. Ele tem cinco números e
mantém explícita a transferência entre checkpoints da mesma arquitetura.

A novidade permanece a equivalência exata e a defesa que retém rotações, em vez
de uma nova matemática do conversor ou da alegação geral de sensibilidade à
redundância já estudada pelo MARDM. Não se muda a interpretação do ranking
MARDM, dos efeitos naturais pequenos nos releases ou dos negativos de treino.
Não usamos a porcentagem subjetiva do parecer como uma probabilidade validada
de aceitação. Os reparos tornam a comparação e a argumentação mais defensáveis.

## Verificação e entrega

As verificações independentes recalculam 680 células de métricas e todos os
intervalos primários/secundários, conferem todas as normas salvas e reconstroem
5.120 fixtures por fórmulas independentes e SciPy. Forwards nativos, rotações,
proteção de canais e padding passam nas checagens. Não se afirma replay geométrico
completo de arrays não salvos. Os estudos foram congelados; 2.153 arquivos novos
tiveram seus SHA256 conferidos. Os 56 inventários históricos permanecem iguais.

O corpo tem oito páginas e duas de referências; o suplemento tem vinte e seis
páginas de conteúdo e duas de referências. A revisão inclui fontes incorporadas,
referências, margens, integridade e as 38 páginas renderizadas. O código portátil
anônimo permanece igual. A revisão visual do agente não substitui leitura fria
humana. Autores, registro e leitura fria ficam com o usuário, conforme a decisão
anterior; não houve commit, contato, publicação ou submissão.
