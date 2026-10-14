# Controle final: estabilidade em sinais e comparação dentro das rotações

O controle crítico foi concluído nos mesmos 2.183 clipes How2Sign e 1.162 CSL,
com pesos, beam, ordem, cortes e pré-processamento nativos. As novas execuções
brutas e canônicas reproduzem exatamente todas as hipóteses anteriores.
O manuscrito continua em [paper/current](../paper/current/README.md), com
[paper](../paper/current/reviewed/main.pdf) e
[suplemento](../paper/current/reviewed/supplement.pdf) no mesmo local.

## O resultado é positivo contra as Gaussianas testadas

Cada campo é reescalado à norma do deslocamento canônico de seu próprio clipe,
sem substituir esse orçamento pelo RMS médio. O controle primário preserva
os mesmos canais de mão esquerda e perturba corpo/mão direita; retêm-se três
campos previamente definidos. A tabela mostra a média dos campos por clipe.

| Corpus / condição | Hipóteses alteradas | Edição média em tokens | Edição normalizada média |
|---|---:|---:|---:|
| How2Sign, canônico | 65,9% | 8,121 | 0,397 |
| How2Sign, Gaussianas de norma igual | 41,9% | 4,994 | 0,243 |
| How2Sign, físico de norma igual | 70,3% | 8,634 | 0,426 |
| CSL, canônico | 39,9% | 5,167 | 0,304 |
| CSL, Gaussianas de norma igual | 22,3% | 2,918 | 0,168 |
| CSL, físico de norma igual | 44,3% | 5,923 | 0,346 |

How2Sign usa palavras; CSL usa caracteres, preservando pontuação. A edição
normalizada divide pelo maior comprimento das duas saídas, com mínimo um.
Ela compara as saídas, sem utilizar a legenda como alvo da intervenção.
Frases vazias não são removidas.

Os contrastes canônico menos Gaussiana são positivos nos dois endpoints:

| Corpus | Excesso de hipóteses alteradas, pontos percentuais | Excesso de edição normalizada |
|---|---|---|
| How2Sign | +23,96 [21,05; 26,74] | +0,154 [0,133; 0,174] |
| CSL | +17,58 [13,91; 21,40] | +0,136 [0,106; 0,167] |

Esses intervalos usam 20.000 bootstraps pareados e ajuste de oito endpoints,
com 149 VIDEO_IDs oficiais no How2Sign e clipes no CSL. A família primária de
vinte endpoints é separada. Os campos individuais permanecem no suplemento e
no [registro completo](../results/cvpr_sign_stability_20261014/analysis.json).
Não se afirma que os três campos sejam datasets ou réplicas independentes de treino.

## A leitura precisa de uma ressalva que o parecer não antecipou

A mudança física de norma igual causa alterações médias ainda maiores.
O contraste de frações How2Sign é negativo com intervalo abaixo de zero; no
CSL, esse intervalo inclui zero. A distância normalizada é maior no controle
físico nos dois corpora, com intervalos ajustados negativos para canônico menos
físico. Portanto, a comparação com Gaussianas não estabelece que toda a
sensibilidade do leitor seja exclusiva da codificação. Tampouco demonstra
que as alterações físicas sejam linguisticamente irrelevantes.

O controle físico mantém normas e cisalhamento, rodando os dois vetores de
cada articulação juntos. Isso evita somar canonicalização à mudança física.
Com norma igual, seus ângulos médios são 2,623 graus no How2Sign e 5,293 no CSL;
os máximos são 5,470 e 9,756 graus. Todos os clipes alcançam o orçamento.

Também executamos exatamente um grau: ele altera 45,4%/13,9% das hipóteses,
com edições normalizadas 0,252/0,102. Esse controle tem norma diferente e é
reportado separadamente. A Gaussiana secundária nos 240 canais altera
37,2%/20,5%, com edições 0,220/0,156. Todos os oito caminhos e seus BLEUs
permanecem nas tabelas; nenhum resultado é removido por ser menos favorável.

O abstract e o corpo incorporam o efeito maior da canonicalização em relação
às Gaussianas, junto com a resposta aos controles físicos. A contribuição
continua sendo equivalência exata e uma exigência de invariância, separada de
acurácia e qualidade linguística. Leitores fracos de uma só família em sinais
continuam sendo uma limitação.

## A frase dentro do bloco de rotações é sustentada, com escopo mais preciso

Em kappa=3, os ratios das médias de mudança de FID são 7,757 no sinal positivo
e 7,731 no negativo. A direção nula e a física usam o mesmo normalizador,
bloco de 126 dimensões, fontes e orçamento por clipe. O corpo diz “about 7.7”,
comparando cada sinal separadamente, e o suplemento mostra os valores completos.

Não adotamos a faixa 5–10 como um limite exato: cruzar os extremos dos dois
sinais dá 5,44–11,02. A comparação com o mesmo sinal é mais clara. Também não
transformamos a frase em preferência universal por escala/cisalhamento:
a fibra aleatória tem efeito menor que rotações aleatórias. É uma comparação
direcional descritiva, não um novo teste prospectivo ou uma nova matemática
do conversor. Veja o [cálculo verificado](../results/cvpr_sign_stability_20261014/rotation_ratio.json).

## Verificação e entrega

A checagem independente reconstrói todos os 26.760 pares de entrada/condição,
confere normas e canais protegidos, matrizes canônicas e invariantes de Gram
nos controles físicos. Uma implementação de edição com matriz completa
recalcula todas as métricas e os 28 intervalos ajustados. Os 648 decodes
nativos de fixtures do primeiro/último batch coincidem exatamente.

A discrepância angular máxima da canonicalização é menor que 3,4e-6 graus;
a discrepância máxima de Gram física é menor que 2,4e-7. Os campos realizados
igualam a norma com erro menor que 4,4e-8 após divisão por max(B,1). As 64
entradas do arquivo congelado tiveram SHA256 conferido; os 59 inventários
históricos permanecem iguais. Não houve treino, novo gerador ou ajuste de recompensa.

O corpo tem oito páginas e duas de referências. O suplemento tem vinte e oito
páginas de conteúdo e duas de referências; as primeiras vinte e seis páginas
são pixel-identicas à entrega anterior inspecionada. A revisão inclui as
páginas novas, fontes, margens, referências e integridade dos PDFs. Os testes
de código e os checks numa cópia leve são realizados antes de encerrar a entrega.
O paper mantém autores anônimos e ID DRAFT. Autores, registro e leitura fria
humana ficam com o usuário; nenhum contato, publicação ou submissão é realizado.
