# Controle de estabilidade do leitor em entradas reais de sinais

Esta rodada fecha o controle pedido no parecer final: distinguir dependência de
codificação de instabilidade genérica do leitor. O manuscrito continua em
`paper/current/`. Nenhum treino, gerador, ajuste de recompensa ou nova seed de
modelo é acrescentado.

## Comparação dentro das rotações

Conferir a razão entre os FIDs médios das direções fixas nula e física, para
cada sinal separadamente, usando os vinte draws já congelados. Confirmar
o mesmo normalizador, fontes, bloco de 126 dimensões e norma por clipe.
Descrever o resultado como comparação destas direções, não como maior
sensibilidade universal à escala/cisalhamento. A divisão dos extremos dos dois
sinais não deve substituir a comparação com o mesmo sinal.

## Controle sem alvo em sinais

Usar os mesmos 2.183 clipes How2Sign e 1.162 CSL, IDs, ordem, cortes, pesos,
vocabularios e beam nativos. Projetar a mão esquerda exatamente como no estudo
original antes de todos os caminhos. Reexecutar bruto e canônico completos e
exigir as mesmas hipóteses congeladas, antes de interpretar os novos controles.

O orçamento de cada clipe é `B = ||C(x)-x||₂` nos 240 canais válidos. O RMS médio
do corpus não é usado como orçamento. A entrada não recebe uma normalização
adicional ausente do leitor original. Todos os controles são independentes de
legendas, referências e gradientes do avaliador.

1. Gaussianas IID nos 150 canais de corpo/mão direita, protegendo os mesmos
   90 canais de mão esquerda. Reescalar cada campo à norma B por clipe. Retêm-se
   três campos previamente definidos e sua média por clipe, sem escolher seeds.
2. Controle gaussiano secundário nos 240 canais, com o primeiro campo definido
   e a mesma norma B. Ele explicita o efeito de permitir perturbação na mão
   esquerda já projetada.
3. Mudança física de um grau, com eixo aleatório por articulação fixo no tempo.
   Rodar os dois vetores originais juntos; preservar suas normas e produto
   escalar, protegendo a mão esquerda. Não substituir a codificação por uma
   canônica, o que misturaria mudança física e mudança de fibra.
4. Mudança física com os mesmos eixos, ângulo por clipe resolvido da norma B
   em [0,pi]. Reportar a distribuição dos ângulos, sem afirmar que todos são
   pequenos. Se a norma for inalcançável, registrar o clipe explicitamente,
   preservando-o no controle de um grau; não excluir silenciosamente entradas.

Somando bruto/canônico, são oito condições por corpus. Os campos Gaussianos
usam seeds 20261014/20261015/20261016, indexadas por corpus e clipe. Esses são
campos de intervenção, não réplicas de treino. A tolerância do orçamento
float32 é `5e-6*max(B,1)`, com as normas realizadas salvas.

## Medidas e incerteza

Os endpoints primários são fração de hipóteses alteradas na granularidade do
score e distância de edição normalizada média. How2Sign usa palavras; CSL usa
caracteres, preservando pontuação. A normalização é pela maior contagem de
tokens entre as duas frases, com mínimo um. Também reportar alterações de
strings exatas, edição média em tokens e todos os BLEUs. Frases vazias permanecem.

Os intervalos são pareados, com 20.000 bootstraps: VIDEO_ID oficial no How2Sign
e clipe no CSL. Usar os mesmos pesos de reamostragem entre condições. A família
primária tem vinte endpoints: cinco condições resumidas, dois corpora e duas
medidas. Os contrastes canônico menos gaussiana média ou físico de norma igual
constituem uma família separada de oito endpoints. Manter os três campos
individuais, sem alegar que são datasets ou réplicas populacionais independentes.

Recomputar métricas, edição e intervalos independentemente. Reconstruir os
controles por fórmulas separadas e SciPy a partir de todas as entradas salvas;
verificar normas, canais protegidos, matrizes e preservação dos coeficientes
nos controles físicos. Usar fixtures do começo e fim para forwards nativos.
O decode lê só movimentos e comprimentos; referências entram somente no score.

## Interpretação antes de ver os controles

Se o canônico muda mais frases que o ruído, isso é especificidade relativa aos
controles testados; não prova qualidade linguística ou imunidade a outros ruídos.
Se o ruído muda tanto quanto ou mais, explicitar instabilidade mais ampla no
abstract e corpo. Comparar também edição média: contar frases alteradas não
mede a intensidade da mudança. A invariância do contrato permanece uma garantia
separada; nenhuma dessas comparações demonstra melhora semântica ou custo zero.

Depois dos controles completos, congelar a evidência, rever o texto, compilar
e validar PDFs/metadados/repositório. O processo de autores, registro e leitura
fria humana permanece com o usuário, conforme sua decisão anterior.
