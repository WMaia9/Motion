# Estado científico — 14 de outubro de 2026

O controle final de estabilidade está completo: oito condições em cada um dos
2.183 clipes How2Sign e 1.162 CSL, com replay bruto/canônico idêntico ao arquivo
anterior. A canonicalização muda 65,9%/39,9% das hipóteses, contra 41,9%/22,3%
com Gaussianas de norma igual, e também tem maior edição normalizada. Os dois
contrastes Gaussianos são positivos após ajuste. Entretanto, controles físicos
de norma igual mudam 70,3%/44,3%; isso não demonstra sensibilidade exclusivamente
à codificação ou acurácia semântica. A comparação física de frações CSL permanece
inconclusiva; todos os controles e intervalos estão preservados.

Dentro das rotações, os ratios fixo-nulo/fixo-físico com o mesmo sinal são
7,757/7,731. A frase é direcional, sem preferência universal por redundância.
Veja a [revisão atual](sign-stability-review.md) e os PDFs em
`paper/current/reviewed/`. Foram verificados todos os 26.760 pares, 648 fixtures
nativas e 28 intervalos. Nenhum treino ou novo gerador foi acrescentado.

A rodada pós-MARDM fechou os quatro reparos do parecer: direções fixas em todos
os blocos, evidência natural em sinais no corpo, orçamento `kappa` separado da
escala de fibra `rho`, e abstract com um resultado sem alvo. Todos os vinte draws
Guo/TMR e suas verificações independentes estão completos. Veja a
[revisão atual](coherence-review.md). Os PDFs ativos continuam em
`paper/current/reviewed/`; o arquivo MARDM anterior permanece imutável.

O controle adicional confirma que L2 igual não basta para atribuir o efeito à
redundância. Direções fixas de posições e velocidades chegam a 1,189 e 0,476 FID,
acima do máximo nulo de 0,117. Coeficientes nulos aleatórios também têm resposta
menor que ruído aleatório de posições/velocidades. O texto retira a leitura de
preferência por redundância; a particularidade da fibra é preservar todas as
rotações e os demais canais. Constância temporal não implica consistência
física entre blocos, que permanecem protegidos nesses controles.

Sem ataque, camada ou nova inferência, as hipóteses salvas confirmam mudanças
em 1.438/2.183 How2Sign e 464/1.162 CSL, contra 0/642 PHOENIX. Os seis BLEUs foram
recalculados independentemente; How2Sign varia somente 3,4779 → 3,4840. Esses
inputs compartilham o pré-processamento nativo original, inclusive da mão esquerda.
Mudanças de frases não estabelecem melhora/piora semântica nem custo zero da defesa.

A [rodada MARDM](mardm-followup-review.md) anterior mantém a análise de oito
releases, os leitores essenciais públicos e TMR com 1.024 clipes por fonte.
A canonicalização mantém as rotações disponíveis; a auditoria não refuta o
mecanismo completo do MARDM.

A rodada anterior acrescentou TMR, calibração física coerente, ataque cego no MLD
com transferência, MoMask/DiP/MotionDiffuse HumanML3D e MoMask/MotionDiffuse KIT.
A igualdade de scores no dado real já canônico é uma checagem da implementação,
sem estabelecer custo zero em dados não canônicos. Veja a
[generalização anterior](generalization-review.md).

O resultado principal agora é uma inconsistência de representação em avaliadores
de movimento: scores mudam para posições e rotações idênticas. A extensão
HumanML3D completou vinte repetições de cinco geradores públicos e dois
checkpoints oficiais de avaliação. Todos os cinco gaps primários de FID são
não nulos após ajuste por dez endpoints; T2M-GPT muda 18,6% e MotionLCM-v1 5,5%
do FID bruto. A ordem média primária permanece igual. Isso confirma dependência
natural de codificação, não exploração intencional de recompensas.

O ataque sem legenda foi repetido em três coortes: bound 0,1 reduz FID em
39–53%, e 0,35 piora todas. As coortes têm sobreposição de grupos, explicitada
nos resultados. O piloto de recompensa MDM piorou retrieval/FID e não testa exploração em
otimização bem-sucedida. A grade DEV não qualificou ajustes para TEST; os ganhos dos pesos públicos MotionRFT sobrevivem à canonicalização.
Veja [a rodada completa e sua análise](motion-release-review.md).

O manuscrito único é [paper/current/](../paper/current/README.md), com oito
páginas de conteúdo e duas de referências; suplemento com vinte e oito mais duas.
Os resultados anteriores de língua de sinais e seus limites permanecem:

| Camada aprendida, TEST completo, três seeds | BLEU antes | BLEU após | Ganho | IC pareado 95% |
|---|---:|---:|---:|---|
| PHOENIX, movimento gerado canônico | 6,39 | 15,21 | 8,81 | [7,82; 9,87] |
| How2Sign, movimento real | 3,48 | 6,11 | 2,64 | [2,27; 3,03] |
| CSL, movimento real | 6,89 | 7,43 | 0,54 | [0,14; 0,97] |

Todas as hipóteses canônicas ficam iguais por clipe. Os idiomas usam tokenização
nativa; How2Sign usa grupos VIDEO_ID. 15,21 contra 15,00 do real não demonstra
superioridade estatística: IC da diferença [−0,85; 1,24]. A camada PHOENIX usa
movimento gerado; as outras duas usam movimento real.

## Resultado novo: estatísticas de norma fixas

Um ataque por clipe no círculo preserva distância canônica **E** e norma
quadrada 6D **N** por articulação e frame. No PHOENIX TEST, com 642 clipes, BLEU passa de
6,624 a 10,716: ganho **4,091**, IC ajustado por quatro comparações
**[3,042; 5,237]**. Com alvo errado: 1,247 → 3,349; ganho 2,102,
IC ajustado [1,219; 3,119]. As hipóteses canônicas permanecem iguais.

O detector RMS tem AUC 0,5003. Nove estatísticas escalares exploratórias,
com limiares definidos em DEV, têm AUC 0,500–0,530 no círculo. Isso não prova
furtividade geral: detectores temporais/multivariados não foram cobertos. A
esfera preserva apenas E e é mais detectável por outras estatísticas.
As camadas aprendidas sob estatísticas fixas continuam negativas; otimização
por clipe tem acesso diferente e não é uma confirmação de amortização.

## Extensão confirmada: HumanML3D

O avaliador público T2M lê as 126 coordenadas 6D sem canonicalizá-las. Em 1.024
clipes TEST fixos, o ataque com caption-alvo leva R-precision de 0,397 a 0,993
no MDM e de 0,462 a 0,996 em movimento real, mantendo rotações e posições.
O ataque melhora retrieval/MM-distance mas piora FID; isso não melhora movimento.

Na primeira coorte de 960 clipes, que excluiu os grupos do ataque com caption, um ataque sem conteúdo de
captions usa somente estatísticas de 2.048 movimentos TRAIN. Bound 0,1 reduz
FID 0,384 → 0,210 (45,3%); bound 0,35 piora para 1,085. O bootstrap pós-hoc por
grupos mantém o sinal da redução, IC ajustado [−0,268; −0,049], condicionado a
um único conjunto/seed de geração. Todos os bounds e controles são preservados.

A extensão verifica mais 33.408 pares geométricos e 432 reconstruções FK nativas.
Ela acrescenta uma arquitetura pública de avaliação de movimento geral, mas
não um segundo tradutor forte de língua de sinais. A origem dos vetores no
espelho público e a diferença numérica para o exemplo do autor são explícitas.
Veja [a análise do parecer e da extensão](humanml-review.md).

## Replicação de treino comum

Sete seeds novas em geodésica/CE bruta, peso 1 e 20 épocas, dão gap bruto−canônico
0,131, IC seed+clip [−0,570; 0,807]. A combinação com as três históricas dá
**0,195 [−0,434; 0,797]**. O endpoint original segue inconclusivo.

A grade exploratória fixada usa pesos 1/3/10 e 20/40 épocas, com três seeds históricas.
Em 40 épocas, pesos 3/10 dão gaps 1,966/1,993, com intervalos seed+clip ajustados
por 11 comparações [0,261; 3,645]/[0,590; 3,586]. O BLEU canônico cai sob maior pressão da CE;
isso não é produção melhor. Nenhum contraste ajustado estabelece monotonicidade.
O transplante de coeficientes em toda a grade também não confirmou transmissão
do texto-alvo: todos os oito intervalos ajustados incluem zero.

## Escopo da entrega

A rodada anterior de replicação acrescentou 23 fits, 64 avaliações de modelos (DEV/TEST),
28 condições de ataques/controles e 24 diagnósticos de transplante.
Todos os protocolos, positivos, negativos, pesos, arrays e hipóteses foram
preservados em relatórios congelados. A checagem independente cobre 58.336
pares geométricos da rodada e 21.184 comparações bit a bit do fluxo esquerdo.
A primeira extensão HumanML3D acrescentou 48 condições de auditoria e seis
sem captions, verificando 33.408 pares. A rodada anterior de releases completa 100 células
gerador/repetição em dois checkpoints oficiais, 140 controles reais fixos,
duas coortes cegas adicionais, nove fits de recompensa MDM e dez gerações
dos checkpoints MotionRFT/base. Sua checagem geométrica principal cobre
391.680 pares de saídas públicas, além dos controles e estudos suplementares.
O Git contém somente código, manuscrito e metadados compactos.

O paper incorpora a auditoria ampliada no mesmo caminho `paper/current/`; o suplemento
contém os controles completos. A bibliografia tem 52 entradas; o principal cita 40.
A [revisão editorial](editorial-review.md) reorganiza essa evidência sem novos treinos.
O kit oficial disponível é o de 2026, com conformidade 2027 ainda provisória.
Não houve submissão externa nem commit automático.

A avaliação de língua de sinais continua limitada a uma família de leitores BT
fracos em How2Sign/CSL, embora HumanML3D agora forneça duas arquiteturas de avaliação
em outro domínio. Um leitor forte independente e/ou avaliação linguística
ampliaria a evidência. Não há criptografia, anonimização, furtividade geral ou
probabilidade de aceite demonstradas.

[Análise independente do parecer e referências](critical-review.md) ·
[Resultados exportados](../results/README.md) ·
[Pesquisa matemática/física anterior](mathematics-and-physics.md)
