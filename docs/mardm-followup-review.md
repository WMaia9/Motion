# Fechamento da rodada MARDM

Os quatro estudos estão completos, com verificação independente e arquivos
imutáveis. A redação mantém uma afirmação: um avaliador que lê rotações 6D brutas
pode distinguir codificações que representam as mesmas rotações. O trabalho
preserva posições **e** rotações, e a defesa mantém informação rotacional.

O manuscrito ativo continua em [paper/current](../paper/current/README.md).
Os PDFs revisados são [principal](../paper/current/reviewed/main.pdf) e
[suplemento](../paper/current/reviewed/supplement.pdf). Esta rodada não acrescentou
treino, ajuste de recompensa, seeds de modelos ou novos geradores.

## Controles com L2 igual

Todos os vinte conjuntos reais salvos foram usados, com 4.640 entradas por
repetição e leitor. Dois orçamentos foram registrados antes dos scores:
`||(y-x)/std_do_leitor|| = rho*sqrt(frames_validos)`, com `rho=1,3`.
O orçamento é igual por clipe, incluindo os 263 canais antes da seleção de
features. Os leitores têm normalizações próprias; seus orçamentos numéricos
não devem ser comparados como se fossem uma unidade física comum.

| Intervenção, rho=3 | Mudança de FID no Guo | IC ajustado para 28 endpoints |
|---|---:|---:|
| Espaço nulo, direção fixa positiva | +0,08230 | [0,08209; 0,08250] |
| Espaço nulo, direção fixa negativa | +0,11688 | [0,11660; 0,11716] |
| Espaço nulo, coeficientes aleatórios | +0,00238 | [0,00232; 0,00244] |
| Rotações alteradas | +0,00349 | [0,00342; 0,00356] |
| Velocidades | +0,01586 | [0,01557; 0,01614] |
| Contatos | 0 | [0; 0] |
| Posições | +0,01826 | [0,01784; 0,01868] |

As direções fixas no espaço nulo mudam o score com orçamento igual e sem mudança
nas rotações ou demais features. O controle aleatório é muito menor. Portanto,
o resultado mostra sensibilidade direcional, sem estabelecer uma preferência
geral por todas as dimensões redundantes. Não é uma reprodução da magnitude da
Tabela 3 do MARDM.

O Guo exclui contatos: esse zero é uma checagem arquitetural. O TMR lê contatos;
no orçamento maior, contatos e a direção nula negativa reduzem R1 em cerca de
0,23 ponto percentual, com intervalos ajustados excluindo zero. A maioria dos
demais intervalos de R1 inclui zero. Ruído contínuo em contatos pode sair do
domínio binário. Os controles fora do espaço nulo podem ser inconsistentes com
os blocos mantidos fixos; eles medem sensibilidade da entrada. A calibração
angular coerente é um experimento separado.

Cada leitor teve 440 células de métricas recalculadas e 2.240 pares de fixtures
reconstruídos independentemente. Todos os vetores salvos de orçamento foram
conferidos. Foram 120 pares de FK por leitor; o maior resíduo de FK das amostras
foi 7,2e-7 m. Não se afirma uma reconstrução geométrica completa dos arrays não
salvos. Veja os resultados [Guo](../results/cvpr_matched_blocks_20261013/guo/analysis.json)
e [TMR](../results/cvpr_matched_blocks_20261013/tmr/analysis.json).

## Mecanismo natural e leitores essenciais

Na análise dos oito releases, três das quatro difusões/consistências têm delta
FID bruto–canônico negativo; os dois VQ têm delta positivo. As médias descritivas
são −0,00867 e +0,01864. T2M, uma VAE, tem o maior delta positivo de todos;
DiP recebe prefixo e aparece separadamente. Não há réplicas independentes de
treino por família, e não se faz inferência causal entre famílias.

As correlações de RMS com delta assinado e absoluto são −0,095 e 0,286, em oito
médias de releases. Os testes exatos não são significativos. Isso não prova
independência nem refuta a explicação completa do MARDM: o contrafactual modifica
somente a codificação das rotações, sem corrigir velocidades, contatos ou efeitos
de distribuição/treino. O [registro completo](../results/cvpr_rotation_mechanism_20261013/analysis.json)
inclui todos os valores por repetição e os testes.

Os checkpoints públicos essenciais do MARDM foram carregados estritamente, com
camadas e normalização nativas. Seus dois leitores são invariantes por exclusão
do bloco de rotações; isso não mede custo de acurácia. Em vinte repetições com
referência real comum, os rankings médios diferem do Guo canônico, ambos com
Kendall 0,857. Pesos, features e arquitetura contribuem para a diferença: não
se identifica um avaliador melhor ou uma reversão confiável de leaderboard.
Todos os 640 endpoints foram recalculados. O FID do encoder CLIP é descritivo,
sem comparação direta de escala com os outros leitores. Veja a
[análise](../results/cvpr_mardm_readers_20261013/analysis.json).

O código, o checkpoint, seu cartão MIT e o hash LFS do arquivo foram conferidos
nas fontes dos autores. A construção evita somente o download de uma torre
visual CLIP não usada; o checkpoint sobrescreve todos os parâmetros antes de
qualquer inferência, com comparação exata de cada tensor do modelo contrastivo.

## TMR completo e posicionamento

O teste direcionado usa agora todos os 1.024 clipes por fonte. R1 passa de 0,578
para 0,993 no MDM e de 0,688 para 1,000 no movimento real. O canônico do MDM é
0,580 **antes e depois** do ataque, distinto do bruto inicial. Os controles de
alvo errado, gradiente canônico e adversário genérico permanecem reportados.
Foram conferidos 6.144 pares de rotações e 48 pares de FK. O comparador genérico
atinge R1 perfeito, mas altera posições em média 0,810/0,899 m e rotações em
25,2/25,7 graus; a defesa não cobre alterações físicas. Os
[resultados completos](../results/cvpr_tmr1024_controls_20261013/analysis.json)
substituem o piloto menor na apresentação atual.

[MARDM](https://arxiv.org/abs/2411.16575v2) já publicou sensibilidade a blocos
redundantes e leitores treinados com features essenciais. A novidade defendida
aqui é mais específica: equivalência exata, controle sem deslocamento físico,
capacidade de manipulação e defesa que preserva rotações. Seus testes de ruído
não isolam essa equivalência. Nos inputs de sinais usados aqui, remover o bloco
de rotações remove todas as features de movimento; canonicalizar mantém o input
físico para a arquitetura existente. Não se afirma que outras arquiteturas ou
representações de sinais sejam impossíveis.

O abstract explicita transferência entre checkpoints da mesma arquitetura.
A igualdade de retrieval em dados reais já canônicos é chamada de checagem da
implementação, sem inferência de custo zero. A adaptação emparelhada de leitores
em sinais não canônicos continua separada. As margens publicadas aparecem na
tabela natural, como contexto de escala, sem testes de significância entre papers.
Todos os resultados negativos e os limites de acesso permanecem acessíveis.

## Entrega e processo externo

O corpo ocupa oito páginas; as referências ficam separadas. O suplemento tem
vinte e três páginas de conteúdo e duas de referências. As 35 páginas finais
foram inspecionadas visualmente; tipografia, margens, referências, fontes e
integridade passaram nas checagens. Os 2.912 arquivos dos quatro novos estudos
também passaram na conferência de SHA256.
O código portátil e as fontes do manuscrito são checados para anonimização;
o repositório experimental completo, com caminhos de proveniência, não é
certificado anônimo para publicação automática.

Os [prazos oficiais](https://cvpr.thecvf.com/Conferences/2027/Dates) são 10 de
novembro para registro, 16 para o paper e 23 para o suplemento, em AoE. O endpoint
de AuthorGuidelines 2027 ainda retornava 404 na checagem registrada; o kit oficial
2026 permanece provisório. As regras finais, incluindo a política de uso de LLMs,
devem ser reconferidas. Autores, registro e leitura fria humana ficam com o usuário,
conforme sua decisão anterior. Nenhum envio ou contato externo foi realizado.
