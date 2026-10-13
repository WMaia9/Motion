# Auditoria de movimento — 13 de outubro de 2026

A objeção de prevalência recebeu uma resposta experimental: a pontuação depende
da codificação nas saídas de cinco geradores públicos, sem acrescentar ataques
ou treinar uma camada. Isso sustenta uma auditoria de consistência de avaliação;
não demonstra que os autores desses geradores manipularam scores.
O manuscrito ativo permanece exclusivamente em `paper/current/`.

## Saídas públicas: 20 repetições por método

Cada array gerado é pontuado bruto e após canonicalizar somente as rotações 6D.
Posições, rotações representadas, velocidades, contatos e duração permanecem iguais.
A referência FID usa 4.640 entradas do loader nativo completo; o MDM emite 1.024
saídas seguindo seu limite nominal de 1.000, e os outros métodos emitem 4.640.
As entradas incluem segmentos e versões espelhadas, não gravações independentes.
Retrieval usa candidatos em batches de 32. Não afirmamos reprodução exata dos
números de leaderboard de cada paper.

| Gerador | FID bruto | FID canônico | Bruto − canônico | IC pareado ajustado |
|---|---:|---:|---:|---|
| MDM50 | 0.45501 | 0.45037 | +0.00463 | [+0.00348; +0.00579] |
| MLD | 0.44529 | 0.44906 | -0.00378 | [-0.00519; -0.00237] |
| MotionLCM-v1 | 0.57470 | 0.60617 | -0.03147 | [-0.03473; -0.02822] |
| T2M-GPT | 0.18281 | 0.14883 | +0.03397 | [+0.03254; +0.03541] |
| T2M | 1.39350 | 1.34714 | +0.04636 | [+0.04414; +0.04858] |

Os intervalos ajustam dez endpoints FID/R1 por leitor. Os cinco gaps de FID
excluem zero; R1 muda pouco, com dois gaps positivos ajustados. Canonicalizar
reduz o FID do T2M-GPT em 18,6% do valor bruto e aumenta o do MotionLCM em 5,5%.
A direção não é universal e a ordem média primária dos cinco modelos não muda.
A diferença é de score para o **mesmo movimento**, não de qualidade humana.

Um segundo checkpoint oficial, obtido no download do T2M-GPT, reavalia todas
as saídas. Tem a mesma arquitetura e normalização, com pesos diferentes.
Quatro gaps de FID excluem zero; MLD fica inconclusivo e T2M muda de direção.
Todos os gaps de R1 incluem zero. A pequena troca de ordem média MLD/MotionLCM
no segundo leitor não estabelece inversão confiável de ranking.
As famílias de intervalos dos dois leitores são reportadas separadamente.

## Recodificações fixas no protocolo nativo

| Codificação de movimento real idêntico | FID médio | Margem nativa de 95% |
|---|---:|---:|
| Original | 0,00099 | 0,00002 |
| Canônica | 0,00099 | 0,00002 |
| Direcional +0,10 | 0,06129 | 0,00057 |
| Direcional −0,10 | 0,08336 | 0,00055 |
| Direcional +0,35 | 2,00261 | 0,00329 |
| Direcional −0,35 | 3,63852 | 0,00432 |
| Aleatória, bound 0,35 | 0,02628 | 0,00044 |

São vinte repetições completas de 4.640 movimentos, com referência independente
em crop/ordem. Os coeficientes direcionais são `(b, −b, b)`, com escalas
`exp(b), exp(−b)` e cisalhamento `b`, antes da normalização. Os efeitos grandes
exigem direção e magnitude 0,35; os controles menores/aleatórios não são zero.
As mudanças direcionais são 4,4/8,0 vezes o FID bruto do MDM no protocolo novo.
Isso não diz que esses coeficientes grandes aparecem naturalmente. Os antigos
2,03/3,55 eram outra subamostra, com referência real idêntica e baseline zero;
não substituímos seus JSONs nem misturamos seus intervalos com os atuais.

## Ataque sem legenda: três coortes

| Seed de geração | FID original | Bound 0,10 | Bound 0,35 |
|---|---:|---:|---:|
| 42 | 0,38385 | 0,20989 | 1,08504 |
| 4242 | 0,37622 | 0,17613 | 1,14775 |
| 7777 | 0,32676 | 0,19868 | 1,42923 |

São 960 saídas por coorte; bound 0,10 reduz FID em 39,2–53,2%, enquanto 0,35
piora nas três. Todas preservam movimento; os controles de gradiente canônico
permanecem no baseline. As novas coortes compartilham 493 grupos entre si e
494/496 com a original. Excluem os grupos do primeiro ataque com legenda, mas
não são clipes independentes nem TEST globalmente inédito. A otimização usa
somente estatísticas de 2.048 movimentos TRAIN e saídas geradas, sem conteúdo
de captions ou embeddings reais TEST. Cada ataque é salvo antes de sua própria
pontuação de referência. Não juntamos coortes sobrepostas como amostras
independentes para produzir um intervalo artificialmente estreito.

## Treino com recompensa: controles negativos

Nove fits atualizam o denoiser MDM, sem camada de coeficientes: MSE, recompensa
bruta e recompensa canônica, em três seeds. Usam 1.000 passos e 2.048 movimentos
TRAIN; os nove pesos foram selados antes de TEST. A recompensa é InfoNCE de uma
previsão denoised-x0, não uma reprodução de otimização ao longo do rollout.

| Braço, média de três seeds | FID bruto/canônico | R1 bruto/canônico |
|---|---|---|
| MSE | 0,602 / 0,602 | 0,410 / 0,408 |
| Recompensa bruta | 3,320 / 2,804 | 0,358 / 0,364 |
| Recompensa canônica | 3,495 / 3,687 | 0,355 / 0,354 |

O piloto não confirma inflação de retrieval. As duas recompensas pioram estas
métricas de amostragem, e os intervalos entre apenas três seeds são largos.
Não exclui exploração com outros objetivos, budgets ou recompensas de rollout.

Separadamente, auditamos o checkpoint **publicado** do MotionRFT e seu base
vinculado pelos autores: cinco seeds emparelhadas, dois leitores, 1.024 saídas e
referência completa. No leitor primário, R1 sobe de 0,434 para 0,514 em bruto e
canônico; no segundo, de 0,504 para 0,587/0,586. Os ganhos sobrevivem à remoção
do canal 6D. FID também melhora em ambos. É um diagnóstico suplementar de cinco
seeds, não a reprodução de leaderboard em vinte repetições. A rede de recompensa
MotionReward é diferente do leitor padrão T2M; não testamos sua vulnerabilidade
nem atribuímos os ganhos do método a hacking.

## O que foi verificado e o que continua aberto

A checagem nativa independente cobre todos os 391.680 pares de saídas públicas,
200 reconstruções FK amostradas, 4.480 pares de controles reais e 280 FK adicionais.
Scores de ambos os leitores e dos controles fixos foram recomputados de embeddings
salvos. O piloto e os checkpoints MotionRFT têm suas próprias verificações.
As rotações se mantêm dentro dos resíduos numéricos medidos; os outros canais
são conferidos bit a bit. Não há mesh/avatar SMPL licenciado neste estudo.

A contribuição principal agora é **consistência de avaliação em movimento**,
com evidência natural e controles causais. Exploração deliberada durante treino
normal permanece não confirmada. Canonicalização fecha a fibra 6D testada;
não resolve outras redundâncias, compreensão linguística ou qualidade humana.
Os BTs de língua de sinais ainda compartilham uma família com leitores fracos.
O segundo leitor HumanML3D replica pesos, não arquitetura. A origem do espelho
processado e sua diferença numérica para o exemplo do autor continuam explícitas.

O paper começa por HumanML3D, mostra um esqueleto nativo no teaser e inclui
um esquema da intervenção. O score 0,397→0,993 é agregado de 1.024 movimentos,
não o score do único clip ilustrado. Treino comum, budgets e controles completos
estão no suplemento. São 46 referências, com os três novos registros conferidos
nas versões finais dos publishers. Não usamos probabilidades de aceite como
medida científica.

As correções de engenharia ficam nos artefatos e na reprodução: MotionLCM-v1
requer Sentence-T5, e o T2M exige agrupamento por duração e comprimento da
caption. Tentativas inválidas e versões executadas permanecem preservadas.
Não foram descartadas repetições válidas para melhorar números.

A página oficial informa registro em **10/11/2026**, submissão em **16/11/2026**
e suplemento em **23/11/2026**, AoE. O kit disponível ainda é o de 2026;
metadados de autores/ID e conformidade final 2027 continuam pendentes.
[Datas oficiais](https://cvpr.thecvf.com/Conferences/2027/Dates).

Fontes primárias: [protocolo MDM](https://github.com/GuyTevet/motion-diffusion-model/blob/main/eval/eval_humanml.py),
[MLD](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Executing_Your_Commands_via_Motion_Diffusion_in_Latent_Space_CVPR_2023_paper.html),
[T2M-GPT](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Generating_Human_Motion_From_Textual_Descriptions_With_Discrete_Representations_CVPR_2023_paper.html),
[MotionLCM](https://link.springer.com/chapter/10.1007/978-3-031-72640-8_22),
[MotionRFT](https://github.com/Xiaofeng-Tan/MotionRFT).
