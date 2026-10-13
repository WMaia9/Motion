# Extensão HumanML3D e avaliação do novo parecer

A hipótese central do parecer foi confirmada no código oficial: HumanML3D usa
263 valores por frame, com 126 valores de rotação 6D em `67:193`. O avaliador
público lê `:259`, excluindo somente contatos dos pés. Ele recebe esse bloco sem
canonicalização. Fontes primárias: [representação do dataset](https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb)
e [wrapper do avaliador](https://github.com/GuyTevet/motion-diffusion-model/blob/main/data_loaders/humanml/networks/evaluator_wrapper.py).

A extensão foi testada, não apenas proposta. Não houve treino de modelos.
O artigo continua sendo editado exclusivamente em `paper/current/`.

## O que o TEST confirmou

| Fonte, 1.024 clipes fixos | R-precision antes | Depois, alvo original | IC ajustado do ganho |
|---|---:|---:|---|
| Movimento real | 0,4623 | 0,9961 | [0,5026; 0,5666] |
| MDM50 oficial | 0,3973 | 0,9928 | [0,5645; 0,6272] |

Somente as coordenadas redundantes de rotação são alteradas, na escala não
normalizada. Raiz, posições, velocidades, contatos e duração permanecem iguais.
Otimização recebe o embedding da caption-alvo e gradientes do avaliador congelado.
É um teste de capacidade adversária, não de exploração espontânea ou qualidade
melhor. R-precision usa cinco permutações fixas com 32 candidatos; intervalos
resampleiam IDs originais/espelhados e ajustam seis endpoints.

Com captions trocadas, R-precision sobe de 0,0283 a 0,8391 no movimento real e de
0,0326 a 0,7875 no MDM. O score contra a caption original cai. Canonicalização
mantém os embeddings; controles com loss no avaliador canônico não redirecionam
retrieval. Nenhum alvo é escolhido retrospectivamente por score.

FID não melhora nesse ataque à caption: no MDM, sobe de 0,3692 para 1,2874.
R-precision e MM-distance podem melhorar enquanto FID piora, com movimento
idêntico. Não apresentamos as três métricas como ganhos simultâneos.

## Um teste adicional responde à objeção de acesso à referência

Um ataque separado usa apenas a distribuição de embeddings de 2.048 movimentos
TRAIN. Ele constrói alvos de transporte gaussiano regularizado e ajusta as
coordenadas redundantes de 960 movimentos MDM de TEST. Os grupos de origem não
aparecem no TEST da primeira auditoria nem em TRAIN. A otimização não usa o
conteúdo de captions ou embeddings de movimento real de TEST.

| Condição | FID bruto | FID canônico | R-precision real |
|---|---:|---:|---:|
| Fonte | 0,3839 | 0,3799 | 0,4133 |
| Transporte, bound 0,1 | 0,2099 | 0,3799 | 0,4113 |
| Transporte, bound 0,35 | 1,0850 | 0,3799 | 0,4173 |
| Loss canônica, bound 0,35 | 0,3839 | 0,3799 | 0,4133 |
| Aleatório, bound 0,35 | 0,3526 | 0,3799 | 0,4092 |

O bound 0,1 reduz FID em 45,3% sem mudar movimento. O bound 0,35 piora o score;
ambos foram fixados antes da avaliação e permanecem reportados. Não há
monotonicidade comprovada ou seleção do melhor iterate em TEST.

Os ataques foram salvos e seus hashes fixados antes de calcular os embeddings
reais e os scores de TEST. O preflight lê bytes de arquivos de referência para
conferir hashes; isso não desserializa nem fornece captions, poses ou embeddings
à otimização. O gerador MDM recebe texto e duração, como entrada legítima de
produção; a camada de ataque não recebe conteúdo de texto.

FID foi registrado como endpoint descritivo. Uma sensibilidade **pós-hoc** de
200 bootstraps pareados por grupo dá IC ajustado por dois bounds para a mudança
[-0,2681; -0,0494] no bound 0,1 e [0,4045; 1,1655] no bound 0,35. Isso é
condicional a um único conjunto gerado, avaliador e seed de geração. Não mede
variabilidade entre seeds; 200 draws também limitam a precisão das caudas.

A matemática é estabelecida: [Bhatia, Jain e Lim](https://arxiv.org/abs/1712.01504)
estudam a geometria Bures–Wasserstein e transporte de covariâncias. Aqui, a
contribuição é o teste restrito à fibra de representação, com controle físico e
referência TRAIN. O mapa usa piso espectral; não alegamos novo teorema, ótimo
não regularizado ou nova difusão.

## Checagens e limitações

A primeira auditoria verifica 27.648 pares e 384 reconstruções FK nativas; o
segundo teste verifica 5.760 pares e 48 reconstruções FK. O erro angular máximo
é 3,8×10⁻⁵ graus; posições RIC e canais não rotacionais ficam bit a bit iguais.
FK difere em menos de 10⁻⁶ unidades de coordenadas. Embeddings canônicos variam
em até 7,2×10⁻⁷. O adapter concorda com o wrapper público em até 4,8×10⁻⁷;
R-precision, MM-distance e FID foram conferidos contra as funções nativas.

Checkpoints vieram de links dos autores. O arquivo de movimentos do Drive estava
com a cota pública esgotada, então usamos um espelho público com revisão e SHA256
LFS fixos. Seu tamanho coincide com o inventário do link oficial, mas a identidade
completa dos bytes originais não pôde ser conferida. O exemplo do autor é float32
e o do espelho float64; diferença máxima 1,94×10⁻⁵. Essa origem e diferença estão
explicitamente documentadas. Não alegamos reprodução integral do leaderboard.

A extensão acrescenta um avaliador público treinado independentemente e um domínio
de movimento geral. Ainda não fornece um segundo tradutor forte de língua de
sinais, julgamento humano ou validação de avatar. Amplitude dos ataques HumanML3D
não é prova de evasão de todos os detectores.

## Como respondemos ao parecer

- Concordamos: o endpoint original de treino comum continua sem replicação. O
  gap na célula CE10/40 não demonstra melhora: 5,39 bruto contra 3,40 canônico
  é aproximadamente **59% de diferença relativa entre scores**, não 59% de erro
  contra qualidade humana. A célula continua com três seeds.
- A Fig. adaptativa agora mostra a distribuição do efeito no leitor por clipe,
  medido por redução de erro de edição de tokens, em vez da diagonal RMS que é
  preservada por construção. Não chamamos essa estatística de BLEU por clipe.
- Priorizamos o teste do avaliador independente e o ataque sem captions antes
  de ampliar seeds de uma célula de treino que não identifica seu mecanismo.
- A redação prioriza capacidade demonstrada, dados reais e a extensão HumanML3D;
  treino comum permanece separado. O suplemento contém todos os controles.
- Discordamos de probabilidades numéricas de aceite sem modelo calibrado. O
  novo resultado fortalece o alcance e a resposta à objeção de referência, mas
  não fornece uma probabilidade de aceite.

[ALERT-Motion, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32657)
ataca geração por prompts. É antecedente relevante; nosso estudo mantém a
representação física e as posições, e muda o que o avaliador lê. Não alegamos
ser o primeiro ataque a text-to-motion.
