# Avaliação independente do parecer de “training objectives”

O parecer identifica um problema real de inferência: recuperar o leitor bruto
após deriva de coordenadas não é demonstrar inflação robusta no mesmo movimento.
O experimento decisivo para essa afirmação é o gap bruto–canônico do **mesmo
output**. Isso não invalida a demonstração separada de um ataque condicionado ao
texto. Também não fornece uma probabilidade calibrada de aceite em CVPR.

## O que concordamos e o que precisa de precisão

| Ponto do parecer | Avaliação independente | Consequência |
|---|---|---|
| Excesso sobre “sem CE” é recuperação de colapso | A decomposição suporta essa explicação; não identifica todo o mecanismo de treino | O excesso deixa de ser a evidência principal; publicar o gap absoluto e o controle CE canônica |
| Uma seed sustenta o gap TEST | Correto nos três runs originais: −0,0095 / +1,0776 / −0,0297 | Replicar com sete seeds novas fixadas antes de fitting, sem descartar runs |
| Texto-fonte torna o ataque inválido | Não. Define um atacante condicionado ao alvo; não demonstra exploração espontânea | Declarar acesso ao texto, leitor congelado e inferência BT sem entrada textual extra |
| Dez seeds positivas tornam o paper forte para CVPR | Não existe esse critério universal | Examinar tamanho de efeito, poder, variabilidade, novidade e leitores independentes |
| AUC próximo de 1 reduz a gravidade prática | Sim, para os ataques e estatística testados | Não alegar furtividade ou evasão geral |
| Detector é fraco contra adversário informado | Hipótese plausível, não resultado nosso | Testar ataque adaptativo mantendo as próprias estatísticas, e preservar resultados negativos |
| Camada aprendida em três corpora com movimento real | A formulação ultrapassa os dados atuais | PHOENIX aprendido usa movimento gerado; How2Sign/CSL aprendido usam movimento real. Ataques por clipe cobrem movimento real nas três |
| 15,21 “supera” 15,00 | Apenas numericamente; IC da diferença [−0,85; 1,24] | Dizer “alcança a escala do real”, sem superioridade estatística ou melhoria de sinais |
| ICCV/ACL seriam o destino se o teste falhar | Sugestão de posicionamento, não garantia ou diagnóstico científico | Escolher venue pelo escopo/contribuição, sem presumir aceite mais fácil |

## Por que um ataque construído continua sendo evidência

[Generative Adversarial Perturbations, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Poursaeed_Generative_Adversarial_Perturbations_CVPR_2018_paper.html)
mostra redes geradoras de ataques, incluindo alvos definidos. Logo, a construção
intencional do atacante não desqualifica um estudo de vulnerabilidade. Essa
referência também **reduz nossa pretensão de novidade**: aprender uma perturbação
em uma passagem já é conhecido. A distinção que precisamos sustentar é alterar
somente a representação redundante de uma rotação, com equivalência verificável,
no contrato específico de avaliação de produção de sinais.

O texto-fonte é entrada legítima de produção texto→sinal. Usá-lo para construir
features adversárias não equivale automaticamente a vazar rótulos de TEST para
fitting: o alvo é fornecido em inferência, como parte do modelo de ameaça. Em
avaliação de vídeo→texto sem caption disponível, essa capacidade seria mais forte
que a permitida à aplicação. O paper deve declarar ambos os contextos. Isso não
permite interpretar um ataque condicionado como tradução melhor, nem inferir que
geradores publicados exploram o canal.

[Huang e Baldwin, Findings EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.340/)
estudam fragilidade de BERTScore/BLEURT/COMET sob ataques e usam validação humana.
O precedente sustenta investigar métricas como objeto científico; não demonstra
nosso efeito e não elimina nossa limitação de uma família de leitores fracos.
A diferença relevante é que nosso controle mantém o objeto geométrico
representado fixo, em vez de alterar a sentença ou assumir pequena mudança
semântica. Não chamamos essa propriedade de novidade matemática.

## Matemática útil e limites da defesa

A representação 6D e sua projeção vêm de
[Zhou et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html).
No domínio não degenerado, os representantes de uma rotação têm
`a = rho*u`, `b = sigma*v + tau*u`, com `rho,sigma > 0`. As coordenadas
`rho,sigma,tau` são descartadas pelo conversor físico.

Uma observação matemática simples organiza o resultado: um leitor é constante
em cada fibra se, e somente se, ele pode ser escrito como uma função da rotação
representada. A direção necessária define a função escolhendo qualquer
representante; a constância torna a escolha irrelevante. A suficiente é a
substituição direta. Isso é fatoração pelo quociente, não um novo teorema ou nova
difusão. `f(C(x))` garante essa propriedade no domínio definido, em aritmética
exata. Idempotência, domínio válido, padding, precisão numérica e hipóteses
concretas continuam exigindo verificação na implementação.

Canonicalização fecha o canal específico independentemente do sucesso de um
ataque contra RMS. Detecção depende de um teste estatístico e pode rejeitar
saídas, enquanto canonicalização escolhe um representante da mesma rotação.
Isso **não prova que todo detector é fraco**, nem que a acurácia linguística é
preservada após mudança de distribuição. Nossos fine-tunes e seus intervalos
medem essa segunda questão separadamente.

[Tramèr et al., NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/hash/11f38f8ecd71867b42433548d1078e38-Abstract.html)
motivam mirar a defesa explicitamente em avaliação adaptativa. A referência não
prova evasão no nosso caso. O novo teste por clipe mantém a distância canônica
(e, no círculo, também a norma 6D) por frame/joint. O insucesso anterior de uma
camada amortizada nessa geometria permanece reportado, mesmo que otimização por
clipe obtenha resultado diferente.

[Tramèr et al., ICML 2020](https://proceedings.mlr.press/v119/tramer20a.html)
mostram perigos de invariância excessiva. Seu termo “invariance-based attack”
tem outra definição: muda o rótulo verdadeiro e mantém a predição. Nosso caso
mantém a rotação e muda a predição. Não confundimos as duas nomenclaturas; não
impomos invariância a transformações que mudam semântica física.

## Replicação que pode resolver a fragilidade de seeds

[Colas et al., 2018](https://arxiv.org/html/1806.08295v2)
relacionam seeds, variância, poder e erros estatísticos. São experimentos de RL;
a aplicação aqui é metodológica. Não há razão para interpretar oito ou dez runs
como número mágico. Bootstrap de clips com três modelos fixos tampouco mede
variabilidade de fitting.

Nos três gaps originais, a média é 0,346 e o desvio entre seeds é 0,634.
Uma ilustração paramétrica, supondo esse efeito e desvio conhecidos, dá somente
cerca de 34% de poder com dez seeds em teste bilateral a 5%. A estimativa com
três runs é instável e não é uma análise de poder calibrada; serve para mostrar
por que dez não garantem um intervalo positivo. O tamanho registrado da rodada
não muda após esse diagnóstico.

A rodada nova registra sete seeds adicionais (11, 97, 909, 1337, 2027, 2718,
31415), geodésica + CE bruta 1 por 20 épocas, e controle CE canônica pareado.
O endpoint principal é o gap nas **sete seeds novas**; agregar as três já vistas
produz uma extensão de dez seeds, explicitamente pós-parecer. Não é um TEST
externo intocado. Todos os runs compartilham uma inicialização pré-treinada.

A dose-resposta é exploratória: pesos 1, 3 e 10; epochs 20 e 40; seeds 42, 123,
777; todos os checkpoints fixos. Runs de 40 épocas partem da mesma inicialização,
com checkpoint de optimizer/RNG, e não fingem continuar checkpoints históricos
que não armazenavam optimizer. Há intervalos com resampling de seed e clip,
ajuste de multiplicidade para os seis endpoints e cinco contrastes, e análise
de retirada de cada seed. Nenhum run é interrompido quando o resultado vira
positivo e nenhum melhor peso é escolhido em TEST.

Pesos maiores também amplificam um gradiente CE já dominante. Um gap positivo
com movimento degradado não seria uma produção melhor. A pesquisa precisa
reportar canonical BLEU, diagnóstico geodésico e RMS junto do gap.

## Regra de escrita

O argumento central é **consistência do avaliador com o conversor físico** e o
ataque condicionado que a viola. Treino comum permanece uma pergunta empírica
separada, com resultado correspondente aos novos testes. Acurácia humana,
leitores independentes, transferência e furtividade não são preenchidos com
frases otimistas ou referências de outros domínios.

## Pesquisa recente e caminhos que têm relação com o achado

[MotionRFT, arXiv:2603.27185](https://arxiv.org/abs/2603.27185) combina
recompensas multidimensionais e várias representações de movimento. Isso torna
insuficiente vender “reward hacking em movimento” como novidade geral. Nossa
pergunta específica é se o avaliador distingue representantes da **mesma**
rotação; o artigo não serve como confirmação desse problema. O projeto e o
repositório apresentam sinais diferentes sobre status editorial; citamos o
preprint verificável, sem atribuir aceite não confirmado.

[FlowHMR, arXiv:2610.03691](https://arxiv.org/abs/2610.03691), publicado como
preprint em 2 de outubro de 2026, combina fidelidade em vídeo e sucesso de
tracking por controlador físico. É uma direção concreta para avaliar movimento
além do texto. Mas um controlador que recebe apenas rotações/posições não vê as
coordenadas redundantes: pela própria equivalência, sua recompensa também fica
constante no canal. Para o problema atual, adicionar uma recompensa física não
substitui o contrato de canonicalização. Essa última conclusão é nossa dedução
matemática, não uma conclusão experimental do FlowHMR. O pipeline oficial exige
ativos corporais licenciados que não temos; não inventamos validação por avatar
ou simulador.

[BackTranslation2.0, arXiv:2606.28673](https://arxiv.org/abs/2606.28673) separa
qualidade gramatical, fonológica, fluência e fidelidade, e valida em BSL com
avaliação humana. Fortalece a necessidade de leitores/avaliação independentes;
não é um checkpoint compatível que possamos substituir diretamente nos nossos
arrays 240D em alemão, inglês e chinês. Nossa invariância geométrica é um
requisito necessário; não cobre expressão facial, gramática espacial ou toda
qualidade linguística.

A aplicação útil de matemática nesta rodada é restringir a intervenção à
interseção de invariantes: uma esfera de distância canônica e um plano de soma
`rho+sigma` constantes deixam um círculo. Isso permite **testar** uma objeção
específica do detector, em vez de apenas renomear uma camada como física ou
criptografia. A isometria e a fatoração por quociente são ferramentas conhecidas;
a contribuição potencial é o resultado empírico controlado no contrato de BT.

## Resultados concluídos e decisão de enquadramento

Foram concluídos **23 novos fine-tunes**, avaliação de **32 condições em DEV e
32 em TEST**, **16 ataques/controles DEV**, **12 ataques/controles TEST**, e
**24 diagnósticos de transplante DEV**. Cada estudo tem seu próprio protocolo,
arrays, hipóteses, checagem independente e manifesto SHA256. Todos os resultados
negativos permanecem disponíveis.

| Pergunta | Evidência nova | Conclusão |
|---|---|---|
| CE1/geodésica/20 é estável? | Sete novas seeds: gap 0,131 [−0,570; 0,807]; dez incluindo históricas: 0,195 [−0,434; 0,797], seed+clip | Não confirmado; mais seeds não resolveram o endpoint |
| CE mais forte produz diferença de representação? | Pesos 3/10, 40 épocas: gaps 1,966/1,993; intervalos seed+clip ajustados por 11 [0,261; 3,645]/[0,590; 3,586] | Sim, no stress test; canonical BLEU também cai, não melhora a produção |
| Dose-resposta monotônica? | Todos os cinco contrastes ajustados incluem zero | Não demonstrada estatisticamente |
| Isso transmite caption no gerador comum? | Transplantes em toda a grade: nenhum dos oito intervalos ajustados exclui zero | Mecanismo não confirmado; não alegar exploração rotineira |
| Um detector RMS sempre identifica ataque bem-sucedido? | Círculo em TEST642: 6,624 → 10,716; ganho 4,091 [3,042; 5,237], ajustado por quatro; RMS AUC 0,5003 | Esse detector não basta; E e norma quadrada 6D N são preservados por frame/joint |
| O alvo errado também pode ser favorecido sob E/N fixos? | 1,248 → 3,349; ganho 2,102 [1,219; 3,119], ajustado por quatro | Sim, sob acesso por clipe ao alvo e aos gradientes do leitor |
| Todas as formas de detecção falham? | Nove thresholds escalares, ajustados em DEV, dão AUC círculo 0,500–0,530; esfera chega a 0,877 | Cobertura limitada; não é furtividade geral, criptografia ou segurança provada |

O resultado do círculo é novo em relação ao parecer: o fracasso da camada
aprendida na mesma superfície não implicava resistência do leitor à otimização
por clipe. A esfera tem orçamento angular maior que o controle aprendido;
não atribuímos sua diferença somente à amortização. O círculo conserva o mesmo
bound de fase e preserva **as duas** estatísticas. A esfera preserva apenas E,
e outras estatísticas simples a detectam melhor.

A checagem independente do TEST reconstruíu 7.704 pares: erro angular máximo
8,96×10⁻⁵ graus, erro máximo de norma quadrada no círculo 1,36×10⁻⁶, e nenhuma
hipótese canônica mudou. O agrupamento exploratório dos 642 nomes por prefixo
antes do número da sentença gera 331 grupos e mantém positivos os intervalos
ajustados do círculo: [2,993; 5,225] para caption real e [1,190; 3,116] para
caption deslocada. São grupos de nomes, não identidades de signers verificadas.

O limite matemático também ficou explícito: E e N deixam um círculo de
representantes. Já a matriz de Gram completa, em rotação fixa e escalas
positivas, determina rho/sigma/tau e portanto as features. Não transformamos
preservação de duas estatísticas em promessa de invisibilidade geral.

O parecer estava certo sobre recuperação de colapso e fragilidade do endpoint
original. Discordamos de tornar exploração espontânea requisito absoluto para
um estudo de vulnerabilidade, de tratar dez seeds como critério universal de
CVPR, e de reformular detectabilidade como evasão sem teste. A rodada agora
oferece um teste positivo sob estatísticas fixas, além dos ataques condicionados
em três corpora. O paper é reenquadrado nessa evidência.

Ainda faltam um leitor BT forte de outra família e/ou avaliação linguística
independente para ampliar o impacto demonstrado. Transferência para o gerador
público e transmissão semântica no treino comum continuam inconclusivas.
Referências ajudam a posicionar o resultado; não fornecem garantia de aceite
nem probabilidade calibrada para CVPR.
