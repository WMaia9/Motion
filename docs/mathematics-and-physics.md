# Pesquisa: ataques aprendidos, geometria e física

Foram consultadas e arquivadas quatorze fontes primárias, incluindo PDFs completos,
metadados dos publishers e implementações oficiais. A pesquisa separa antecedentes
já estabelecidos de hipóteses testáveis no nosso problema. Não chamamos de nova
matemática a representação6D, geodésicas emSO(3), simetrias gauge ou difusão em
variedades.

## Antecedentes que precisam entrar no artigo

| Fonte primária | O que estabelece | Consequência para nós |
|---|---|---|
| [Universal Adversarial Perturbations, CVPR2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.html) | Perturbações compartilhadas entre muitas entradas | Ataques transferíveis/universais antecedem nosso estudo; a nossa camada condicionada por clip/texto não é uma perturbação universal fixa |
| [Generative Adversarial Perturbations, CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Poursaeed_Generative_Adversarial_Perturbations_CVPR_2018_paper.html) | Redes treinadas produzem ataques em uma passagem, incluindo ataques com alvo | Aprender uma rede para atacar outra não é novidade isolada; nossa restrição é equivalência física declarada |
| [Adversarial Transformation Networks, AAAI2018](https://ojs.aaai.org/index.php/AAAI/article/view/11672) | Transformações adversariais amortizadas e controle da saída | A crítica ao alvo disponível é pertinente; isso define um diagnóstico de capacidade, não a ocorrência em treinamento comum |
| [One Perturbation is Enough, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Fang_One_Perturbation_is_Enough_On_Generating_Universal_Adversarial_Perturbations_against_ICCV_2025_paper.html) | Ataques universais contra alinhamento de visão e linguagem | Manipular captions também tem antecedentes; não anunciar a ideia genérica como pioneira |
| [DRaFT, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html) | Fine-tuning de difusão por gradientes de rewards diferenciáveis | Motiva investigar objetivos de geração reais, além do módulo adversarial separado |
| [EasyTune, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html) | Otimização de rewards de movimento em passos de denoising | O regime de reward por predição denoised tem uso prático; nossos experimentos auditam uma incompatibilidade de representação, sem reproduzir esse método |
| [SoliReward, CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lian_SoliReward_Mitigating_Susceptibility_to_Reward_Hacking_and_Annotation_Noise_in_CVPR_2026_paper.html) | Reward hacking e robustez de modelos de avaliação de vídeo | Robustez de rewards não é um tema novo; precisamos de um mecanismo específico e evidência causal |

A distinção central é que a transformação conhecida permite segurar as rotações
fixas, em vez de presumir que uma perturbação perceptualmente pequena preserva
o conteúdo. A camada com caption disponível deve ser apresentada como ataque
amortizado condicionado. A experiência do denoiser comum é uma questão separada.

## Geometria: por que trocar a loss é um teste relevante

[On the Continuity of Rotation Representations](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html)
já define o converter6D usado. [RigMo, CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_RigMo_Unifying_Rig_and_Motion_Learning_for_Generative_Animation_CVPR_2026_paper.html)
usa supervisão geométrica em animação, incluindo a distância geodésica de rotações,
em um objetivo composto. Isso motiva a escolha de uma loss geométrica; não prova
que o objetivo completo de RigMo deixe coordenadas específicas livres.

Para d=(a,b), as direções(a,0),(0,b),(0,a) alteram somente a representação da
mesma rotação. Se L depende de R(d), sua derivada nessas direções é zero em
aritmética exata. MSE entre features6D canônicas geralmente tem derivada e
curvatura nessas direções: ela escolhe um representante, além de uma rotação.
Essa é uma consequência conhecida da regra da cadeia e do converter.

O experimento correto mede esse efeito, em vez de declarar que o resultado
negativo anterior era garantido. Usamos o mesmo denoiser, inicialização, dados,
ruído e minibatches; variamos MSE/geodésica, CE bruta/canônica e pesos. A loss
geodésica é média de ângulos ao quadrado em radianos, calculados por atan2.
Não inserimos um módulo especial de gauge no gerador desse teste. Reportamos
mudanças físicas/canônicas e diferenças entre os caminhos raw/canonical de
cada saída. Não pressupomos que dois geradores treinados produzam a mesma
rotação. Os pontos finais5/20épocas são fixos e todos os resultados ficam salvos.

## Difusão em variedades e espaços quocientes

[Riemannian Score-Based Generative Modelling, NeurIPS2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/105112d52254f86d5854f3da734a52b4-Abstract-Conference.html)
e [Riemannian Diffusion Models, NeurIPS2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/123d3e814e257e0781e5d328232ead9b-Abstract-Conference.html)
já constroem difusão sobre variedades. [Riemannian Diffusion Mixture, ICML2024](https://proceedings.mlr.press/v235/jo24a.html)
trata a geometria com processos de ponte. [Quotient-Space Diffusion Models, ICLR2026](https://arxiv.org/abs/2604.21809v2)
remove redundância por simetria em geração molecular e fornece um framework
teórico próprio. Portanto, criar simplesmente uma difusão emSO(3) ou dizer
“difusão no quociente” não basta para sustentar uma alegação de novidade.

Aplicação possível: modelar diretamente os observáveis físicos e usar um reader
compatível com esse contrato. Isso poderia eliminar o canal antes da avaliação.
Mas treinar uma difusão intrínseca envolve outro processo de ruído e sampler;
nossa troca de loss dentro do denoiser/sampler existente não é implementação
desses frameworks. Não apresentar a experiência atual como nova teoria de
score matching ou como reprodução de resultados daqueles artigos.

[Landing with the Score, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d1a6fe144bdff7b95138bca8097d4e69-Abstract-Conference.html)
obtém informação de variedade a partir de denoising quando sua geometria não
está explicitamente disponível. Nosso converter é conhecido; um projetor
aprendido acrescentaria incerteza a um controle que podemos calcular exatamente.
Esse trabalho oferece uma direção futura para representações cuja geometria
seja implícita, não uma justificativa para complicar o experimento atual.

## Física: simetria e observáveis

[Sampling the Schwinger Model with Gauge-Equivariant Diffusion, PAI2026](https://arxiv.org/abs/2606.27481v2)
usa simetriaU(1) em amostragem de uma teoria de campos e compara observáveis
com simulaçõesMCMC. O princípio útil aqui é distinguir coordenadas de descrição
de observáveis. O grupo, os objetivos e os dados daquele trabalho são distintos
dos nossos. Não estamos simulando uma teoria física nem aplicando um novo
teorema de Noether a língua de sinais.

Nossa tradução operacional desse princípio é testar duas coisas separadas:
(a) se a loss de movimento reconhece a classe de equivalência correta;
(b) se o avaliador recompensa graus de liberdade descartados por essa loss.
Isso dá um experimento falsificável com controle geométrico e hipóteses explícitas.

## Novo controle construído a partir dessa geometria

Escreva a=ρu e b=σv+τu para os vetores canônicosu,v, comρ,σpositivos.
Defina s=(ρ−1,σ−1,τ). Então a distância quadrática ao encoding canônico é
||s||². Uma rotação auxiliar de s preserva essa distância, embora altere a
representação da mesma rotação física. Com||s||<0,95, os novosρ,σ permanecem
positivos. Fora dessa região, a transformação é a identidade.

Uma rotação auxiliar em torno do eixo(1,1,0) também preservaρ+σ. Como
||d||²−||d−C(d)||²=2(ρ+σ)−2, ela preserva simultaneamente a norma6D total e a
distância ao canônico em cada frame/junta. A “rotação auxiliar” ocorre no espaço
de coordenadas redundantes; não roda uma junta do personagem. Essa distância
numérica não é energia cinética.

Registramos e testamos primeiro esse controle mais restrito, com três seeds e
controles de loss/condicionamento. O RMS ficou igual e o detector escalar porRMS
teveAUC≈0,50, mas o ganho da camada completa foi0,0649BLEU, com IC incluindo zero.
O resultado negativo foi preservado. Uma extensão exploratória ao conjunto
esférico completo mantém a distância ao canônico, mas libera a norma6D total;
ela foi registrada após o estudo do círculo. O ganho DEV foi −0,0525 BLEU, IC [−0,4955;0,3909], também negativo/inconclusivo. Ambos os estudos foram congelados e reportados juntos.
Não há garantia de evasão de detectores que examinem outras estatísticas.

## O que ainda não devemos afirmar

- Criptografia ou anonimização: não há chave, segurança criptográfica ou teste de identidade nesta rodada.
- Melhoria de sinais: ganho de BT por representação não mede melhoria de compreensão humana.
- ReproduçãoSOTA: o checkpoint público do SignSparK não foi treinado em PHOENIX; a tabelaGF/SR-T2P do paper excluiBT. O replay novo emCSL usa o pipeline público com dicionárioTRAIN, cujas diferenças do paper são declaradas pelos próprios autores.
- Superioridade sobre movimento real:15,21vs15,00 é comparação de pontos; o IC pareado da diferença[−0,8454;1,2357] inclui zero.
- Novidade mundial: os antecedentes acima impedem apresentar ataques amortizados, reward hacking ou difusão geométrica em si como invenções nossas.

A oportunidade científica é demonstrar uma incompatibilidade entre o objetivo
físico da produção e sua avaliação, explicar quando o treinamento a explora,
e medir o fechamento do canal e seus custos. Esse argumento precisa dos novos
resultados e dos controles, além de uma boa redação.

## Resultado do teste motivado pela pesquisa

O denoiser geodésico sem camada adversarial terminou a confirmação. No TEST, o excesso bruto sobre treino sem CE foi +2,80 [1,85;3,79], mas o gap absoluto e o contraste contra CE canônica incluem zero. Portanto, a pesquisa motivou um teste informativo, sem validar uma nova teoria ou transmissão semântica robusta no gerador. A camada condicionada confirmou ganhos em How2Sign e CSL; os ataques com distância constante e o transfer nativo permaneceram negativos/inconclusivos. Detalhes em [RESPOSTA_A_CRITICA.md](reviewer-response.md).
