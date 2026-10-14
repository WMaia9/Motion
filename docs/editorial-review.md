# Revisão editorial — uma afirmação, uma história

O manuscrito continua em `paper/current/`. Esta revisão reorganiza a evidência
já congelada; não acrescenta treino, não troca seeds e não escolhe checkpoints.

A afirmação central é: **avaliadores que recebem rotações 6D brutas podem mudar
seus scores quando a codificação muda, mesmo com posições e rotações idênticas**.
A intervenção preserva o movimento e a defesa preserva a informação rotacional.

## O que aprendemos com os premiados

A [lista oficial da CVF](https://www.thecvf.com/?page_id=413) confirma os três
papers abaixo como Best Paper do CVPR. Foram examinados os PDFs primários,
a primeira página, a sequência de seções, as figuras e os fechamentos. As
observações de escrita são nossa análise, não critérios oficiais de premiação.

| Paper premiado | Observação da apresentação | Aplicação nesta revisão |
|---|---|---|
| [VGGT, 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf) | Teaser de largura total antes do abstract. A figura apresenta a capacidade central; o texto organiza método e aplicações em torno dela. | Manter o teaser inicial, encurtar a legenda e fazer a figura comunicar movimento idêntico e score diferente. |
| [Generative Image Dynamics, 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Generative_Image_Dynamics_CVPR_2024_paper.pdf) | Teaser antes do abstract mostrando representação e uso. Os experimentos separam aplicações e ablações; limitações aparecem no fechamento. | Mostrar mecanismo, evidência e uso da defesa. Conservar ablações completas no suplemento, com escopo explícito no principal. |
| [Rich Human Feedback, 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Liang_Rich_Human_Feedback_for_Text-to-Image_Generation_CVPR_2024_paper.pdf) | Sem teaser antes do abstract. A introdução motiva uma falha concreta, apresenta a contribuição e depois demonstra sua utilidade e generalização. | Abrir pela falha de avaliação e seu efeito mensurável. Separar contribuição, generalização e custo da defesa. |

Não há um formato visual único dos vencedores. Adotamos clareza da afirmação,
hierarquia das figuras e progressão das provas; o kit CVPR oficial disponível
continua intacto. Essas escolhas não são uma promessa de prêmio ou aceite.

## Mudanças no paper

- Abstract com 157 palavras e quatro números, focado no resultado e na defesa.
- Introdução centrada em avaliação de movimento, com distinção explícita de MARDM.
- Primeira evidência: recodificação fixa, sem alvo, treino ou gradiente.
- Figura de recodificação e calibração física com a mesma referência reconstruída
  e seu custo informado, incluindo os controles menores e aleatórios.
- Auditoria natural apresentada em uma tabela: oito releases HumanML3D e dois
  settings KIT. Gaps positivos e negativos permanecem, com seus intervalos.
- Uma tabela reúne os ataques Guo/TMR; os diferentes tamanhos de coorte ficam
  explícitos e não são tratados como comparação de acurácia entre arquiteturas.
- Ataque cego e transferência em uma figura que mostra todos os bounds,
  incluindo a piora no bound maior e a sobreposição das coortes MDM.
- Três corpora de sinais reunidos em uma tabela. Acesso ao texto-fonte,
  tokenizações e igualdade das hipóteses canônicas são explícitos.
- Resultados de fechamento e custo da defesa continuam no corpo.
- Interface de avaliação explicita canonicalização antes da normalização,
  com conclusão própria e orientada à adoção do contrato.

Ablations da camada, figura adaptativa, orçamento, replay CSL e testes de treino
ficam no suplemento. Os pilotos de recompensa malsucedidos permanecem lá com
todos os resultados e sua interpretação correta. O principal não promete
exploração durante treino normal. A comparação PHOENIX contra movimento real
não é usada para sugerir superioridade; seu teste inconclusivo permanece no
suplemento. A defesa não é apresentada como robustez contra qualquer ataque.

Os limites que mudam a interpretação acompanham a respectiva afirmação:
acesso ao alvo; diferenças entre protocolos; sinais variáveis; coortes com
sobreposição; referência FID pareada; custo de reconstrução; leitores BT fracos;
uso dos TEST públicos durante o desenvolvimento. Mover detalhes não os elimina.

## Artefatos

O mapa de mudanças e as figuras/tabelas derivadas de JSONs congelados ficam no
arquivo local `reports/cvpr_editorial_review_20261013/`. Essa pasta é um snapshot
de revisão, não outro manuscrito de trabalho. PDFs revisados e proveniência
continuam em `paper/current/reviewed/` e `paper/current/provenance.json`.

O principal final tem sete páginas de conteúdo e duas de referências; o
suplemento tem vinte mais duas. As 31 páginas foram inspecionadas, incluindo
legibilidade, margens, referências e ordem de floats. A conclusão permanece
inteira na última página de conteúdo. Fontes estão incorporadas e o estilo
CVPR permanece intacto. A auditoria de afirmações verifica números e acesso
ao alvo contra as evidências congeladas.

Autores, registro e leitura fria humana continuam adiados conforme solicitado.
Não houve submissão, publicação ou commit automático. O formato final CVPR 2027
continua provisório até a disponibilidade de suas instruções completas.
