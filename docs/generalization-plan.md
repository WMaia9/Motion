# Rodada final de generalização — iniciada em 13/10/2026

O objetivo é testar uma falha mensurável de consistência, sem prometer inflação
sistemática, exploração em treino normal ou qualidade humana melhor. Os
relatórios congelados anteriores permanecem imutáveis. O manuscrito ativo é
somente `paper/current/`; o PDF revisado anterior fica válido até a nova entrega.

1. **TMR:** arquitetura ACTOR/Transformer e DistilBERT, pesos oficiais com MD5
   verificado. Vinte repetições de todos os arrays públicos e de movimento real;
   bruto/canônico, recodificações direcionais pequenas/grandes e aleatórias.
   Inferência nativa com média da VAE, normalização própria e cosseno. As coortes
   são as do nosso loader; não reivindicamos o protocolo completo de anotação
   AMASS/segmentos/filtragem do benchmark oficial TMR. Ataques com alvo explícito
   e controle genérico de L2 são capacidade construída, separados da auditoria.
2. **Calibração física:** ângulos locais 0,1/0,3/1/3/10/30 graus, eixos fixos por
   junta/clipe ou independentes por frame. FK, posições RIC, velocidades e contatos
   são atualizados juntos. A reconstrução de ângulo zero é pontuada e comparada
   ao dado original; não ocultamos seu custo. Todas as curvas compartilham a
   referência reconstruída, com a mesma crop de quatro frames para velocidades.
   Isso não define uma conversão universal entre FID e graus nem percepção humana.
3. **Treino com recompensa:** grade finita de lr/peso, controle MSE e escolha
   em DEV antes de TEST. Se nenhum ajuste melhorar retrieval em amostragem livre,
   reportar que o piloto não conseguiu fazê-lo; não concluir ausência do canal.
   MotionReward consome as 263 features, incluindo 6D, sem canonicalização,
   através de sua projeção própria. Sobrevivência dos ganhos sob canonicalização
   não prova qualidade física/humana nem ausência de qualquer uso do canal.
4. **Mais releases/KIT:** os downloads oficiais MoMask HumanML3D e KIT agora
   funcionaram. MotionDiffuse e KIT serão auditados conforme os assets e as
   verificações nativas disponíveis. Checkpoints aparentados não serão contados
   como novas arquiteturas para alcançar artificialmente oito métodos.
5. **Ataque cego/transferência:** repetir sobre outputs MLD e pontuar todos os
   outputs com os dois checkpoints Guo, sem otimizar contra o leitor de destino.
   Fixar amostras, budgets e condições antes da pontuação; mostrar ambos os budgets.
6. **Custo da defesa:** R1 real bruto/canônico nos leitores Guo e TMR, com
   incerteza entre as vinte repetições e diferença de embedding medida.

A escrita será reorganizada com uma motivação geral, abstract com no máximo
seis valores numéricos e resultados em parágrafos curtos. Reportar mudanças
absolutas e sinal junto das margens publicadas, ressalvando protocolos diferentes.
O estudo CVPR2025 de Meng et al. é antecedente direto: já mostrou viés devido
à redundância e treinou leitores sobre dimensões essenciais. Nossa distinção
precisa ser preservação de posições **e rotações/FK**, canal da própria codificação
6D e canonicalização que mantém a informação rotacional. Não reivindicar o
primeiro diagnóstico de viés devido à redundância.

Correções factuais do parecer: o T2M no segundo leitor tem gap médio −0,00810,
não −0,011 (este é aproximadamente um extremo do intervalo). As coortes novas
contêm 769/761 grupos; sua interseção é 493. As sobreposições com a primeira são
494/496, que não são o total de grupos. A dependência continua sendo declarada.

Depois destes testes e das correções, congelar a entrega. Não ampliar a grade
por perseguir um resultado positivo. Autores, ID de submissão e leitura fria por
pessoas do laboratório exigem participação humana; não serão inventados.
Datas oficiais: registro10/11/2026, envio16/11/2026 e suplemento23/11/2026, AoE:
https://cvpr.thecvf.com/Conferences/2027/Dates
