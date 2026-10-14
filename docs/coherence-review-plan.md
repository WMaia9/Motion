# Controles de direção e restauração da evidência natural

Esta rodada responde somente à lista fechada do parecer pós-MARDM. O manuscrito
continua em `paper/current/`; novos diretórios em `reports/` são arquivos de
evidência, sem novos locais de edição do paper.

## Direções fixas em blocos não nulos

Usar os mesmos vinte conjuntos reais salvos, 4.640 entradas por repetição,
legendas, comprimentos, máscaras, normalizações e pesos de Guo/TMR. Nenhum novo
gerador, treino, seed de modelo ou ajuste de recompensa.

Adicionar rotações, velocidades, posições e contatos com ambos os sinais e
os dois orçamentos existentes. O raio passa a se chamar `kappa` no manuscrito,
reservando `rho` à escala da fibra. A norma por clipe permanece
`||(y-x)/std_do_leitor||₂ = kappa*sqrt(frames_válidos)`, antes da seleção de
features. As unidades dos dois leitores não são intercambiáveis.

A direção local `(1,1,1)/sqrt(3)` é fixa em todas as articulações e frames.
Nas rotações, a mudança SO(3) usa esse eixo e um ângulo por clipe resolvido
somente pela norma, até pi/2. Posições e velocidades recebem deslocamentos
constantes nas coordenadas não normalizadas. Contatos usam a direção com
quatro entradas iguais; podem sair do domínio binário. Guo não lê contatos.
Direção constante no tempo não implica um movimento fisicamente consistente
entre blocos: os demais canais ficam protegidos, como nos controles originais.

As dezesseis condições por leitor são retidas: quatro blocos, dois sinais,
dois raios. Cada família prospectiva cobre 32 endpoints, com intervalos
pareados t19 ajustados por Bonferroni: FID/R1 em Guo, R1/cosseno em TMR.
Contrastes fixo menos aleatório usam os resultados antigos nas mesmas
repetições, numa família secundária de 32 endpoints. A família histórica de
28 endpoints não é reescrita nem reinterpretada como um único teste prospectivo.

O protocolo foi escrito antes dos scores. A primeira tentativa encontrou
quantização de offsets constantes em contatos binários float32. Os arquivos
parciais e fontes anteriores foram preservados. A tolerância relativa de norma
é 2e-5 para contatos, 5e-6 para os demais blocos; fórmulas, raios, sinais,
pesos e dados não mudaram. Todas as condições finais são executadas novamente.

A verificação independente reconstrói os primeiros oito fixtures por
condição/repetição com SciPy, confere todas as normas salvas, proteção de
canais/padding, forwards nativos, métricas e intervalos. Não se afirma replay
geométrico completo de arrays não salvos. Interpretação prevista: direções
coerentes fora da fibra podem ter scores grandes; a particularidade da fibra
é alterar a codificação sem alterar rotações ou qualquer outro bloco.

## Dependência natural em sinais

Recuperar hipóteses brutas/canônicas dos arquivos congelados, sem novo decode
ou treino. Recontar todos os pares e recomputar os seis BLEUs com contagem
independente de n-gramas. Registrar a normalização de caracteres no CSL.
Os caminhos comparados compartilham o pré-processamento nativo da mão esquerda;
“sem ataque” descreve estes inputs reais, sem afirmar bytes intocados da release.

Restaurar os resultados no corpo, na tabela de sinais e no suplemento. Mudança
de frases com BLEU agregado próximo não demonstra melhora ou piora semântica.
Destacar esse resultado sem alvo no abstract e motivação de domínios só de
rotações. Conferir todos os usos de `rho`/`kappa`, figures e tabelas.

## Fechamento

Congelar os dois estudos completos, preservar os arquivos anteriores, atualizar
as figuras a partir de JSONs verificados e manter todos os sinais, raios e
controles acessíveis. Compilar com o kit disponível, conferir oito páginas de
corpo, referências, figuras, PDFs, fontes e repositório leve. Registro,
autores e leitura fria humana continuam sob responsabilidade do usuário.
