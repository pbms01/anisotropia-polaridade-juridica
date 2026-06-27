# Parecer de revisão adversarial — *Uma leitura linguística do erro*

Revisor simulado: hostil, perfil cs.CL, mandato de rejeitar. Cada ataque que era
testável foi executado com código. Vereditos ancorados em dados, não em opinião.

---

## Sumário executivo

O paper sobrevive como **contribuição válida**, mas **três afirmações na sua forma
atual estão empíricamente infladas** e seriam corretamente atacadas por um revisor.
A correção não destrói a tese; reposiciona-a de "falha determinística do
bi-encoder" para "sinal de polaridade tênue e frágil, com risco real em
recuperação mas não colapso garantido". Esta versão é mais defensável e, por ser
mais honesta, mais difícil de refutar.

Dois achados novos obrigam revisão antes de submeter:

1. **O controle de paráfrase original era lexicalmente injusto** (Jaccard 0,40 vs
   0,79 dos contrastivos). Corrigido o pareamento, o gap real é maior do que o
   reportado — ou seja, MiniLM e mpnet separam paráfrase de inversão **melhor** do
   que o paper afirma. A Figura 2 exagera a falha.
2. **A ponte cosseno→recuperação não se sustenta na forma forte.** Em recuperação
   temática realista, os modelos rankeiam a frase correta acima da oposta em 9–10
   de 10 casos. A afirmação "recupera adversários como aliados" precisa virar
   "a margem que separa aliado de adversário é estreita e frágil".

---

## Ataque 1 — Controle de paráfrase confunde duas variáveis · **PROCEDE**

**A acusação.** O paper conclui "cosseno alto nos contrastivos é erro porque o
controle também é alto". Mas contrastivos e controles não diferiam só na dimensão
preserva/inverte sentido: diferiam na sobreposição lexical. Medido:

| Grupo | Jaccard médio |
|---|---|
| Pares contrastivos | 0,791 |
| Controle original (ctl-01..03) | 0,400 |

O controle tinha metade da sobreposição. Logo era um baseline injusto: pedia
cosseno alto com menos pista lexical. Parte do cosseno alto nos contrastivos é
mero artefato de quase-identidade de superfície, não falha de discriminação
semântica.

**Teste de defesa.** Construído controle pareado (Jaccard 0,767, comparável aos
0,791). Remedição:

| Modelo | Contrastivo μ | Controle pareado μ | Gap real |
|---|---|---|---|
| MiniLM | 0,708 | 0,920 | **+0,212** |
| mpnet | 0,674 | 0,940 | **+0,266** |
| distiluse | 0,900 | 0,919 | +0,018 |

**Veredito.** O ataque procede e **corrige o paper contra a própria tese**: com
pareamento justo, MiniLM e mpnet separam muito melhor do que a Figura 2 sugere
(gap 0,21–0,27, não 0,09–0,15). A falha real é mais estreita: o cosseno absoluto
dos contrastivos permanece em 0,67–0,71 (alto para recuperação), mas o modelo
*distingue na média*. Só distiluse não separa (gap 0,018). **A Figura 2 deve ser
substituída pelo controle pareado.**

---

## Ataque 2 — n=4 não sustenta "estrato mais perigoso" · **PROCEDE**

O abstract e a §5.1 afirmam que a inversão de polo é "o estrato mais perigoso".
Repousa em **4 pares**. Valores consistentes (0,86–1,0), o que ajuda, mas
afirmação superlativa em abstract a partir de n=4 é frágil e seria marcada.

**Veredito.** Procede. Duas saídas: (a) ampliar inversão_polo para n≥15, ou (b)
rebaixar a linguagem de "é o mais perigoso" para "foi o estrato de maior cosseno
nesta amostra (n=4), o que merece investigação dedicada". Recomendo (a) antes de
submeter; (b) como mínimo inegociável.

---

## Ataque 3 — Sem teste de significância; incerteza não reportada · **PROCEDE**

Bootstrap (5000 reamostragens, controle original):

| Modelo | Gap | IC95 | Significância |
|---|---|---|---|
| MiniLM | +0,089 | [+0,000, +0,181] | no fio (toca zero) |
| mpnet | +0,154 | [+0,054, +0,256] | exclui zero |
| distiluse | −0,049 | [−0,142, +0,024] | inclui zero |

**Veredito.** Procede. O veredito "o bi-encoder não separa" é **heterogêneo por
modelo**, não uniforme: forte só para distiluse, marginal para MiniLM, e o mpnet
chega a separar. O paper deve reportar IC e abandonar a generalização única.

---

## Ataque 4 — Cosseno alto não implica erro de recuperação · **PROCEDE (o mais grave)**

**A acusação.** O paper salta de "cos(A, ¬A)=0,7" para "recupera o documento
errado". Ilegítimo: recuperação decide por *ranking contra query*, não por
similaridade par-a-par. Pergunta certa: dada uma query temática, a negação da
resposta é rankeada acima da resposta?

**Teste.** 10 casos, query temática expressando intenção de polaridade, corpus =
{frase correta, frase oposta}:

| Modelo | correto > oposto | margem média | margem mínima |
|---|---|---|---|
| MiniLM | **10/10** | +0,230 | +0,051 |
| mpnet | 9/10 | +0,238 | −0,103 |
| distiluse | 9/10 | +0,074 | −0,000 |

**Veredito.** Procede e **derruba a forma forte da tese aplicada**. Os modelos
distinguem no ranking mesmo com cosseno par-a-par alto, porque a query tem cosseno
ainda maior com a frase correta. O paper mediu o nível errado (par-a-par) para a
conclusão que tirou (recuperação). A afirmação "recupera adversários como aliados"
**não se sustenta como descrita**.

**O que sobrevive.** A margem é estreita (mínimo +0,05 em MiniLM; mpnet e distiluse
já erram 1/10). Sob query temática pobre (só o tema, sem intenção de polaridade), o
caso "tutela urgência" deu margem de **+0,003** — quase empate entre deferido e
indeferido. Logo a tese defensável é de **fragilidade**: o sinal de polaridade
existe mas é tênue e degrada com query pobre, ruído ou documento longo. É uma
afirmação de risco, não de falha garantida.

---

## Ataque 5 — Robustez sob query degradada · informativo

Query pobre (2 palavras, só tema), MiniLM: 5/5 acertos, mas margem de +0,003 a
+0,261. Confirma o Ataque 4: há queries realistas no fio da navalha, sem virar
falha determinística. Sustenta a versão fraca, refuta a forte.

---

## Ataques que NÃO procedem (defesa do paper)

- **"O cross-encoder salva tudo, então não há problema."** Não procede. O
  cross-encoder detecta contradição (19/20) mas é O(n²) e não indexável: inviável
  para corpus grande sem recuperação prévia por bi-encoder. A vulnerabilidade na
  primeira etapa permanece arquiteturalmente real.
- **"São modelos velhos; os novos resolveram."** Parcial. O paper já limita o
  escopo a modelos abertos em produção e nomeia a extensão. Honesto, não fatal.
- **"Cosseno 0,7 é normal e bom."** Não procede como refutação: o ponto não é o
  valor absoluto, é a falta de separação suficiente sob pareamento lexical, que o
  Ataque 1 confirma para distiluse.

---

## Veredito por afirmação do paper

| Afirmação no paper | Status pós-revisão |
|---|---|
| Bi-encoders dão cosseno alto a pares de sentido oposto | **Mantida** (0,67–0,90) |
| Negação adverbial é o estrato menos problemático | **Mantida e reforçada** |
| Inversão de polo é "o estrato mais perigoso" | **Rebaixar** (n=4) |
| O bi-encoder não separa inversão de paráfrase | **Corrigir** — separa em 2/3 modelos sob pareamento justo |
| Cross-encoder recupera a distinção (mitigação cara) | **Mantida** (19/20) |
| Sistema "recupera adversários como aliados" | **Reformular** — vira fragilidade, não falha |
| Vulnerabilidade é arquitetural | **Mantida** |

---

## Plano de correção mínimo antes de submeter

1. Substituir Figura 2 e a §5.1 pelo **controle pareado** (gap 0,21/0,27/0,02).
2. Adicionar **§ de recuperação** com o teste temático (10/10, 9/10, 9/10) e
   reformular toda menção a "recupera o errado" como **margem estreita e frágil**.
3. Reportar **IC95 bootstrap** em todas as comparações de gap.
4. Ampliar inversão_polo para n≥15 **ou** rebaixar a linguagem superlativa.
5. Reescrever abstract: trocar "não distingue" por "distingue por margem tênue que
   colapsa sob query pobre", e manter o achado forte e honesto (a separação existe
   no texto e é o passo de compressão que a estreita).

Resultado líquido: um paper mais fraco em retórica, mais forte em ciência, e
**aprovável** porque antecipa e desarma seus próprios revisores.
