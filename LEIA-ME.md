# Pacote de Replicação

**Artigo:** *Cosseno 0,99 não é compreensão: anisotropia, polaridade e a competência jurídica diante da recuperação densa*
**Autor:** Pedro Borges Mourão

Este pacote contém tudo o que é necessário para verificação independente dos
resultados do artigo. Cada número reportado no paper pode ser reproduzido a partir
dos scripts e dados aqui incluídos.

## VERSÃO ATUAL: v5 (resposta à 3ª revisão adversarial)
Use `artigo_v5.pdf` / `artigo_v5.tex`. v1 a v4 ficam no histórico (pasta 5).

A 3ª revisão reproduziu o MiniLM ao milésimo e achou DOIS números fantasma — não
reproduzíveis por nenhuma definição nos dados — que sustentavam o abstract. Ambos
corrigidos na v5:
1. O "0,27" (gap do MiniLM) não existia; as definições reais dão +0,188 (bruto) ou
   -0,038 (só polo). APOSENTADO. `recalculo_canonico_gaps.py` recomputa todos os
   gaps por estrato com IC95 e gera `gaps_canonicos.json`.
2. O "8/10" da recuperação dos antigos não existia; os dados dão 19/20, 16/20,
   19/20. Figura regenerada por `regen_figC.py`.

O achado que a correção revelou é mais forte que o erro: o gap negativo da inversão
de polo é UNIVERSAL (4 modelos), não só do e5. A fragilidade de polaridade é
invariante estrutural; o e5 apenas a esconde sob anisotropia. Demais correções
(ataques 3-8): paradoxo reancorado na margem, falsos positivos do cross-encoder
reportados, módulos de prudência (dependência do controle, parser de reasoning,
família/língua única, ausência de baseline humano).

---

## Estrutura do pacote

```
1_artigo/                  manuscrito (PDF) e fonte LaTeX
2_scripts_replicacao/      scripts que geram cada resultado
3_dados_brutos/            saídas brutas (JSON) que sustentam os números
4_figuras_e_tabelas/       figuras (PDF) e tabelas (TeX) que o artigo compila
5_historico_versoes/       v1 a v3 do paper (rastreabilidade da evolução)
6_revisoes_adversariais/   parecer adversarial que guiou as revisões
```

---

## Dependências

**Python** (todos os experimentos rodam em CPU, sem GPU):
```
pip install sentence-transformers matplotlib numpy anthropic
```
- `sentence-transformers` (testado v5.x) — modelos de embedding e cross-encoder
- `anthropic` — apenas para o experimento de verificação por raciocínio (Seção 9)
- `matplotlib`, `numpy` — figuras e cálculo

**LaTeX** (para recompilar o PDF): TeX Live com `babel-portuges`. O preâmbulo
usa `microtype` com `expansion=false` para compatibilidade com Computer Modern.

**Modelos** (baixados automaticamente do Hugging Face na primeira execução):
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- `sentence-transformers/distiluse-base-multilingual-cased-v2`
- `intfloat/multilingual-e5-large` (~2,2 GB)
- `cross-encoder/nli-deberta-v3-base`

**API** (apenas Seção 9): chave Anthropic em `ANTHROPIC_API_KEY`. Modelos usados:
`claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`.

---

## Mapa de verificação: cada resultado → script → dado

| Resultado no paper | Script (em `2_scripts_replicacao/`) | Dado bruto (em `3_dados_brutos/`) |
|---|---|---|
| Conjunto de 40 pares mínimos (Tab. 1) | `dataset_v2.py` | — (embutido no script) |
| Cosseno por estrato, 3 modelos + IC95 (Tab. 3, Fig. 1) | `experimento_v2.py` | `brutos_v2.json`, `agg_v2.json` |
| Recuperação temática (Seção 6) | `recuperacao_v2.py` | `recup_v2.json` |
| Cross-encoder NLI por estrato (Seção 7, Tab. 4, Fig. 4) | `figura_crossencoder.py` | `nli_v2.json`, `crossencoder_tabela.json` |
| e5: cosseno bruto + baseline anisotrópico (Seção 5) | `experimento_e5_standalone.py` | (gera `resultado_e5.json`) |
| e5: normalização HEROS + anisotropia (Seção 5, Fig. 2) | `experimento_e5_normalizado.py` | (gera `resultado_e5_norm.json`) |
| Robustez de prefixo query/passage (nota de rodapé, Seção 4) | `teste_prefixo_e5.py` | (gera `teste_prefixo_e5.json`) |
| Matriz consolidada dos 4 modelos (Fig. 1) | `consolida_4modelos.py` | `consolidado_v3.json` |
| Verificação por raciocínio, 3 modelos × 2 formulações (Seção 9, Fig. 5) | `experimento_reasoning_polo_multi.py` | `resultado_multi.json` |
| Análise dos resultados de raciocínio | `analise_resultado_colado.py` | `resultado_multi.json` |
| Auditoria dos 16 pares + gap normalizado do e5 sobre subconjunto limpo (Seção de Limitações) | `remedicao_gap_limpo.py` | `gap_limpo.json` |
| Figuras v3 (heatmap, e5 bruto×norm, paradoxo) | `figuras_v3.py` | `consolidado_v3.json` |

---

## Como reproduzir

### A. Recompilar o PDF a partir da fonte
```
cd 1_artigo
# copie as figuras e tabelas necessárias para o mesmo diretório:
cp ../4_figuras_e_tabelas/*.pdf ../4_figuras_e_tabelas/*.tex .
pdflatex artigo_v4.tex && pdflatex artigo_v4.tex
```

### B. Reproduzir os números dos 3 modelos abertos (Seções 5–7)
```
cd 2_scripts_replicacao
python experimento_v2.py        # cosseno + IC95 → brutos_v2.json, agg_v2.json
python recuperacao_v2.py        # recuperação temática → recup_v2.json
```
Compare as saídas com os JSON de referência em `3_dados_brutos/`.

### C. Reproduzir o modelo de fronteira e5 (Seção 5, achado central)
```
cd 2_scripts_replicacao
python experimento_e5_standalone.py     # baixa e5 (~2,2 GB) na 1ª vez
python experimento_e5_normalizado.py    # normalização HEROS + anisotropia
python teste_prefixo_e5.py              # confirma que o baseline ~0,84 independe do prefixo
```
Resultado esperado: cosseno bruto da inversão de polo ~0,99; baseline anisotrópico
~0,84; gap normalizado ~0,09; recuperação 10/10 com margem mínima ~0,005.

### D. Reproduzir a verificação por raciocínio (Seção 9)
```
cd 2_scripts_replicacao
export ANTHROPIC_API_KEY="sua-chave"     # (Windows: $env:ANTHROPIC_API_KEY="...")
python experimento_reasoning_polo_multi.py    # 96 chamadas → resultado_multi.json
```
Ou, sem rodar a API, analise o `resultado_multi.json` já incluído:
```
python analise_resultado_colado.py
```
Resultado esperado (do `resultado_multi.json` incluído): sob "papel temático",
Opus 16/16, Sonnet 15/16, Haiku 13/16; sob "contradição", 3/16, 0/16, 7/16;
recuperação dos 7 pares recíprocos: 7/7 (Opus e Sonnet), 5/7 (Haiku).

---

## Notas sobre verificabilidade e limites

- **Determinismo:** os scripts de embedding usam `random.seed(42)` no bootstrap e
  são determinísticos. O experimento de raciocínio (Seção 9) chama modelos
  generativos e pode variar levemente entre execuções; o `resultado_multi.json`
  incluído contém a execução reportada no paper, com as respostas brutas dos
  modelos no campo `brutos`.
- **n pequeno:** o conjunto tem 40 pares (16 de inversão de polo). O paper reivindica
  o padrão, não diferenças de poucos pares entre modelos. Ver Seção de Limitações.
- **Pares imperfeitos:** a inspeção das respostas (Seção 9) revelou que alguns pares
  de inversão de polo, ao trocar mecanicamente os termos, geram frase juridicamente
  incoerente (p. ex. credor/devedor). Isto está registrado como limitação; é ruído
  de construção a corrigir em versão ampliada, e não altera o padrão agregado.
- **Auditoria dos pares de inversão de polo (resolvida):** os 16 pares foram
  auditados manualmente (12 limpos, 3 limítrofes, 1 impossível — credor/devedor).
  O padrão sobrevive ao subconjunto limpo: cross-encoder 7/12 (vs 9/16), reasoning
  por papel temático Opus/Sonnet 12/12 e Haiku 10/12. O `remedicao_gap_limpo.py`
  remede o gap normalizado do e5 e o resultado (em `gap_limpo.json`) é o achado mais
  forte do paper: o gap é NEGATIVO (-0,016 nos 16; -0,034 nos 12 limpos), e mais
  forte nos limpos — o e5 considera a inversão de polo mais próxima da identidade
  que duas paráfrases legítimas. A depuração fortaleceu a tese.
- **Pendência declarada no paper:** rodar `teste_prefixo_e5.py` e inserir os dois
  baselines (query/query e passage/passage) na nota de rodapé da Seção 4; o script
  está incluído e a fundamentação atual repousa na documentação oficial do e5.

---

## Histórico de versões (pasta 5)

O paper evoluiu sob revisão adversarial sucessiva, documentada para transparência:
- **v1:** position paper inicial (afirmações sem dado próprio).
- **v2:** primeira versão empírica (controle lexical pareado, IC95, recuperação).
- **v3:** reposicionada após teste do e5-large (achado de anisotropia).
- **v4 (atual):** seção do cross-encoder com tabela, delimitação de escopo, e seção
  de verificação por raciocínio multimodelo.

O parecer adversarial que motivou a transição v1→v2 está em `6_revisoes_adversariais/`.
