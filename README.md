# Cosseno 0,99 não é compreensão

Pacote de replicação do artigo *Cosseno 0,99 não é compreensão: anisotropia, polaridade e a competência jurídica diante da recuperação densa*, de **Pedro Borges Mourão**. Versão atual: **v5**.

Cada número reportado no paper pode ser reproduzido a partir dos scripts e dados aqui incluídos. O guia detalhado de verificação (mapa resultado → script → dado e passo a passo) está em [`LEIA-ME.md`](LEIA-ME.md).

## O argumento em uma frase

Similaridade de cosseno alta entre dois textos não é evidência de equivalência semântica: modelos de embedding denso preservam a polaridade jurídica (afirmação contra negação, autor contra réu) como um sinal estruturalmente fraco, e a fraqueza atravessa as gerações de modelos.

## Achado central

Em quatro modelos abertos (MiniLM, mpnet, distiluse e o de fronteira multilingual-e5-large), a inversão de polo processual (autor para réu) tem gap normalizado negativo ou nulo: o modelo trata uma sentença e sua inversão de polo como mais próximas da identidade do que duas paráfrases legítimas. No e5-large isso fica encoberto pela anisotropia (cosseno em torno de 0,84 entre sentenças jurídicas quaisquer, e 0,99 entre uma sentença e sua negação); normalizada pelo baseline, a fragilidade reaparece. O padrão é um invariante estrutural, não uma idiossincrasia do e5.

## Novidades da v5 (resposta à 3ª revisão adversarial)

A terceira revisão reproduziu o MiniLM ao milésimo e encontrou dois números que não se reproduziam por nenhuma definição nos dados. Ambos corrigidos:

- O gap de 0,27 do MiniLM foi aposentado; as definições reais dão +0,188 (bruto) ou -0,038 (só polo). O `recalculo_canonico_gaps.py` recomputa todos os gaps por estrato com IC95 e gera `gaps_canonicos.json`.
- A recuperação dos modelos antigos passou de "8/10" para os valores reais 19/20, 16/20 e 19/20, com a figura regenerada por `regen_figC.py`.

A correção fortaleceu a tese: o gap negativo da inversão de polo é universal aos quatro modelos, e o e5 apenas o esconde sob anisotropia.

## Estrutura do pacote

- `1_artigo/` manuscrito v5 (PDF) e fonte LaTeX
- `2_scripts_replicacao/` scripts que geram cada resultado
- `3_dados_brutos/` saídas brutas (JSON) que sustentam os números
- `4_figuras_e_tabelas/` figuras (PDF) e tabelas (TeX) que o artigo compila

## Dependências

Todos os experimentos de embedding rodam em CPU, sem GPU:

```
pip install sentence-transformers matplotlib numpy anthropic
```

Os modelos (MiniLM, mpnet, distiluse, `multilingual-e5-large` com cerca de 2,2 GB e o cross-encoder `nli-deberta-v3-base`) são baixados do Hugging Face na primeira execução. A verificação por raciocínio (Seção 9) requer a chave da API Anthropic em `ANTHROPIC_API_KEY`. Recompilar o PDF exige TeX Live com `babel-portuges`.

## Reprodução rápida

```
cd 2_scripts_replicacao
python experimento_v2.py             # cosseno + IC95 nos 3 modelos abertos
python recalculo_canonico_gaps.py    # gaps canônicos por estrato + IC95 -> gaps_canonicos.json
python experimento_e5_standalone.py  # e5-large: cosseno bruto da inversão de polo
python experimento_e5_normalizado.py # normalização HEROS + anisotropia
```

Compare as saídas com os JSON de referência em `3_dados_brutos/`. O mapeamento completo de cada figura e tabela está em [`LEIA-ME.md`](LEIA-ME.md).

## Como citar

```bibtex
@unpublished{mourao2026cosseno,
  author = {Mour{\~a}o, Pedro Borges},
  title  = {Cosseno 0,99 n{\~a}o {\'e} compreens{\~a}o: anisotropia, polaridade e a compet{\^e}ncia jur{\'i}dica diante da recupera{\c c}{\~a}o densa},
  year   = {2026},
  note   = {Preprint, v5}
}
```

## Licença

O código em `2_scripts_replicacao/` está sob a licença MIT (ver [`LICENSE`](LICENSE)). O texto e as figuras do artigo (`1_artigo/`, `4_figuras_e_tabelas/`) permanecem de autoria de Pedro Borges Mourão e são redistribuídos aqui para fins de verificação independente.
