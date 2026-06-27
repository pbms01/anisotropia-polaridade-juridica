# Cosseno 0,99 não é compreensão

Pacote de replicação do artigo *Cosseno 0,99 não é compreensão: anisotropia, polaridade e a competência jurídica diante da recuperação densa*, de **Pedro Borges Mourão**.

Cada número reportado no paper pode ser reproduzido a partir dos scripts e dados aqui incluídos. O guia detalhado de verificação (mapa resultado → script → dado e passo a passo completo) está em [`LEIA-ME.md`](LEIA-ME.md).

## O argumento em uma frase

Similaridade de cosseno alta entre dois textos não é evidência de equivalência semântica: modelos de embedding denso colapsam inversões de polaridade jurídica (trocar autor por réu, credor por devedor) em vetores quase idênticos, e o efeito sobrevive à correção de anisotropia.

## Achado central

No modelo de fronteira `intfloat/multilingual-e5-large`, a inversão de polo registra cosseno bruto de aproximadamente 0,99. Depois de corrigir a anisotropia do espaço (baseline em torno de 0,84), o gap normalizado fica negativo: cerca de -0,016 nos 16 pares de inversão e -0,034 no subconjunto limpo de 12 pares. Em outras palavras, o modelo considera a inversão de polo *mais* próxima da identidade do que duas paráfrases legítimas do mesmo enunciado.

A recuperação temática acerta 10 de 10. Já a verificação por raciocínio com LLMs depende inteiramente do enquadramento: sob a instrução de "papel temático", Opus acerta 16/16, Sonnet 15/16 e Haiku 13/16; sob a instrução de "contradição", o desempenho desaba para 3/16, 0/16 e 7/16, respectivamente.

## Estrutura do pacote

- `1_artigo/` manuscrito (PDF) e fonte LaTeX (v4, atual)
- `2_scripts_replicacao/` scripts que geram cada resultado
- `3_dados_brutos/` saídas brutas (JSON) que sustentam os números
- `4_figuras_e_tabelas/` figuras (PDF) e tabelas (TeX) que o artigo compila
- `5_historico_versoes/` v1 a v3 do paper (rastreabilidade da evolução)
- `6_revisoes_adversariais/` parecer adversarial que guiou as revisões

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
python experimento_e5_standalone.py  # e5-large: cosseno bruto da inversão de polo
python experimento_e5_normalizado.py # normalização HEROS + anisotropia
python remedicao_gap_limpo.py        # gap normalizado no subconjunto limpo
```

Compare as saídas com os JSON de referência em `3_dados_brutos/`. O resultado esperado e o mapeamento completo de cada figura e tabela estão em [`LEIA-ME.md`](LEIA-ME.md).

## Como citar

```bibtex
@unpublished{mourao2026cosseno,
  author = {Mour{\~a}o, Pedro Borges},
  title  = {Cosseno 0,99 n{\~a}o {\'e} compreens{\~a}o: anisotropia, polaridade e a compet{\^e}ncia jur{\'i}dica diante da recupera{\c c}{\~a}o densa},
  year   = {2026},
  note   = {Preprint}
}
```

## Licença

O código em `2_scripts_replicacao/` está sob a licença MIT (ver [`LICENSE`](LICENSE)). O texto e as figuras do artigo (`1_artigo/`, `4_figuras_e_tabelas/`, `5_historico_versoes/`) permanecem de autoria de Pedro Borges Mourão e são redistribuídos aqui para fins de verificação independente.
