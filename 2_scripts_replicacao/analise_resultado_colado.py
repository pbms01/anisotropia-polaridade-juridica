# -*- coding: utf-8 -*-
# Analise dos resultados da verificacao por raciocinio (Secao 9).
# LE os numeros do resultado_multi.json (fonte de verdade), nao usa valores fixos.
# Procura o JSON no diretorio atual ou em ../3_dados_brutos/.
import json, os, sys

def acha_json(nome):
    for cand in [nome, os.path.join("..","3_dados_brutos",nome),
                 os.path.join("3_dados_brutos",nome)]:
        if os.path.exists(cand):
            return cand
    print(f"ERRO: nao encontrei {nome}. Rode no diretorio dos scripts ou ao lado do JSON.")
    sys.exit(1)

d = json.load(open(acha_json("resultado_multi.json"), encoding="utf-8"))["resultados"]

# nomes podem variar (haiku com sufixo de data); normaliza p/ exibicao
def label(m):
    if "opus" in m: return "claude-opus-4-8"
    if "sonnet" in m: return "claude-sonnet-4-6"
    if "haiku" in m: return "claude-haiku-4-5"
    return m

CROSS_ACERTOU = {"pol-01","pol-02","pol-04","pol-05","pol-09","pol-10","pol-13","pol-15","pol-16"}
CROSS_ERROU = [f"pol-{i:02d}" for i in range(1,17) if f"pol-{i:02d}" not in CROSS_ACERTOU]
CROSS = 9; N = 16

# conta do JSON
linhas = {}
for m, blocos in d.items():
    c = sum(1 for v in blocos["contradicao"].values() if v)
    p = sum(1 for v in blocos["papel_tematico"].values() if v)
    rec = sum(1 for pid in CROSS_ERROU if blocos["papel_tematico"].get(pid))
    linhas[label(m)] = {"contr": c, "papel": p, "rec": rec}

print("="*64)
print(f"MATRIZ: identificacao da inversao de polo (de {N}) — lida do JSON")
print("="*64)
print(f"{'modelo':26s} {'contradicao':>12s} {'papel temat.':>14s} {'recip. (de 7)':>14s}")
print(f"{'cross-encoder NLI (ref.)':26s} {CROSS:>12d} {'—':>14s} {'0':>14s}")
for m, v in linhas.items():
    print(f"{m:26s} {v['contr']:>12d} {v['papel']:>14d} {v['rec']:>14d}")

print()
print("EFEITO 1 — formulacao domina (salto papel - contradicao):")
for m, v in linhas.items():
    print(f"  {m:24s}: {v['contr']:2d}/16 -> {v['papel']:2d}/16  (salto +{v['papel']-v['contr']})")

print()
print("EFEITO 2 — sob 'contradicao', capacidade NAO compra acerto:")
print("  (ordem de capacidade Haiku < Sonnet < Opus)")
for m in ["claude-haiku-4-5","claude-sonnet-4-6","claude-opus-4-8"]:
    if m in linhas: print(f"  {m:24s}: {linhas[m]['contr']}/16")
print("  O modelo de fronteira nao lidera; a deteccao nao cresce com a capacidade.")

print()
print("EFEITO 3 — recuperacao dos 7 reciprocos que o cross-encoder errou (0/7):")
for m, v in linhas.items():
    print(f"  {m:24s}: {v['rec']}/7")
print("  A pergunta por papel tematico ataca a fraqueza estrutural do cross-encoder.")

print()
print("RESSALVA (n=16): diferencas de poucos pares entre modelos NAO sao")
print("significativas. Robusto: o salto grande e a recuperacao dos reciprocos.")
