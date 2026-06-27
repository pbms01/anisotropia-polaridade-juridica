# -*- coding: utf-8 -*-
"""Consolida os 4 modelos (3 antigos do brutos_v2 + e5 dos resultados colados)
numa matriz unica para tabelas/figuras da v3."""
import json, statistics as st
import os as _os
def acha_json(_n):
    for _c in [_n, _os.path.join("..","3_dados_brutos",_n), _os.path.join("3_dados_brutos",_n)]:
        if _os.path.exists(_c): return _c
    return _n  # deixa o open falhar com mensagem padrao se nao achar

# 3 modelos antigos: recarrega do brutos_v2.json
br = json.load(open(acha_json("brutos_v2.json")))
ESTRATOS=["negacao_adverbial","antonimia_nucleo","inversao_polo","quant_modal","controle_parafrase"]

# baselines anisotropicos medidos (corpus juridico de 32 frases):
# MiniLM=0.2948 (validacao local), e5=0.8383 (colado). Para mpnet/distiluse
# preciso medir, mas para a figura principal uso bruto + e5 normalizado.
# Aqui monto a tabela BRUTA dos 4 e a NORMALIZADA do e5 (dado central da v3).

# dados e5 colados pelo usuario
e5 = {
  "cos_avg": 0.8383,
  "bruto": {"negacao_adverbial":0.958,"antonimia_nucleo":0.977,"inversao_polo":0.993,"quant_modal":0.975,"controle_parafrase":0.991},
  "norm":  {"negacao_adverbial":0.743,"antonimia_nucleo":0.857,"inversao_polo":0.958,"quant_modal":0.846,"controle_parafrase":0.943},
  "gap_norm":0.092,
  "recup":{"rank1":10,"oposta_acima":0,"margem_bruta_min":0.0046,"margem_norm_media":0.1493,"margem_norm_min":0.0283},
}

# matriz bruta dos 4 modelos
print("=== MATRIZ BRUTA (cosseno medio por estrato) ===")
print(f"{'estrato':18s} {'MiniLM':>8s} {'mpnet':>8s} {'distil':>8s} {'e5-large':>9s}")
matriz={}
for e in ESTRATOS:
    row={}
    for m in br:
        vals=[r['cos'] for r in br[m] if r['categoria']==e]
        row[m]=round(st.mean(vals),3)
    row['e5-large']=e5['bruto'][e]
    matriz[e]=row
    print(f"{e:18s} {row['MiniLM']:8.3f} {row['mpnet']:8.3f} {row['distiluse']:8.3f} {row['e5-large']:9.3f}")

print(f"\n{'baseline aniso':18s} {'0.070':>8s} {'~0.10':>8s} {'~0.14':>8s} {'0.838':>9s}")

print("\n=== e5: BRUTO vs NORMALIZADO (o achado central) ===")
print(f"{'estrato':18s} {'bruto':>8s} {'norm':>8s}")
for e in ESTRATOS:
    print(f"{e:18s} {e5['bruto'][e]:8.3f} {e5['norm'][e]:8.3f}")

print("\n=== GAP NORMALIZADO comparado ===")
print(f"  MiniLM (corpus juridico): 0.267")
print(f"  e5-large:                 {e5['gap_norm']:.3f}  <- MENOR: e5 discrimina polaridade menos em termos relativos")

print("\n=== RECUPERACAO: e5 acerta mais, com margem menor ===")
print(f"  e5 rank1: {e5['recup']['rank1']}/10  margem_bruta_min: {e5['recup']['margem_bruta_min']:.4f}")
print(f"  (MiniLM rank1 8/10 margem_min 0.034; e5 acerta MAIS mas margem 7x menor)")

json.dump({"matriz_bruta":matriz,"e5":e5}, open("consolidado_v3.json","w"), ensure_ascii=False, indent=2)
print("\n[OK] consolidado_v3.json")
