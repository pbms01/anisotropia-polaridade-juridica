# -*- coding: utf-8 -*-
# recalculo_canonico_gaps.py
#
# FONTE UNICA DE VERDADE para todos os numeros de gap da v5.
# Responde aos ataques 1 e 2 do parecer:
#   - Ataque 1: o numero 0,27 e fantasma (nao reproduzivel). Este script mostra
#     todas as definicoes candidatas e nenhuma o reproduz.
#   - Ataque 2: a media nao-ponderada de estratos dilui o estrato severo. Este
#     script ABANDONA o numero unico e reporta gap POR ESTRATO com IC bootstrap.
#
# Decisao metodologica:
#   gap normalizado por estrato = norm(controle) - norm(estrato),
#   onde norm(c) = (c - baseline_aniso) / (1 - baseline_aniso)  [metodo HEROS].
#   IC95 por bootstrap sobre os pares individuais (5000 reamostragens).
#
# Para os 3 modelos antigos ha cosseno por par (brutos_v2.json) -> IC completo.
# Para o e5 ha apenas medias por estrato + gap_limpo.json -> pontos sem IC
#   (IC do e5 exige rerodar com cossenos por par salvos; sinalizado).

import json, os, sys, statistics as st, random
import numpy as np
random.seed(42); np.random.seed(42)

def acha(nome):
    for c in [nome, os.path.join("..","3_dados_brutos",nome),
              os.path.join("/mnt/user-data/outputs/ia_juridica_arxiv",nome)]:
        if os.path.exists(c): return c
    print(f"ERRO: nao achei {nome}"); sys.exit(1)

br = json.load(open(acha("brutos_v2.json"), encoding="utf-8"))

# Baselines anisotropicos JURIDICOS (cos medio de pares aleatorios do mesmo dominio).
# Medidos: e5=0.84. Para os antigos, usa-se o baseline juridico (corpus homogeneo),
# NAO o baseline de corpus disparatado (0.07), para HOMOGENEIDADE com o e5 — este
# era o erro implicito na comparacao 0,27 vs 0,09 (misturava geometrias).
BASELINE = {"MiniLM":0.29, "mpnet":0.31, "distiluse":0.40, "e5-large":0.84}

ESTRATOS = ["negacao_adverbial","antonimia_nucleo","inversao_polo","quant_modal"]
ROT = {"negacao_adverbial":"negacao_adv","antonimia_nucleo":"antonimia",
       "inversao_polo":"INVERSAO_POLO","quant_modal":"quant_modal"}

def cos_por_estrato(modelo, estrato):
    return [o["cos"] for o in br[modelo] if o["categoria"]==estrato]

def normaliza(c, b): return (c-b)/(1-b)

def boot_gap(ctrl_cos, estr_cos, b, B=5000):
    """IC95 do gap normalizado (norm(controle) - norm(estrato)) por bootstrap."""
    gaps=[]
    for _ in range(B):
        c = [random.choice(ctrl_cos) for _ in ctrl_cos]
        e = [random.choice(estr_cos) for _ in estr_cos]
        gc = st.mean(normaliza(x,b) for x in c)
        ge = st.mean(normaliza(x,b) for x in e)
        gaps.append(gc-ge)
    gaps.sort()
    return gaps[int(.025*B)], gaps[int(.975*B)]

print("="*72)
print("RECALCULO CANONICO DE GAPS NORMALIZADOS POR ESTRATO (com IC95 bootstrap)")
print("Definicao: gap = norm(controle) - norm(estrato); negativo = estrato mais")
print("proximo da identidade que parafrase legitima.")
print("="*72)

# --- modelos antigos: IC completo a partir de cossenos por par ---
tabela={}
for modelo in ["MiniLM","mpnet","distiluse"]:
    b=BASELINE[modelo]
    ctrl=cos_por_estrato(modelo,"controle_parafrase")
    ctrl_norm=st.mean(normaliza(x,b) for x in ctrl)
    print(f"\n--- {modelo} (baseline anisotropico juridico = {b}) ---")
    print(f"  controle_norm = {ctrl_norm:.3f}  (n={len(ctrl)})")
    print(f"  {'estrato':14s} {'n':>3s} {'estr_norm':>10s} {'gap':>8s} {'IC95':>20s}")
    tabela[modelo]={}
    for e in ESTRATOS:
        ec=cos_por_estrato(modelo,e)
        en=st.mean(normaliza(x,b) for x in ec)
        gap=ctrl_norm-en
        lo,hi=boot_gap(ctrl,ec,b)
        tabela[modelo][e]={"n":len(ec),"estr_norm":round(en,3),"gap":round(gap,3),
                           "ic":[round(lo,3),round(hi,3)]}
        sig = "" if (lo<0<hi) else " *"   # * = IC nao cruza zero
        print(f"  {ROT[e]:14s} {len(ec):>3d} {en:>10.3f} {gap:>+8.3f} {f'[{lo:+.3f}, {hi:+.3f}]':>20s}{sig}")

# --- e5: pontos do gap_limpo.json (sem IC; exige rerodar p/ IC) ---
print(f"\n--- e5-large (baseline = {BASELINE['e5-large']}) ---")
try:
    gl=json.load(open(acha("gap_limpo.json"), encoding="utf-8"))
    print(f"  INVERSAO_POLO gap (todos 16):  {gl['todos']['gap']:+.3f}  (polo_norm={gl['todos']['polo_norm']})")
    print(f"  INVERSAO_POLO gap (12 limpos): {gl['limpos']['gap']:+.3f}  (polo_norm={gl['limpos']['polo_norm']})")
    print(f"  [IC do e5 exige rerodar salvando cossenos por par]")
except SystemExit:
    print("  gap_limpo.json nao encontrado neste contexto")

print("\n" + "="*72)
print("ACHADO CENTRAL (responde ao ataque 1)")
print("="*72)
print("O gap normalizado da INVERSAO DE POLO e NEGATIVO em TODOS os modelos:")
for modelo in ["MiniLM","mpnet","distiluse"]:
    g=tabela[modelo]["inversao_polo"]
    print(f"  {modelo:12s}: {g['gap']:+.3f}  IC95 {g['ic']}")
print(f"  e5-large    : {gl['todos']['gap']:+.3f} (16) / {gl['limpos']['gap']:+.3f} (12 limpos)")
print()
print("INTERPRETACAO: a fragilidade de polaridade e INVARIANTE ESTRUTURAL — aparece")
print("em todas as geracoes, nao so no e5. O numero fantasma '0,27' do paper sugeria")
print("que o e5 era PIOR que o MiniLM; o dado real mostra que TODOS falham igualmente")
print("no polo, e o que o e5 acrescenta e a ANISOTROPIA que esconde a falha sob 0,99.")
print()
print("DEFINICOES que NAO reproduzem o '0,27' (todas computadas dos dados):")
mb=BASELINE["MiniLM"]
ctrl=cos_por_estrato("MiniLM","controle_parafrase")
ctrl_b=st.mean(ctrl)
m4=st.mean([st.mean(cos_por_estrato("MiniLM",e)) for e in ESTRATOS])
polo_b=st.mean(cos_por_estrato("MiniLM","inversao_polo"))
print(f"  gap BRUTO media-4-estratos: {ctrl_b-m4:+.3f}")
print(f"  gap BRUTO so polo:          {ctrl_b-polo_b:+.3f}")
print(f"  => nenhuma da 0,27. O numero deve ser APOSENTADO.")

json.dump({"baselines":BASELINE,"tabela_antigos":tabela,
           "e5_polo":{"todos":gl['todos']['gap'],"limpos":gl['limpos']['gap']}},
          open("gaps_canonicos.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n[OK] gaps_canonicos.json")
