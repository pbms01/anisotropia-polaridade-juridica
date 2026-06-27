# -*- coding: utf-8 -*-
"""Experimento v2: cosseno + bootstrap IC95 por estrato e do gap pareado."""
import json, statistics as st, random
import numpy as np
from sentence_transformers import SentenceTransformer
from dataset_v2 import PARES
random.seed(42)

MODELOS=[('MiniLM','sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
         ('mpnet','sentence-transformers/paraphrase-multilingual-mpnet-base-v2'),
         ('distiluse','sentence-transformers/distiluse-base-multilingual-cased-v2')]

def boot_ci(vals,fn=st.mean,n=5000,a=0.025):
    bs=sorted(fn([random.choice(vals) for _ in vals]) for _ in range(n))
    return bs[int(n*a)],bs[int(n*(1-a))]

def boot_gap_ci(contr,ctrl,n=5000,a=0.025):
    bs=sorted(st.mean([random.choice(ctrl) for _ in ctrl])-st.mean([random.choice(contr) for _ in contr]) for _ in range(n))
    return bs[int(n*a)],bs[int(n*(1-a))]

brutos={}
for nome,hub in MODELOS:
    m=SentenceTransformer(hub)
    regs=[]
    for pid,a,b,cat in PARES:
        e=m.encode([a,b],normalize_embeddings=True,show_progress_bar=False)
        regs.append({'par':pid,'categoria':cat,'cos':round(float(np.dot(e[0],e[1])),4)})
    brutos[nome]=regs
    print(f'[OK] {nome}')

json.dump(brutos,open('brutos_v2.json','w'),ensure_ascii=False,indent=2)

ESTRATOS=['negacao_adverbial','antonimia_nucleo','inversao_polo','quant_modal','controle_parafrase']
print('\n=== POR ESTRATO (com IC95 bootstrap) ===')
agg={}
for m in brutos:
    agg[m]={}
    for est in ESTRATOS:
        vals=[r['cos'] for r in brutos[m] if r['categoria']==est]
        lo,hi=boot_ci(vals)
        agg[m][est]={'mu':round(st.mean(vals),3),'sd':round(st.pstdev(vals),3),'lo':round(lo,3),'hi':round(hi,3),'n':len(vals)}
        print(f'{m:10s} {est:18s} mu={st.mean(vals):.3f} IC95=[{lo:.3f},{hi:.3f}] n={len(vals)}')

print('\n=== GAP PAREADO (contrastivo vs controle pareado) com IC95 ===')
gaps={}
for m in brutos:
    contr=[r['cos'] for r in brutos[m] if r['categoria']!='controle_parafrase']
    ctrl=[r['cos'] for r in brutos[m] if r['categoria']=='controle_parafrase']
    g=st.mean(ctrl)-st.mean(contr)
    lo,hi=boot_gap_ci(contr,ctrl)
    gaps[m]={'contr':round(st.mean(contr),3),'ctrl':round(st.mean(ctrl),3),'gap':round(g,3),'lo':round(lo,3),'hi':round(hi,3)}
    sig='exclui 0' if lo>0 else 'inclui 0'
    print(f'{m:10s} contr={st.mean(contr):.3f} ctrl={st.mean(ctrl):.3f} gap={g:+.3f} IC95=[{lo:+.3f},{hi:+.3f}] {sig}')

json.dump({'agg':agg,'gaps':gaps},open('agg_v2.json','w'),ensure_ascii=False,indent=2)
