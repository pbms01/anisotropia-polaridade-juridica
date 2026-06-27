# -*- coding: utf-8 -*-
# Regenera v3_figC com dados REAIS de recup_v2.json (corrige o "8/10" fantasma)
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.size":9,"axes.titlesize":9.5,"axes.labelsize":9,
    "figure.dpi":200,"savefig.dpi":200,"axes.spines.top":False,
    "axes.spines.right":False,"font.family":"DejaVu Sans"})

rec=json.load(open("/mnt/user-data/outputs/ia_juridica_arxiv/recup_v2.json"))
# dados reais: taxa e margem_min por modelo. e5 vem do experimento_e5 standalone (10 casos, margem 0.005)
mods=["MiniLM","mpnet","distiluse","e5-large"]
acertos=[rec["MiniLM"]["acertos"],rec["mpnet"]["acertos"],rec["distiluse"]["acertos"],10]
totais=[rec["MiniLM"]["total"],rec["mpnet"]["total"],rec["distiluse"]["total"],10]
margem_min=[rec["MiniLM"]["margem_min"],rec["mpnet"]["margem_min"],rec["distiluse"]["margem_min"],0.005]

fig,ax=plt.subplots(figsize=(6.2,3.1))
ax2=ax.twinx()
x=np.arange(len(mods))
# barras de taxa (proporcao, nao placar absoluto)
taxas=[a/t for a,t in zip(acertos,totais)]
b=ax.bar(x-0.18,taxas,0.36,color="#2980b9",label="taxa de acerto de ranking")
l=ax2.plot(x+0.18,margem_min,"o-",color="#c0392b",lw=2,markersize=8,label="margem mínima")
ax.set_xticks(x);ax.set_xticklabels([f"{m}\n({a}/{t})" for m,a,t in zip(mods,acertos,totais)],fontsize=7.5)
ax.set_ylabel("taxa de acerto de ranking",color="#2980b9");ax.set_ylim(0,1.08)
ax2.set_ylabel("margem mínima de cosseno",color="#c0392b")
ax2.axhline(0,color="gray",ls=":",lw=0.8)
for i,v in enumerate(margem_min):
    ax2.text(i+0.18,v+0.004,f"{v:.3f}",ha="center",fontsize=6.8,color="#c0392b")
ax.set_title("A margem que sustenta a recuperação encolhe à indistinção\n(as falhas concentram-se em pares de inversão de polo)",fontsize=9)
fig.tight_layout()
fig.savefig("v3_figC.pdf",bbox_inches="tight");fig.savefig("v3_figC.png",bbox_inches="tight")
print("v3_figC regenerada com dados reais:",dict(zip(mods,[f'{a}/{t}' for a,t in zip(acertos,totais)])))
