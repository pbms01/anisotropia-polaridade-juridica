# -*- coding: utf-8 -*-
import json, statistics as st
import os as _os
def acha_json(_n):
    for _c in [_n, _os.path.join("..","3_dados_brutos",_n), _os.path.join("3_dados_brutos",_n)]:
        if _os.path.exists(_c): return _c
    return _n  # deixa o open falhar com mensagem padrao se nao achar
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"axes.titlesize":9.5,"axes.labelsize":9,
    "figure.dpi":200,"savefig.dpi":200,"axes.spines.top":False,
    "axes.spines.right":False,"font.family":"DejaVu Sans"})

con=json.load(open(acha_json("consolidado_v3.json")))
mat=con["matriz_bruta"]; e5=con["e5"]
ESTRATOS=["negacao_adverbial","antonimia_nucleo","inversao_polo","quant_modal","controle_parafrase"]
ROT={"negacao_adverbial":"Negação\nadverbial","antonimia_nucleo":"Antonímia\nde núcleo",
     "inversao_polo":"Inversão\nde polo","quant_modal":"Quantif./\nmodal",
     "controle_parafrase":"Controle\npareado"}
MODS=["MiniLM","mpnet","distiluse","e5-large"]

# FIG A: heatmap dos 4 modelos (bruto) + linha de baseline anisotropico
M=np.array([[mat[e][m] for e in ESTRATOS] for m in MODS])
fig,ax=plt.subplots(figsize=(6.6,3.0))
im=ax.imshow(M,cmap="RdYlGn_r",vmin=0.4,vmax=1.0,aspect="auto")
ax.set_xticks(range(len(ESTRATOS)));ax.set_xticklabels([ROT[e] for e in ESTRATOS])
ax.set_yticks(range(len(MODS)));ax.set_yticklabels(MODS)
for i in range(len(MODS)):
    for j in range(len(ESTRATOS)):
        ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="black",fontsize=8.5,fontweight="bold")
ax.add_patch(plt.Rectangle((len(ESTRATOS)-1-0.5,-0.5),1,len(MODS),fill=False,edgecolor="navy",lw=2,ls="--"))
# anota baseline aniso a direita
baselines=["0.07","~0.10","~0.14","0.84"]
for i,b in enumerate(baselines):
    ax.text(len(ESTRATOS)-0.35,i,f"  baseline={b}",va="center",fontsize=6.5,color="#444")
cb=fig.colorbar(im,ax=ax,fraction=0.025,pad=0.13);cb.set_label("cosseno bruto",fontsize=8)
ax.set_title("Cosseno bruto: o e5-large esmaga tudo perto de 0,99 (espaço anisotrópico, baseline 0,84)")
fig.tight_layout();fig.savefig("v3_figA.pdf",bbox_inches="tight");fig.savefig("v3_figA.png",bbox_inches="tight");print("A")

# FIG B (CENTRAL): e5 bruto vs normalizado lado a lado
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.2,3.2),sharey=False)
x=np.arange(len(ESTRATOS))
cores=["#2980b9" if e!="controle_parafrase" else "#27ae60" for e in ESTRATOS]
# bruto
bruto=[e5["bruto"][e] for e in ESTRATOS]
ax1.bar(x,bruto,color=cores); ax1.set_ylim(0,1.05)
ax1.axhline(e5["cos_avg"],color="red",ls="--",lw=1)
ax1.text(0.1,e5["cos_avg"]+0.01,f"baseline aniso={e5['cos_avg']:.2f}",fontsize=6.5,color="red")
ax1.set_xticks(x);ax1.set_xticklabels([ROT[e] for e in ESTRATOS],fontsize=7)
ax1.set_ylabel("cosseno BRUTO");ax1.set_title("Bruto: indistinguível a olho nu\n(tudo entre 0,96 e 0,99)",fontsize=8.5)
for i,v in enumerate(bruto): ax1.text(i,v+0.005,f"{v:.2f}",ha="center",fontsize=6.5)
# normalizado
norm=[e5["norm"][e] for e in ESTRATOS]
ax2.bar(x,norm,color=cores); ax2.set_ylim(0,1.05)
ax2.set_xticks(x);ax2.set_xticklabels([ROT[e] for e in ESTRATOS],fontsize=7)
ax2.set_ylabel("cosseno NORMALIZADO (HEROS)")
ax2.set_title("Normalizado: a estrutura reaparece\npolo (azul) cola no controle (verde)",fontsize=8.5)
for i,v in enumerate(norm): ax2.text(i,v+0.005,f"{v:.2f}",ha="center",fontsize=6.5)
fig.suptitle("e5-large: remover a anisotropia revela a mesma fragilidade dos modelos antigos",
             fontsize=10,y=1.02)
fig.tight_layout();fig.savefig("v3_figB.pdf",bbox_inches="tight");fig.savefig("v3_figB.png",bbox_inches="tight");print("B")

# FIG C: o paradoxo — e5 acerta mais recuperacao, com margem menor
fig,ax=plt.subplots(figsize=(6.0,3.0))
mods=["MiniLM","mpnet","distiluse","e5-large"]
rank1=[8,8,8,10]   # com distratores (MiniLM/mpnet/distiluse medidos antes=8; e5=10)
margem_min=[0.034,-0.103,-0.000,0.0046]  # margem minima de recuperacao
ax2=ax.twinx()
b=ax.bar(np.arange(len(mods))-0.2,rank1,0.4,color="#2980b9",label="acerto rank1 (de 10)")
l=ax2.plot(np.arange(len(mods))+0.2,margem_min,"o-",color="#c0392b",label="margem mínima")
ax.set_xticks(range(len(mods)));ax.set_xticklabels(mods,fontsize=8)
ax.set_ylabel("acerto de ranking (de 10)",color="#2980b9");ax.set_ylim(0,11)
ax2.set_ylabel("margem mínima de cosseno",color="#c0392b")
ax2.axhline(0,color="gray",ls=":",lw=0.8)
for i,v in enumerate(rank1): ax.text(i-0.2,v+0.2,str(v),ha="center",fontsize=8,color="#2980b9")
for i,v in enumerate(margem_min): ax2.text(i+0.2,v+0.006,f"{v:.3f}",ha="center",fontsize=6.5,color="#c0392b")
ax.set_title("O paradoxo do e5: acerta MAIS a recuperação (10/10),\npor margem MENOR (0,005) — a fragilidade fica disfarçada")
fig.tight_layout();fig.savefig("v3_figC.pdf",bbox_inches="tight");fig.savefig("v3_figC.png",bbox_inches="tight");print("C")

# tabela LaTeX consolidada dos 4 modelos (bruto) + coluna e5 normalizado
with open("v3_tabela.tex","w") as f:
    f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
    f.write("Estrato & MiniLM & mpnet & distiluse & e5 (bruto) & e5 (norm.) \\\\\n\\midrule\n")
    for e in ESTRATOS:
        f.write(f"{e.replace('_','\\_')} & {mat[e]['MiniLM']:.3f} & {mat[e]['mpnet']:.3f} & "
                f"{mat[e]['distiluse']:.3f} & {e5['bruto'][e]:.3f} & {e5['norm'][e]:.3f} \\\\\n")
    f.write("\\midrule\n")
    f.write(f"\\textit{{baseline aniso.}} & 0.07 & $\\sim$0.10 & $\\sim$0.14 & \\multicolumn{{2}}{{c}}{{0.84}} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("tab")
