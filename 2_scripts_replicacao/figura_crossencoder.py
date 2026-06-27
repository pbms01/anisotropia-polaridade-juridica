# -*- coding: utf-8 -*-
import json
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

tab=json.load(open(acha_json("crossencoder_tabela.json")))
ordem=["negacao_adverbial","antonimia_nucleo","inversao_polo","quant_modal","controle_parafrase"]
ROT={"negacao_adverbial":"Negação\nadverbial","antonimia_nucleo":"Antonímia\nde núcleo",
     "inversao_polo":"Inversão\nde polo","quant_modal":"Quantif./\nmodal",
     "controle_parafrase":"Controle\npareado"}

# barras empilhadas de classificacao NLI por estrato (proporcao)
fig,ax=plt.subplots(figsize=(6.4,3.1))
x=np.arange(len(ordem))
contr=[tab[c]["contr"] for c in ordem]
ent=[tab[c]["entail"] for c in ordem]
neu=[tab[c]["neutro"] for c in ordem]
ns=[tab[c]["n"] for c in ordem]
# proporcoes
pc=[c/n for c,n in zip(contr,ns)]
pe=[e/n for e,n in zip(ent,ns)]
pn=[nn/n for nn,n in zip(neu,ns)]
b1=ax.bar(x,pc,color="#c0392b",label="contradição (correto p/ inversão)")
b2=ax.bar(x,pe,bottom=pc,color="#e67e22",label="implicação (ERRO: A→B)")
b3=ax.bar(x,pn,bottom=[a+b for a,b in zip(pc,pe)],color="#bbb",label="neutro")
ax.set_xticks(x);ax.set_xticklabels([ROT[c] for c in ordem],fontsize=7.5)
ax.set_ylabel("proporção de classificações");ax.set_ylim(0,1.05)
# anota contagem de contradicao
for i,(c,n) in enumerate(zip(contr,ns)):
    ax.text(i,pc[i]/2 if pc[i]>0.1 else pc[i]+0.03,f"{c}/{n}",ha="center",va="center",
            fontsize=8,color="white" if pc[i]>0.2 else "black",fontweight="bold")
ax.legend(fontsize=7,loc="lower center",ncol=1,framealpha=0.9)
ax.set_title("Cross-encoder NLI (nli-deberta-v3-base): detecção de contradição por estrato\n"
             "a inversão de polo é a única onde a atenção cruzada falha (9/16); 6 viram 'implicação'")
fig.tight_layout();fig.savefig("v4_crossenc.pdf",bbox_inches="tight");fig.savefig("v4_crossenc.png",bbox_inches="tight")
print("ok")

# tabela latex de contingencia
with open("v4_crossenc_tabela.tex","w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
    f.write("Estrato & Contradição & Implicação & Neutro & $n$ \\\\\n\\midrule\n")
    for c in ordem:
        t=tab[c]
        f.write(f"{c.replace('_','\\_')} & {t['contr']} & {t['entail']} & {t['neutro']} & {t['n']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("tab")
