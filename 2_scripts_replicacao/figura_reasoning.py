# -*- coding: utf-8 -*-
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.size":9,"axes.titlesize":9.5,"axes.labelsize":9,
    "figure.dpi":200,"savefig.dpi":200,"axes.spines.top":False,
    "axes.spines.right":False,"font.family":"DejaVu Sans"})

# NUMEROS REAIS do resultado_multi.json
modelos = ["Haiku 4.5\n(mais barato)", "Sonnet 4.6", "Opus 4.8\n(mais capaz)"]
contr = [7, 0, 3]      # contradicao (do JSON)
papel = [13, 15, 16]   # papel tematico (do JSON)
CROSS = 9

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2))

x = np.arange(len(modelos)); w = 0.36
b1 = ax1.bar(x - w/2, contr, w, label='pergunta "contradição?"', color="#c0392b")
b2 = ax1.bar(x + w/2, papel, w, label='pergunta "qual parte tem o papel?"', color="#27ae60")
ax1.axhline(CROSS, color="#2c3e50", ls="--", lw=1.2)
ax1.text(2.42, CROSS+0.25, f"cross-encoder = {CROSS}", fontsize=6.6, color="#2c3e50", ha="right")
ax1.set_xticks(x); ax1.set_xticklabels(modelos, fontsize=7.5)
ax1.set_ylabel("inversão de polo detectada (de 16)"); ax1.set_ylim(0, 17)
for b in list(b1)+list(b2):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{int(b.get_height())}",
             ha="center", fontsize=8, fontweight="bold")
ax1.legend(fontsize=6.6, loc="upper left")
ax1.set_title("A formulação da pergunta domina", fontsize=9)

# painel 2: recuperacao dos 7 reciprocos que o cross-encoder errou
rec = [5, 7, 7]  # haiku, sonnet, opus — dos 7 reciprocos
ax2.bar(range(3), rec, color=["#7f8c8d","#27ae60","#27ae60"])
ax2.axhline(0, color="#c0392b", lw=1.2, ls="--")
ax2.text(2.4, 0.15, "cross-encoder = 0/7", fontsize=6.6, color="#c0392b", ha="right")
ax2.set_xticks(range(3)); ax2.set_xticklabels(["Haiku","Sonnet","Opus"], fontsize=8)
ax2.set_ylabel("pares recíprocos recuperados (de 7)"); ax2.set_ylim(0, 7.6)
for i,v in enumerate(rec):
    ax2.text(i, v+0.1, f"{v}/7", ha="center", fontsize=8.5, fontweight="bold")
ax2.set_title("Recupera os recíprocos que\no cross-encoder errou (0/7)", fontsize=9)

fig.tight_layout()
fig.savefig("v4_reasoning.pdf", bbox_inches="tight")
fig.savefig("v4_reasoning.png", bbox_inches="tight")
print("ok - figura com numeros reais do JSON")
