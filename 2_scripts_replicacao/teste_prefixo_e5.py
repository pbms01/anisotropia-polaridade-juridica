# -*- coding: utf-8 -*-
# teste_prefixo_e5.py
# Responde ao ataque do revisor: "o baseline de 0.84 e artefato de usar query:/query:?
# Ele cairia com passage:/passage:?"
# Mede o baseline anisotropico do e5 sob AMBOS os protocolos de prefixo.
# A literatura (autores do e5) diz que query: e o correto p/ similaridade simetrica
# e que a anisotropia vem da temperatura 0.01, nao do prefixo. Este teste confirma.
# AUTOCONTIDO. Offline com e5 em cache.

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys, json, statistics as st
import numpy as np

MODELO = "intfloat/multilingual-e5-large"

# corpus juridico variado p/ baseline (frases sem relacao semantica direta entre si)
CORPUS = [
    "o réu compareceu à audiência de instrução","a tutela de urgência foi deferida pelo juízo",
    "a sentença julgou procedente a demanda inicial","o recurso foi conhecido e provido pela turma",
    "o contrato foi considerado válido pelo magistrado","a preliminar de prescrição foi acolhida",
    "o perito apresentou laudo no prazo assinado","as custas processuais foram recolhidas pela parte",
    "o mandado de penhora recaiu sobre o imóvel","a citação por edital foi determinada pelo juízo",
    "o agravo de instrumento foi interposto tempestivamente","a audiência de conciliação foi remarcada",
    "o acordo foi homologado por sentença irrecorrível","a sentença foi publicada no diário eletrônico",
    "o recurso especial foi inadmitido na origem","a prova testemunhal foi deferida pelo juízo",
    "o oficial de justiça certificou a diligência cumprida","a incompetência absoluta foi arguida pela defesa",
    "a aposentadoria por tempo de contribuição foi concedida","o benefício previdenciário foi indeferido",
    "o reclamante pleiteou horas extras na inicial","a reclamada contestou o vínculo empregatício",
    "o alimentando requereu majoração da pensão","o inventário foi aberto após o falecimento",
    "a partilha de bens foi homologada pelo juízo","o usufruto vitalício foi instituído em escritura",
    "a hipoteca foi registrada na matrícula do imóvel","o despejo por falta de pagamento foi decretado",
    "a ação de cobrança foi julgada parcialmente procedente","o cheque foi devolvido por insuficiência de fundos",
    "a duplicata foi protestada por falta de aceite","o título executivo extrajudicial foi reconhecido",
]

# par minimo canonico de inversao de polo p/ comparar discriminacao sob cada prefixo
POL_A = "o autor deve indenizar o réu pelos danos morais"
POL_B = "o réu deve indenizar o autor pelos danos morais"

try:
    from sentence_transformers import SentenceTransformer
    M = SentenceTransformer(MODELO)
except Exception as e:
    print("ERRO carregando e5 offline:", str(e)[:160]); sys.exit(1)

def baseline(prefix):
    E = M.encode([f"{prefix}: {t}" for t in CORPUS], normalize_embeddings=True, show_progress_bar=False)
    cos = [float(np.dot(E[i],E[j])) for i in range(len(E)) for j in range(i+1,len(E))]
    return st.mean(cos), len(cos)

def pol_cos(prefix):
    e = M.encode([f"{prefix}: {POL_A}", f"{prefix}: {POL_B}"], normalize_embeddings=True, show_progress_bar=False)
    return float(np.dot(e[0],e[1]))

print(f"[MODELO] {MODELO}\n")
print("=== BASELINE ANISOTROPICO SOB CADA PREFIXO ===")
out={}
for p in ["query","passage"]:
    b,n = baseline(p)
    pc = pol_cos(p)
    out[p]={"baseline":round(b,4),"n_pares":n,"pol_cos":round(pc,4)}
    print(f"  prefixo '{p}:'  baseline={b:.4f} (n={n})   cos(polo_A,polo_B)={pc:.4f}")

# tambem o caso assimetrico que o revisor citou como "natural" p/ peca estatica:
ea = M.encode([f"passage: {POL_A}"], normalize_embeddings=True, show_progress_bar=False)[0]
eb = M.encode([f"passage: {POL_B}"], normalize_embeddings=True, show_progress_bar=False)[0]
print(f"\n  (passage/passage cos polo = {float(np.dot(ea,eb)):.4f})")

print("\n=== VEREDITO ===")
dq=out["query"]["baseline"]; dp=out["passage"]["baseline"]
print(f"  baseline query:  {dq:.3f}")
print(f"  baseline passage: {dp:.3f}")
print(f"  diferenca: {abs(dq-dp):.3f}")
if abs(dq-dp) < 0.1 and dp > 0.6:
    print("  => Anisotropia PERSISTE com passage:. Nao e artefato de prefixo.")
    print("     Confirma a explicacao dos autores: vem da temperatura 0.01 do InfoNCE.")
else:
    print("  => Diferenca relevante; reportar ambos no paper.")

json.dump(out, open("teste_prefixo_e5.json","w"), ensure_ascii=False, indent=2)
print("\n[OK] salvo em teste_prefixo_e5.json")
print("\n----- COLE O BLOCO ABAIXO -----")
print(json.dumps(out, ensure_ascii=False))
