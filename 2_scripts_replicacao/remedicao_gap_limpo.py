# -*- coding: utf-8 -*-
# remedicao_gap_limpo.py
# Responde ao revisor: o gap normalizado do e5 na inversao de polo se sustenta
# quando medido SO sobre os pares juridicamente coerentes?
# Remede o gap normalizado (metodo HEROS) sobre: 16 (todos), 15 (sem impossivel),
# 12 (so limpos). Se o padrao sobrevive, a afirmacao "estrato mais severo" fica solida.
# AUTOCONTIDO. Offline com e5 em cache.

import os
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
import sys, json, statistics as st
import numpy as np

MODELO="intfloat/multilingual-e5-large"

# 16 pares de inversao de polo (A, B)
POL = {
 "pol-01":("O autor deve indenizar o réu pelos danos morais.","O réu deve indenizar o autor pelos danos morais."),
 "pol-02":("Condeno a parte ré ao pagamento das custas processuais.","Condeno a parte autora ao pagamento das custas processuais."),
 "pol-03":("A apelante sustenta a tese da nulidade.","A apelada sustenta a tese da nulidade."),
 "pol-04":("O exequente é parte legítima para a execução.","O executado é parte legítima para a execução."),
 "pol-05":("O autor deve pagar honorários ao advogado do réu.","O réu deve pagar honorários ao advogado do autor."),
 "pol-06":("Cabe ao impetrante demonstrar o direito líquido e certo.","Cabe à autoridade coatora demonstrar o direito líquido e certo."),
 "pol-07":("O agravante requer a reforma da decisão.","O agravado requer a reforma da decisão."),
 "pol-08":("O embargante alega excesso de execução.","O embargado alega excesso de execução."),
 "pol-09":("A vítima reconheceu o acusado na audiência.","O acusado reconheceu a vítima na audiência."),
 "pol-10":("O locador deve restituir a caução ao locatário.","O locatário deve restituir a caução ao locador."),
 "pol-11":("O credor cedeu o crédito ao terceiro interessado.","O devedor cedeu o crédito ao terceiro interessado."),
 "pol-12":("Compete ao requerente provar o fato constitutivo.","Compete ao requerido provar o fato constitutivo."),
 "pol-13":("O reclamante comprovou a jornada extraordinária.","A reclamada comprovou a jornada extraordinária."),
 "pol-14":("O alimentante pleiteia a redução da pensão.","O alimentando pleiteia a redução da pensão."),
 "pol-15":("O comprador deve entregar a coisa ao vendedor.","O vendedor deve entregar a coisa ao comprador."),
 "pol-16":("O mandante responde pelos atos do mandatário.","O mandatário responde pelos atos do mandante."),
}
# controle de parafrase (8) para o gap
CTRL = [
 ("O réu não compareceu à audiência de instrução.","O réu não compareceu à audiência de inquirição."),
 ("O pedido de tutela de urgência foi indeferido.","O pedido de tutela de urgência foi rejeitado."),
 ("A sentença julgou improcedente a demanda.","A sentença julgou improcedente a ação."),
 ("O autor deve indenizar o réu pelos danos morais.","O autor deve ressarcir o réu pelos danos morais."),
 ("Nenhum dos requisitos da aposentadoria foi preenchido.","Nenhum dos requisitos da aposentadoria foi satisfeito."),
 ("O acusado não confessou a autoria do delito.","O acusado não confessou a autoria do crime."),
 ("A citação não foi realizada de forma válida.","A citação não foi efetuada de forma válida."),
 ("A preliminar de prescrição foi acolhida.","A preliminar de prescrição foi admitida."),
]
BASE = [  # corpus p/ baseline anisotropico
 "o réu compareceu à audiência de instrução","a tutela de urgência foi deferida pelo juízo",
 "a sentença julgou procedente a demanda inicial","o recurso foi conhecido e provido pela turma",
 "o contrato foi considerado válido pelo magistrado","a preliminar de prescrição foi acolhida",
 "o perito apresentou laudo no prazo assinado","as custas processuais foram recolhidas pela parte",
 "o mandado de penhora recaiu sobre o imóvel","a citação por edital foi determinada pelo juízo",
 "o agravo de instrumento foi interposto tempestivamente","a audiência de conciliação foi remarcada",
 "o acordo foi homologado por sentença irrecorrível","a sentença foi publicada no diário eletrônico",
 "o recurso especial foi inadmitido na origem","a prova testemunhal foi deferida pelo juízo",
]

# subconjuntos da auditoria
LIMPOS = ["pol-01","pol-02","pol-03","pol-05","pol-07","pol-08","pol-09","pol-10","pol-12","pol-13","pol-14","pol-15"]
SEM_IMP = LIMPOS + ["pol-04","pol-06","pol-16"]
TODOS = list(POL.keys())

try:
    from sentence_transformers import SentenceTransformer
    M=SentenceTransformer(MODELO)
except Exception as e:
    print("ERRO e5 offline:",str(e)[:140]); sys.exit(1)

def enc(ts): return M.encode([f"query: {t}" for t in ts], normalize_embeddings=True, show_progress_bar=False)
def cos(u,v): return float(np.dot(u,v))
def normaliza(c, base): return (c-base)/(1-base)

EB=enc(BASE)
rand=[cos(EB[i],EB[j]) for i in range(len(EB)) for j in range(i+1,len(EB))]
base=st.mean(rand)
print(f"baseline anisotropico = {base:.4f}\n")

# cosseno normalizado do controle (fixo)
ctrl_norm=st.mean([normaliza(cos(*enc([a,b])),base) for a,b in CTRL])

print("=== GAP NORMALIZADO (controle - inversao_polo) por subconjunto ===")
print(f"controle_norm = {ctrl_norm:.3f} (fixo)\n")
print(f"{'subconjunto':20s} {'n':>3s} {'polo_norm':>10s} {'gap':>8s}")
out={}
for nome,sub in [("todos",TODOS),("sem_impossivel",SEM_IMP),("limpos",LIMPOS)]:
    polo_norm=st.mean([normaliza(cos(*enc(list(POL[p]))),base) for p in sub])
    gap=ctrl_norm-polo_norm
    out[nome]={"n":len(sub),"polo_norm":round(polo_norm,3),"gap":round(gap,3)}
    print(f"{nome:20s} {len(sub):>3d} {polo_norm:>10.3f} {gap:>8.3f}")

print()
print("VEREDITO: se o gap permanece baixo (~0,09) e a ordem de grandeza nao muda")
print("entre os 16 e os 12 limpos, a afirmacao 'estrato mais severo' sobrevive a")
print("auditoria e fica MAIS forte (medida sobre conjunto limpo).")
json.dump({"baseline":round(base,4),"controle_norm":round(ctrl_norm,3),"subconjuntos":out},
          open("gap_limpo.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("\n[OK] gap_limpo.json")
print("\n----- COLE O BLOCO -----")
print(json.dumps(out,ensure_ascii=False))
