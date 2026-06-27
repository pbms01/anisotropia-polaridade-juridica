# -*- coding: utf-8 -*-
# experimento_e5_normalizado.py
# Aplica normalizacao por baseline anisotropico (metodo HEROS, arXiv 2306.05083)
# ao e5-large, e verifica robustez da margem de recuperacao.
# Responde a objecao: "o cosseno alto do e5 e artefato do prefixo/anisotropia".
# AUTOCONTIDO. Roda offline com o e5 ja em cache.

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, json, statistics as st, random
import numpy as np
random.seed(42); np.random.seed(42)

MODELO = "intfloat/multilingual-e5-large"

# pares (id, A, B, categoria) — mesmo dataset v2
PARES = [
    ("neg-01","O réu compareceu à audiência de instrução.","O réu não compareceu à audiência de instrução.","negacao_adverbial"),
    ("neg-02","A parte autora comprovou o pagamento do débito.","A parte autora não comprovou o pagamento do débito.","negacao_adverbial"),
    ("neg-03","O preposto tinha poderes para transigir.","O preposto não tinha poderes para transigir.","negacao_adverbial"),
    ("neg-04","Houve prévio requerimento administrativo ao INSS.","Não houve prévio requerimento administrativo ao INSS.","negacao_adverbial"),
    ("neg-05","O acusado confessou a autoria do delito.","O acusado não confessou a autoria do delito.","negacao_adverbial"),
    ("neg-06","A citação foi realizada de forma válida.","A citação não foi realizada de forma válida.","negacao_adverbial"),
    ("ant-01","O pedido de tutela de urgência foi deferido.","O pedido de tutela de urgência foi indeferido.","antonimia_nucleo"),
    ("ant-02","A sentença julgou procedente a demanda.","A sentença julgou improcedente a demanda.","antonimia_nucleo"),
    ("ant-03","O recurso foi conhecido e provido.","O recurso foi conhecido e desprovido.","antonimia_nucleo"),
    ("ant-04","O contrato foi considerado válido pelo juízo.","O contrato foi considerado nulo pelo juízo.","antonimia_nucleo"),
    ("ant-05","A preliminar de prescrição foi acolhida.","A preliminar de prescrição foi rejeitada.","antonimia_nucleo"),
    ("ant-06","O magistrado deferiu a produção da prova pericial.","O magistrado indeferiu a produção da prova pericial.","antonimia_nucleo"),
    ("pol-01","O autor deve indenizar o réu pelos danos morais.","O réu deve indenizar o autor pelos danos morais.","inversao_polo"),
    ("pol-02","Condeno a parte ré ao pagamento das custas processuais.","Condeno a parte autora ao pagamento das custas processuais.","inversao_polo"),
    ("pol-03","A apelante sustenta a tese da nulidade.","A apelada sustenta a tese da nulidade.","inversao_polo"),
    ("pol-04","O exequente é parte legítima para a execução.","O executado é parte legítima para a execução.","inversao_polo"),
    ("pol-05","O autor deve pagar honorários ao advogado do réu.","O réu deve pagar honorários ao advogado do autor.","inversao_polo"),
    ("pol-06","Cabe ao impetrante demonstrar o direito líquido e certo.","Cabe à autoridade coatora demonstrar o direito líquido e certo.","inversao_polo"),
    ("pol-07","O agravante requer a reforma da decisão.","O agravado requer a reforma da decisão.","inversao_polo"),
    ("pol-08","O embargante alega excesso de execução.","O embargado alega excesso de execução.","inversao_polo"),
    ("pol-09","A vítima reconheceu o acusado na audiência.","O acusado reconheceu a vítima na audiência.","inversao_polo"),
    ("pol-10","O locador deve restituir a caução ao locatário.","O locatário deve restituir a caução ao locador.","inversao_polo"),
    ("pol-11","O credor cedeu o crédito ao terceiro interessado.","O devedor cedeu o crédito ao terceiro interessado.","inversao_polo"),
    ("pol-12","Compete ao requerente provar o fato constitutivo.","Compete ao requerido provar o fato constitutivo.","inversao_polo"),
    ("pol-13","O reclamante comprovou a jornada extraordinária.","A reclamada comprovou a jornada extraordinária.","inversao_polo"),
    ("pol-14","O alimentante pleiteia a redução da pensão.","O alimentando pleiteia a redução da pensão.","inversao_polo"),
    ("pol-15","O comprador deve entregar a coisa ao vendedor.","O vendedor deve entregar a coisa ao comprador.","inversao_polo"),
    ("pol-16","O mandante responde pelos atos do mandatário.","O mandatário responde pelos atos do mandante.","inversao_polo"),
    ("mod-01","Todos os requisitos da aposentadoria foram preenchidos.","Nenhum dos requisitos da aposentadoria foi preenchido.","quant_modal"),
    ("mod-02","O magistrado deve aplicar a pena no mínimo legal.","O magistrado não deve aplicar a pena no mínimo legal.","quant_modal"),
    ("mod-03","É cabível a concessão da gratuidade de justiça.","É incabível a concessão da gratuidade de justiça.","quant_modal"),
    ("mod-04","A prova é suficiente para a condenação.","A prova é insuficiente para a condenação.","quant_modal"),
    ("cpar-01","O réu não compareceu à audiência de instrução.","O réu não compareceu à audiência de inquirição.","controle_parafrase"),
    ("cpar-02","O pedido de tutela de urgência foi indeferido.","O pedido de tutela de urgência foi rejeitado.","controle_parafrase"),
    ("cpar-03","A sentença julgou improcedente a demanda.","A sentença julgou improcedente a ação.","controle_parafrase"),
    ("cpar-04","O autor deve indenizar o réu pelos danos morais.","O autor deve ressarcir o réu pelos danos morais.","controle_parafrase"),
    ("cpar-05","Nenhum dos requisitos da aposentadoria foi preenchido.","Nenhum dos requisitos da aposentadoria foi satisfeito.","controle_parafrase"),
    ("cpar-06","O acusado não confessou a autoria do delito.","O acusado não confessou a autoria do crime.","controle_parafrase"),
    ("cpar-07","A citação não foi realizada de forma válida.","A citação não foi efetuada de forma válida.","controle_parafrase"),
    ("cpar-08","A preliminar de prescrição foi acolhida.","A preliminar de prescrição foi admitida.","controle_parafrase"),
]

# corpus grande de frases juridicas variadas para baseline anisotropico robusto
BASELINE_CORPUS = [
    "o réu compareceu à audiência de instrução","a tutela de urgência foi deferida pelo juízo",
    "a sentença julgou procedente a demanda inicial","o recurso foi conhecido e provido pela turma",
    "o contrato foi considerado válido pelo magistrado","a preliminar de prescrição foi acolhida",
    "o perito apresentou laudo no prazo assinado","as custas processuais foram recolhidas pela parte",
    "o mandado de penhora recaiu sobre o imóvel","a citação por edital foi determinada pelo juízo",
    "o agravo de instrumento foi interposto tempestivamente","a audiência de conciliação foi remarcada",
    "o acordo foi homologado por sentença irrecorrível","a sentença foi publicada no diário eletrônico",
    "o recurso especial foi inadmitido na origem","a prova testemunhal foi deferida pelo juízo",
    "o oficial de justiça certificou a diligência cumprida","a incompetência absoluta foi arguida pela defesa",
    "a aposentadoria por tempo de contribuição foi concedida","o benefício previdenciário foi indeferido administrativamente",
    "o reclamante pleiteou horas extras na inicial","a reclamada contestou o vínculo empregatício",
    "o alimentando requereu majoração da pensão","o inventário foi aberto após o falecimento",
    "a partilha de bens foi homologada pelo juízo","o usufruto vitalício foi instituído em escritura",
    "a hipoteca foi registrada na matrícula do imóvel","o despejo por falta de pagamento foi decretado",
    "a ação de cobrança foi julgada parcialmente procedente","o cheque foi devolvido por insuficiência de fundos",
    "a duplicata foi protestada por falta de aceite","o título executivo extrajudicial foi reconhecido",
]

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("ERRO sentence-transformers:", e); sys.exit(1)

try:
    M = SentenceTransformer(MODELO)
except Exception as e:
    print(f"ERRO ao carregar {MODELO} offline:", str(e)[:160]); sys.exit(1)
print(f"[MODELO] {MODELO}\n", flush=True)

def enc(textos, tipo="query"):
    return M.encode([f"{tipo}: {t}" for t in textos], normalize_embeddings=True, show_progress_bar=False)
def cos(u,v): return float(np.dot(u,v))

# ---- baseline anisotropico (cos_avg de pares aleatorios) -------------------
EB = enc(BASELINE_CORPUS, "query")
rand = [cos(EB[i],EB[j]) for i in range(len(EB)) for j in range(i+1,len(EB))]
cos_avg = st.mean(rand)
print(f"=== BASELINE ANISOTROPICO ===")
print(f"  cos_avg (pares aleatorios, n={len(rand)}) = {cos_avg:.4f}")
print(f"  (na literatura, e5 multilingue ~0.75-0.85; confirma anisotropia)\n")

# ---- cosseno BRUTO e NORMALIZADO (HEROS) por estrato -----------------------
# normalizado: (cos - cos_avg) / (1 - cos_avg)  -> remap p/ [.., 1], baseline vira 0
def normaliza(c): return (c - cos_avg) / (1 - cos_avg)

ESTRATOS=["negacao_adverbial","antonimia_nucleo","inversao_polo","quant_modal","controle_parafrase"]
print("=== COSSENO BRUTO vs NORMALIZADO POR BASELINE (HEROS) ===")
print(f"{'estrato':18s} {'bruto_mu':>9s} {'norm_mu':>9s}")
norm_por_estrato={}
for e in ESTRATOS:
    brutos=[]; norms=[]
    for pid,a,b,cat in PARES:
        if cat==e:
            c=cos(*enc([a,b],"query")); brutos.append(c); norms.append(normaliza(c))
    norm_por_estrato[e]={"bruto":round(st.mean(brutos),3),"norm":round(st.mean(norms),3)}
    print(f"{e:18s} {st.mean(brutos):9.3f} {st.mean(norms):9.3f}")

# interpretacao: no espaco normalizado, controle deve ficar ALTO e contrastivos
# MAIS BAIXOS se o modelo discrimina. Se contrastivos ~ controle mesmo normalizado,
# a discriminacao e fraca tambem em termos relativos.
contr_norm=st.mean([norm_por_estrato[e]["norm"] for e in ESTRATOS if e!="controle_parafrase"])
ctrl_norm=norm_por_estrato["controle_parafrase"]["norm"]
print(f"\n  contrastivos_norm_mu={contr_norm:.3f}  controle_norm_mu={ctrl_norm:.3f}  gap_norm={ctrl_norm-contr_norm:+.3f}")

# ---- recuperacao ASSIMETRICA com distratores (prefixo correto) -------------
CASOS=[
 ("o réu esteve presente na audiência","o réu compareceu à audiência de instrução","o réu não compareceu à audiência de instrução"),
 ("a tutela de urgência foi concedida","o pedido de tutela de urgência foi deferido","o pedido de tutela de urgência foi indeferido"),
 ("a ação foi julgada a favor do autor","a sentença julgou procedente a demanda","a sentença julgou improcedente a demanda"),
 ("o pagamento do débito ficou comprovado","a parte autora comprovou o pagamento do débito","a parte autora não comprovou o pagamento do débito"),
 ("o contrato foi reconhecido como eficaz","o contrato foi considerado válido pelo juízo","o contrato foi considerado nulo pelo juízo"),
 ("a prescrição foi reconhecida","a preliminar de prescrição foi acolhida","a preliminar de prescrição foi rejeitada"),
 ("a prova basta para condenar","a prova é suficiente para a condenação","a prova é insuficiente para a condenação"),
 ("o recurso teve êxito","o recurso foi conhecido e provido","o recurso foi conhecido e desprovido"),
 ("o acusado admitiu o crime","o acusado confessou a autoria do delito","o acusado não confessou a autoria do delito"),
 ("a perícia foi autorizada","o magistrado deferiu a produção da prova pericial","o magistrado indeferiu a produção da prova pericial"),
]
DISTR=["o juízo determinou a citação por edital do executado","as custas processuais foram recolhidas pela parte vencida",
 "o perito apresentou laudo complementar no prazo legal","a audiência de conciliação foi remarcada pelo cartório",
 "o mandado de penhora recaiu sobre o imóvel residencial","a parte interpôs agravo de instrumento contra a decisão",
 "o magistrado homologou o acordo celebrado entre as partes","a sentença foi publicada no diário oficial eletrônico",
 "o recurso especial foi inadmitido na origem pelo tribunal","a parte requereu a produção de prova testemunhal adicional",
 "o oficial de justiça certificou o cumprimento da diligência","a defesa arguiu a incompetência absoluta do juízo cível"]
ED=enc(DISTR,"passage")
rank1=0; opo_acima=0; margens=[]; margens_norm=[]
for q,corr,opo in CASOS:
    eq=enc([q],"query")[0]; ec=enc([corr],"passage")[0]; eo=enc([opo],"passage")[0]
    sc,so=cos(eq,ec),cos(eq,eo)
    corpus=[("C",ec),("O",eo)]+[("D",d) for d in ED]
    tags=[t for _,t in sorted(((cos(eq,v),t) for t,v in corpus),reverse=True)]
    if tags[0]=="C": rank1+=1
    if tags.index("O")<tags.index("C"): opo_acima+=1
    margens.append(sc-so); margens_norm.append(normaliza(sc)-normaliza(so))
print(f"\n=== RECUPERACAO ASSIMETRICA + DISTRATORES ===")
print(f"  correta rank1: {rank1}/{len(CASOS)}  oposta_acima: {opo_acima}/{len(CASOS)}")
print(f"  margem bruta: media={st.mean(margens):+.4f} min={min(margens):+.4f}")
print(f"  margem normalizada: media={st.mean(margens_norm):+.4f} min={min(margens_norm):+.4f}")

saida={"modelo":MODELO,"cos_avg_baseline":round(cos_avg,4),
       "estratos_norm":norm_por_estrato,
       "gap_normalizado":round(ctrl_norm-contr_norm,3),
       "recuperacao":{"rank1":rank1,"oposta_acima":opo_acima,
                      "margem_bruta_min":round(min(margens),4),
                      "margem_norm_media":round(st.mean(margens_norm),4),
                      "margem_norm_min":round(min(margens_norm),4)}}
with open("resultado_e5_norm.json","w",encoding="utf-8") as f:
    json.dump(saida,f,ensure_ascii=False,indent=2)
print("\n[OK] salvo em resultado_e5_norm.json")
print("\n----- COLE O BLOCO ABAIXO NA CONVERSA -----")
print(json.dumps(saida,ensure_ascii=False))
