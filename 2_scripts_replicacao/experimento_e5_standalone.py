# -*- coding: utf-8 -*-
# experimento_e5_standalone.py
# Mede um embedder moderno (e5 / BGE) no dataset v2 de pares minimos forenses,
# para comparacao direta com MiniLM/mpnet/distiluse. Fecha o ataque de "baseline antigo".
# AUTOCONTIDO: os 40 pares estao embutidos abaixo, nao depende de dataset_v2.py.
# Roda OFFLINE por padrao: usa apenas modelos ja presentes no cache local.

import os
os.environ["HF_HUB_OFFLINE"] = "1"          # nao tenta baixar nada
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, json, statistics as st, random
import numpy as np

random.seed(42)
np.random.seed(42)

# ===========================================================================
# DATASET v2 — 40 pares minimos forenses (pt-BR), embutidos.
# (id, frase_A, frase_B, categoria)
# ===========================================================================
PARES = [
    # --- NEGAÇÃO ADVERBIAL (6) ---
    ("neg-01", "O réu compareceu à audiência de instrução.", "O réu não compareceu à audiência de instrução.", "negacao_adverbial"),
    ("neg-02", "A parte autora comprovou o pagamento do débito.", "A parte autora não comprovou o pagamento do débito.", "negacao_adverbial"),
    ("neg-03", "O preposto tinha poderes para transigir.", "O preposto não tinha poderes para transigir.", "negacao_adverbial"),
    ("neg-04", "Houve prévio requerimento administrativo ao INSS.", "Não houve prévio requerimento administrativo ao INSS.", "negacao_adverbial"),
    ("neg-05", "O acusado confessou a autoria do delito.", "O acusado não confessou a autoria do delito.", "negacao_adverbial"),
    ("neg-06", "A citação foi realizada de forma válida.", "A citação não foi realizada de forma válida.", "negacao_adverbial"),

    # --- ANTONÍMIA DE NÚCLEO (6) ---
    ("ant-01", "O pedido de tutela de urgência foi deferido.", "O pedido de tutela de urgência foi indeferido.", "antonimia_nucleo"),
    ("ant-02", "A sentença julgou procedente a demanda.", "A sentença julgou improcedente a demanda.", "antonimia_nucleo"),
    ("ant-03", "O recurso foi conhecido e provido.", "O recurso foi conhecido e desprovido.", "antonimia_nucleo"),
    ("ant-04", "O contrato foi considerado válido pelo juízo.", "O contrato foi considerado nulo pelo juízo.", "antonimia_nucleo"),
    ("ant-05", "A preliminar de prescrição foi acolhida.", "A preliminar de prescrição foi rejeitada.", "antonimia_nucleo"),
    ("ant-06", "O magistrado deferiu a produção da prova pericial.", "O magistrado indeferiu a produção da prova pericial.", "antonimia_nucleo"),

    # --- INVERSÃO DE POLO (16) ---
    ("pol-01", "O autor deve indenizar o réu pelos danos morais.", "O réu deve indenizar o autor pelos danos morais.", "inversao_polo"),
    ("pol-02", "Condeno a parte ré ao pagamento das custas processuais.", "Condeno a parte autora ao pagamento das custas processuais.", "inversao_polo"),
    ("pol-03", "A apelante sustenta a tese da nulidade.", "A apelada sustenta a tese da nulidade.", "inversao_polo"),
    ("pol-04", "O exequente é parte legítima para a execução.", "O executado é parte legítima para a execução.", "inversao_polo"),
    ("pol-05", "O autor deve pagar honorários ao advogado do réu.", "O réu deve pagar honorários ao advogado do autor.", "inversao_polo"),
    ("pol-06", "Cabe ao impetrante demonstrar o direito líquido e certo.", "Cabe à autoridade coatora demonstrar o direito líquido e certo.", "inversao_polo"),
    ("pol-07", "O agravante requer a reforma da decisão.", "O agravado requer a reforma da decisão.", "inversao_polo"),
    ("pol-08", "O embargante alega excesso de execução.", "O embargado alega excesso de execução.", "inversao_polo"),
    ("pol-09", "A vítima reconheceu o acusado na audiência.", "O acusado reconheceu a vítima na audiência.", "inversao_polo"),
    ("pol-10", "O locador deve restituir a caução ao locatário.", "O locatário deve restituir a caução ao locador.", "inversao_polo"),
    ("pol-11", "O credor cedeu o crédito ao terceiro interessado.", "O devedor cedeu o crédito ao terceiro interessado.", "inversao_polo"),
    ("pol-12", "Compete ao requerente provar o fato constitutivo.", "Compete ao requerido provar o fato constitutivo.", "inversao_polo"),
    ("pol-13", "O reclamante comprovou a jornada extraordinária.", "A reclamada comprovou a jornada extraordinária.", "inversao_polo"),
    ("pol-14", "O alimentante pleiteia a redução da pensão.", "O alimentando pleiteia a redução da pensão.", "inversao_polo"),
    ("pol-15", "O comprador deve entregar a coisa ao vendedor.", "O vendedor deve entregar a coisa ao comprador.", "inversao_polo"),
    ("pol-16", "O mandante responde pelos atos do mandatário.", "O mandatário responde pelos atos do mandante.", "inversao_polo"),

    # --- QUANTIFICADOR / MODAL (4) ---
    ("mod-01", "Todos os requisitos da aposentadoria foram preenchidos.", "Nenhum dos requisitos da aposentadoria foi preenchido.", "quant_modal"),
    ("mod-02", "O magistrado deve aplicar a pena no mínimo legal.", "O magistrado não deve aplicar a pena no mínimo legal.", "quant_modal"),
    ("mod-03", "É cabível a concessão da gratuidade de justiça.", "É incabível a concessão da gratuidade de justiça.", "quant_modal"),
    ("mod-04", "A prova é suficiente para a condenação.", "A prova é insuficiente para a condenação.", "quant_modal"),

    # --- CONTROLE PAREADO (8) — paráfrase legítima, Jaccard equiparado ---
    ("cpar-01", "O réu não compareceu à audiência de instrução.", "O réu não compareceu à audiência de inquirição.", "controle_parafrase"),
    ("cpar-02", "O pedido de tutela de urgência foi indeferido.", "O pedido de tutela de urgência foi rejeitado.", "controle_parafrase"),
    ("cpar-03", "A sentença julgou improcedente a demanda.", "A sentença julgou improcedente a ação.", "controle_parafrase"),
    ("cpar-04", "O autor deve indenizar o réu pelos danos morais.", "O autor deve ressarcir o réu pelos danos morais.", "controle_parafrase"),
    ("cpar-05", "Nenhum dos requisitos da aposentadoria foi preenchido.", "Nenhum dos requisitos da aposentadoria foi satisfeito.", "controle_parafrase"),
    ("cpar-06", "O acusado não confessou a autoria do delito.", "O acusado não confessou a autoria do crime.", "controle_parafrase"),
    ("cpar-07", "A citação não foi realizada de forma válida.", "A citação não foi efetuada de forma válida.", "controle_parafrase"),
    ("cpar-08", "A preliminar de prescrição foi acolhida.", "A preliminar de prescrição foi admitida.", "controle_parafrase"),
]

# ===========================================================================
# 0) Qual modelo usar. Tenta a lista em ordem; usa o 1o presente no cache.
#    Se souber o nome/caminho exato, ponha em MODELO_FORCADO.
# ===========================================================================
MODELO_FORCADO = None   # ex.: "intfloat/multilingual-e5-large" ou r"C:\caminho\bge-m3"
CANDIDATOS = [
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-small",
    "BAAI/bge-m3",
    "BAAI/bge-large-en-v1.5",
]

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("ERRO: sentence-transformers nao instalado neste ambiente.")
    print(e); sys.exit(1)


def carrega_modelo():
    nomes = [MODELO_FORCADO] if MODELO_FORCADO else CANDIDATOS
    ultimo = None
    for nome in nomes:
        if not nome:
            continue
        try:
            return nome, SentenceTransformer(nome)
        except Exception as ex:
            ultimo = (nome, str(ex)[:160])
    print("ERRO: nenhum modelo encontrado no cache local offline.")
    if ultimo:
        print(f"  ultima tentativa: {ultimo[0]} -> {ultimo[1]}")
    print("  Para descobrir o nome exato do que voce tem, rode:")
    print('    python -c "from huggingface_hub import scan_cache_dir; print([r.repo_id for r in scan_cache_dir().repos])"')
    print("  e ponha o nome em MODELO_FORCADO no topo do script.")
    sys.exit(1)


NOME, M = carrega_modelo()
EH_E5 = "e5" in NOME.lower()
print(f"[MODELO] {NOME}  (prefixos e5={'sim' if EH_E5 else 'nao'})\n", flush=True)


def enc(textos, tipo="query"):
    if EH_E5:
        textos = [f"{tipo}: {t}" for t in textos]
    return M.encode(textos, normalize_embeddings=True, show_progress_bar=False)


def cos(u, v):
    return float(np.dot(u, v))


def boot_ci(vals, fn=st.mean, n=5000, a=0.025):
    if len(vals) < 2:
        return (vals[0], vals[0]) if vals else (0.0, 0.0)
    bs = sorted(fn([random.choice(vals) for _ in vals]) for _ in range(n))
    return bs[int(n * a)], bs[int(n * (1 - a))]


# 1) Cosseno por estrato
ESTRATOS = ["negacao_adverbial", "antonimia_nucleo", "inversao_polo",
            "quant_modal", "controle_parafrase"]
por_estrato = {e: [] for e in ESTRATOS}
for pid, a, b, cat in PARES:
    ea, eb = enc([a, b], "query")
    por_estrato[cat].append(cos(ea, eb))

print("=== 1) COSSENO POR ESTRATO (com IC95) ===")
agg = {}
for e in ESTRATOS:
    v = por_estrato[e]
    lo, hi = boot_ci(v)
    agg[e] = {"mu": round(st.mean(v), 3), "sd": round(st.pstdev(v), 3),
              "lo": round(lo, 3), "hi": round(hi, 3), "n": len(v)}
    print(f"  {e:18s} mu={st.mean(v):.3f}  IC95=[{lo:.3f},{hi:.3f}]  n={len(v)}")

# 2) Gap pareado
contr = [c for e in ESTRATOS if e != "controle_parafrase" for c in por_estrato[e]]
ctrl = por_estrato["controle_parafrase"]


def boot_gap(contr, ctrl, n=5000, a=0.025):
    bs = sorted(st.mean([random.choice(ctrl) for _ in ctrl]) -
                st.mean([random.choice(contr) for _ in contr]) for _ in range(n))
    return bs[int(n * a)], bs[int(n * (1 - a))]


gap = st.mean(ctrl) - st.mean(contr)
glo, ghi = boot_gap(contr, ctrl)
print("\n=== 2) GAP PAREADO (contrastivo vs controle, mesmo Jaccard) ===")
print(f"  contrastivo_mu={st.mean(contr):.3f}  controle_mu={st.mean(ctrl):.3f}")
print(f"  gap={gap:+.3f}  IC95=[{glo:+.3f},{ghi:+.3f}]  "
      f"{'exclui 0 (SEPARA)' if glo > 0 else 'inclui 0 (NAO separa)'}")

# 3) Anisotropia
NEUTRAS = [
    "o réu compareceu à audiência de instrução",
    "a tarifa de energia subiu no último trimestre",
    "a fotossíntese converte luz em energia química",
    "o pedido de tutela de urgência foi deferido",
    "a seleção venceu a partida nos pênaltis",
    "o algoritmo ordena a lista em tempo logarítmico",
    "a receita leva farinha, ovos e açúcar",
    "a prescrição extingue a pretensão do credor",
    "o satélite entrou em órbita baixa terrestre",
    "as chuvas alagaram a zona sul da cidade",
    "a aposentadoria exige tempo mínimo de contribuição",
    "o pintor usou tons de azul e ocre",
    "o contrato foi considerado nulo pelo juízo",
    "a vacina reduz a carga viral circulante",
    "o trem atrasou por falha na sinalização",
    "a apelação foi conhecida e provida pelo tribunal",
]
EN = enc(NEUTRAS, "query")
rand_cos = [cos(EN[i], EN[j]) for i in range(len(EN)) for j in range(i + 1, len(EN))]
aniso = st.mean(rand_cos)
pol_mu = agg["inversao_polo"]["mu"]
print("\n=== 3) ANISOTROPIA (baseline do espaco) ===")
print(f"  cosseno medio entre frases aleatorias = {aniso:.3f}")
print(f"  cosseno inversao_polo = {pol_mu:.3f}   delta = {pol_mu - aniso:+.3f}")
print(f"  {'espaco ISOTROPICO: cosseno alto eh sinal real' if aniso < 0.4 else 'espaco ANISOTROPICO: cuidado'}")

# 4) Recuperacao com distratores (assimetrico)
CASOS = [
    ("o réu esteve presente na audiência", "o réu compareceu à audiência de instrução", "o réu não compareceu à audiência de instrução"),
    ("a tutela de urgência foi concedida", "o pedido de tutela de urgência foi deferido", "o pedido de tutela de urgência foi indeferido"),
    ("a ação foi julgada a favor do autor", "a sentença julgou procedente a demanda", "a sentença julgou improcedente a demanda"),
    ("o pagamento do débito ficou comprovado", "a parte autora comprovou o pagamento do débito", "a parte autora não comprovou o pagamento do débito"),
    ("o contrato foi reconhecido como eficaz", "o contrato foi considerado válido pelo juízo", "o contrato foi considerado nulo pelo juízo"),
    ("a prescrição foi reconhecida", "a preliminar de prescrição foi acolhida", "a preliminar de prescrição foi rejeitada"),
    ("a prova basta para condenar", "a prova é suficiente para a condenação", "a prova é insuficiente para a condenação"),
    ("o recurso teve êxito", "o recurso foi conhecido e provido", "o recurso foi conhecido e desprovido"),
    ("o acusado admitiu o crime", "o acusado confessou a autoria do delito", "o acusado não confessou a autoria do delito"),
    ("a perícia foi autorizada", "o magistrado deferiu a produção da prova pericial", "o magistrado indeferiu a produção da prova pericial"),
]
DISTRATORES = [
    "o juízo determinou a citação por edital do executado",
    "as custas processuais foram recolhidas pela parte vencida",
    "o perito apresentou laudo complementar no prazo legal",
    "a audiência de conciliação foi remarcada pelo cartório",
    "o mandado de penhora recaiu sobre o imóvel residencial",
    "a parte interpôs agravo de instrumento contra a decisão",
    "o magistrado homologou o acordo celebrado entre as partes",
    "a sentença foi publicada no diário oficial eletrônico",
    "o recurso especial foi inadmitido na origem pelo tribunal",
    "a parte requereu a produção de prova testemunhal adicional",
    "o oficial de justiça certificou o cumprimento da diligência",
    "a defesa arguiu a incompetência absoluta do juízo cível",
]
ED = enc(DISTRATORES, "passage")
rank1 = 0
oposta_acima = 0
margens = []
for q, corr, opo in CASOS:
    eq = enc([q], "query")[0]
    ec = enc([corr], "passage")[0]
    eo = enc([opo], "passage")[0]
    corpus = [("CORRETA", ec), ("OPOSTA", eo)] + [("D", d) for d in ED]
    sims = sorted(((cos(eq, v), tag) for tag, v in corpus), reverse=True)
    tags = [t for _, t in sims]
    if tags[0] == "CORRETA":
        rank1 += 1
    if tags.index("OPOSTA") < tags.index("CORRETA"):
        oposta_acima += 1
    margens.append(cos(eq, ec) - cos(eq, eo))

print("\n=== 4) RECUPERACAO COM DISTRATORES (2 + 12, assimetrico) ===")
print(f"  correta em rank 1: {rank1}/{len(CASOS)}")
print(f"  oposta ranqueada ACIMA da correta: {oposta_acima}/{len(CASOS)}")
print(f"  margem(correta-oposta): media={st.mean(margens):+.3f}  min={min(margens):+.3f}")

# 5) Salva e imprime JSON
saida = {
    "modelo": NOME,
    "estratos": agg,
    "gap_pareado": {"contr": round(st.mean(contr), 3), "ctrl": round(st.mean(ctrl), 3),
                    "gap": round(gap, 3), "lo": round(glo, 3), "hi": round(ghi, 3)},
    "anisotropia": {"baseline_aleatorio": round(aniso, 3), "inversao_polo": round(pol_mu, 3),
                    "delta": round(pol_mu - aniso, 3)},
    "recuperacao_distratores": {"rank1": rank1, "total": len(CASOS),
                                "oposta_acima": oposta_acima,
                                "margem_media": round(st.mean(margens), 3),
                                "margem_min": round(min(margens), 3)},
}
with open("resultado_e5.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)
print("\n[OK] resultado salvo em resultado_e5.json")
print("\n----- COLE O BLOCO ABAIXO NA CONVERSA -----")
print(json.dumps(saida, ensure_ascii=False))
