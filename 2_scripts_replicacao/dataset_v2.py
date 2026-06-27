# -*- coding: utf-8 -*-
"""
Dataset v2 — corrige Ataques 1 e 2.
- inversao_polo ampliado de 4 para 16 pares (resposta ao n=4)
- controle de parafrase LEXICALMENTE PAREADO substitui o controle injusto
- mantem os demais estratos
"""
PARES = [
    # --- NEGAÇÃO ADVERBIAL (6) ---
    ("neg-01","O réu compareceu à audiência de instrução.","O réu não compareceu à audiência de instrução.","negacao_adverbial"),
    ("neg-02","A parte autora comprovou o pagamento do débito.","A parte autora não comprovou o pagamento do débito.","negacao_adverbial"),
    ("neg-03","O preposto tinha poderes para transigir.","O preposto não tinha poderes para transigir.","negacao_adverbial"),
    ("neg-04","Houve prévio requerimento administrativo ao INSS.","Não houve prévio requerimento administrativo ao INSS.","negacao_adverbial"),
    ("neg-05","O acusado confessou a autoria do delito.","O acusado não confessou a autoria do delito.","negacao_adverbial"),
    ("neg-06","A citação foi realizada de forma válida.","A citação não foi realizada de forma válida.","negacao_adverbial"),

    # --- ANTONÍMIA DE NÚCLEO (6) ---
    ("ant-01","O pedido de tutela de urgência foi deferido.","O pedido de tutela de urgência foi indeferido.","antonimia_nucleo"),
    ("ant-02","A sentença julgou procedente a demanda.","A sentença julgou improcedente a demanda.","antonimia_nucleo"),
    ("ant-03","O recurso foi conhecido e provido.","O recurso foi conhecido e desprovido.","antonimia_nucleo"),
    ("ant-04","O contrato foi considerado válido pelo juízo.","O contrato foi considerado nulo pelo juízo.","antonimia_nucleo"),
    ("ant-05","A preliminar de prescrição foi acolhida.","A preliminar de prescrição foi rejeitada.","antonimia_nucleo"),
    ("ant-06","O magistrado deferiu a produção da prova pericial.","O magistrado indeferiu a produção da prova pericial.","antonimia_nucleo"),

    # --- INVERSÃO DE POLO (16) — AMPLIADO ---
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

    # --- QUANTIFICADOR / MODAL (4) ---
    ("mod-01","Todos os requisitos da aposentadoria foram preenchidos.","Nenhum dos requisitos da aposentadoria foi preenchido.","quant_modal"),
    ("mod-02","O magistrado deve aplicar a pena no mínimo legal.","O magistrado não deve aplicar a pena no mínimo legal.","quant_modal"),
    ("mod-03","É cabível a concessão da gratuidade de justiça.","É incabível a concessão da gratuidade de justiça.","quant_modal"),
    ("mod-04","A prova é suficiente para a condenação.","A prova é insuficiente para a condenação.","quant_modal"),

    # --- CONTROLE PAREADO (8) — substitui o controle injusto ---
    ("cpar-01","O réu não compareceu à audiência de instrução.","O réu não compareceu à audiência de inquirição.","controle_parafrase"),
    ("cpar-02","O pedido de tutela de urgência foi indeferido.","O pedido de tutela de urgência foi rejeitado.","controle_parafrase"),
    ("cpar-03","A sentença julgou improcedente a demanda.","A sentença julgou improcedente a ação.","controle_parafrase"),
    ("cpar-04","O autor deve indenizar o réu pelos danos morais.","O autor deve ressarcir o réu pelos danos morais.","controle_parafrase"),
    ("cpar-05","Nenhum dos requisitos da aposentadoria foi preenchido.","Nenhum dos requisitos da aposentadoria foi satisfeito.","controle_parafrase"),
    ("cpar-06","O acusado não confessou a autoria do delito.","O acusado não confessou a autoria do crime.","controle_parafrase"),
    ("cpar-07","A citação não foi realizada de forma válida.","A citação não foi efetuada de forma válida.","controle_parafrase"),
    ("cpar-08","A preliminar de prescrição foi acolhida.","A preliminar de prescrição foi admitida.","controle_parafrase"),
]

def jaccard(a,b):
    sa=set(a.lower().replace('.','').replace(',','').split())
    sb=set(b.lower().replace('.','').replace(',','').split())
    return len(sa&sb)/len(sa|sb)

if __name__=="__main__":
    from collections import Counter
    import statistics as st
    c=Counter(p[3] for p in PARES)
    print("Total:",len(PARES))
    for k,v in c.items(): print(f"  {k}: {v}")
    print("\nJaccard por grupo (verificar pareamento):")
    contr=[jaccard(p[1],p[2]) for p in PARES if p[3]!='controle_parafrase']
    ctrl=[jaccard(p[1],p[2]) for p in PARES if p[3]=='controle_parafrase']
    print(f"  contrastivos: {st.mean(contr):.3f}")
    print(f"  controle pareado: {st.mean(ctrl):.3f}")
