# -*- coding: utf-8 -*-
"""
Recuperação v2 (Ataque 4): query temática vs {correto, oposto}.
20 casos cobrindo negação, antonímia, polo e modal.
Mede acerto de ranking E margem, com IC95 da taxa de acerto.
"""
import numpy as np, statistics as st, random, json
from sentence_transformers import SentenceTransformer
random.seed(42)

# (query temática com intenção de polaridade, frase correta, frase oposta)
CASOS=[
 ("o réu esteve presente na audiência","o réu compareceu à audiência de instrução","o réu não compareceu à audiência de instrução"),
 ("a tutela de urgência foi concedida","o pedido de tutela de urgência foi deferido","o pedido de tutela de urgência foi indeferido"),
 ("a ação foi julgada a favor do autor","a sentença julgou procedente a demanda","a sentença julgou improcedente a demanda"),
 ("o pagamento do débito ficou comprovado","a parte autora comprovou o pagamento do débito","a parte autora não comprovou o pagamento do débito"),
 ("o contrato foi reconhecido como eficaz","o contrato foi considerado válido pelo juízo","o contrato foi considerado nulo pelo juízo"),
 ("a prescrição foi reconhecida","a preliminar de prescrição foi acolhida","a preliminar de prescrição foi rejeitada"),
 ("os requisitos da aposentadoria estão presentes","todos os requisitos da aposentadoria foram preenchidos","nenhum dos requisitos da aposentadoria foi preenchido"),
 ("a prova basta para condenar","a prova é suficiente para a condenação","a prova é insuficiente para a condenação"),
 ("o recurso teve êxito","o recurso foi conhecido e provido","o recurso foi conhecido e desprovido"),
 ("a citação é válida","a citação foi realizada de forma válida","a citação não foi realizada de forma válida"),
 ("o acusado admitiu o crime","o acusado confessou a autoria do delito","o acusado não confessou a autoria do delito"),
 ("a perícia foi autorizada","o magistrado deferiu a produção da prova pericial","o magistrado indeferiu a produção da prova pericial"),
 ("o preposto podia fazer acordo","o preposto tinha poderes para transigir","o preposto não tinha poderes para transigir"),
 ("houve pedido administrativo ao INSS","houve prévio requerimento administrativo ao INSS","não houve prévio requerimento administrativo ao INSS"),
 ("a gratuidade pode ser concedida","é cabível a concessão da gratuidade de justiça","é incabível a concessão da gratuidade de justiça"),
 # polo: query indica QUEM tem a obrigação
 ("o réu é quem paga a indenização","o autor deve indenizar o réu pelos danos morais","o réu deve indenizar o autor pelos danos morais"),
 ("a parte autora arca com as custas","condeno a parte ré ao pagamento das custas processuais","condeno a parte autora ao pagamento das custas processuais"),
 ("o executado tem legitimidade","o exequente é parte legítima para a execução","o executado é parte legítima para a execução"),
 ("o locatário devolve a caução","o locador deve restituir a caução ao locatário","o locatário deve restituir a caução ao locador"),
 ("o requerido prova o fato","compete ao requerente provar o fato constitutivo","compete ao requerido provar o fato constitutivo"),
]
# Nota: nos casos de polo, a "frase correta" é a que responde à intenção da query.
# Para pol, a query descreve uma situação; a frase correta é a que NÃO corresponde
# à query — então invertemos: a query "o réu é quem paga" tem como resposta correta
# "o réu deve indenizar o autor" (frase oposta). Corrigir o gabarito:
GABARITO_INVERTE={15,16,17,18,19}  # índices onde a frase B é a correta p/ a query

MODELOS=[('MiniLM','sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
         ('mpnet','sentence-transformers/paraphrase-multilingual-mpnet-base-v2'),
         ('distiluse','sentence-transformers/distiluse-base-multilingual-cased-v2')]

out={}
for nome,hub in MODELOS:
    m=SentenceTransformer(hub)
    acertos=0; margens=[]; falhas=[]
    flags=[]
    for i,(q,a,b) in enumerate(CASOS):
        correto,oposto=(b,a) if i in GABARITO_INVERTE else (a,b)
        eq,ec,eo=m.encode([q,correto,oposto],normalize_embeddings=True,show_progress_bar=False)
        sc,so=float(np.dot(eq,ec)),float(np.dot(eq,eo))
        ok=sc>so; acertos+=ok; margens.append(sc-so); flags.append(ok)
        if not ok: falhas.append((q,round(sc,3),round(so,3)))
    # IC95 bootstrap da taxa
    bs=sorted(sum(random.choice(flags) for _ in flags)/len(flags) for _ in range(5000))
    lo,hi=bs[125],bs[4875]
    out[nome]={'acertos':acertos,'total':len(CASOS),'taxa':round(acertos/len(CASOS),3),
               'ic_lo':round(lo,3),'ic_hi':round(hi,3),
               'margem_media':round(st.mean(margens),3),'margem_min':round(min(margens),3),
               'falhas':falhas}
    print(f'{nome:10s}: {acertos}/{len(CASOS)} taxa={acertos/len(CASOS):.2f} IC95=[{lo:.2f},{hi:.2f}] margem_med={st.mean(margens):+.3f} margem_min={min(margens):+.3f}')
    for f in falhas: print(f'     FALHA: "{f[0]}" correto={f[1]} oposto={f[2]}')

json.dump(out,open('recup_v2.json','w'),ensure_ascii=False,indent=2)
