# -*- coding: utf-8 -*-
# experimento_reasoning_polo_multi.py
#
# Roda a matriz MODELOS x FORMULACOES na inversao de polo (16 pares).
# Eixo do experimento: o ganho vem da CAPACIDADE do modelo ou da FORMULACAO
# da pergunta? Comparacao-chave: dentro de cada modelo, contradicao vs papel_tematico.
#
# Modelos Anthropic disponiveis na API: opus-4.8, sonnet-4.6, haiku-4.5.
# (Mythos/Fable estao com acesso suspenso por controle de exportacao.)
#
# Comparador externo: cross-encoder nli-deberta-v3-base = 9/16.
#
# Uso:
#   pip install anthropic
#   $env:ANTHROPIC_API_KEY="..."
#   python experimento_reasoning_polo_multi.py
#
# Salva resultado_multi.json e imprime a matriz + analise de padrao.

import os, sys, json, re, time

# Windows PowerShell usa cp1252 por padrao; respostas dos modelos podem conter
# caracteres unicode (setas, aspas tipograficas). Forca UTF-8 na saida.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

PARES = [
    ("pol-01","O autor deve indenizar o réu pelos danos morais.","O réu deve indenizar o autor pelos danos morais."),
    ("pol-02","Condeno a parte ré ao pagamento das custas processuais.","Condeno a parte autora ao pagamento das custas processuais."),
    ("pol-03","A apelante sustenta a tese da nulidade.","A apelada sustenta a tese da nulidade."),
    ("pol-04","O exequente é parte legítima para a execução.","O executado é parte legítima para a execução."),
    ("pol-05","O autor deve pagar honorários ao advogado do réu.","O réu deve pagar honorários ao advogado do autor."),
    ("pol-06","Cabe ao impetrante demonstrar o direito líquido e certo.","Cabe à autoridade coatora demonstrar o direito líquido e certo."),
    ("pol-07","O agravante requer a reforma da decisão.","O agravado requer a reforma da decisão."),
    ("pol-08","O embargante alega excesso de execução.","O embargado alega excesso de execução."),
    ("pol-09","A vítima reconheceu o acusado na audiência.","O acusado reconheceu a vítima na audiência."),
    ("pol-10","O locador deve restituir a caução ao locatário.","O locatário deve restituir a caução ao locador."),
    ("pol-11","O credor cedeu o crédito ao terceiro interessado.","O devedor cedeu o crédito ao terceiro interessado."),
    ("pol-12","Compete ao requerente provar o fato constitutivo.","Compete ao requerido provar o fato constitutivo."),
    ("pol-13","O reclamante comprovou a jornada extraordinária.","A reclamada comprovou a jornada extraordinária."),
    ("pol-14","O alimentante pleiteia a redução da pensão.","O alimentando pleiteia a redução da pensão."),
    ("pol-15","O comprador deve entregar a coisa ao vendedor.","O vendedor deve entregar a coisa ao comprador."),
    ("pol-16","O mandante responde pelos atos do mandatário.","O mandatário responde pelos atos do mandante."),
]

# pares que o cross-encoder NLI acertou (detectou contradicao) e errou
CROSSENC_ACERTOU = {"pol-01","pol-02","pol-04","pol-05","pol-09","pol-10","pol-13","pol-15","pol-16"}
CROSSENC_ERROU = [p[0] for p in PARES if p[0] not in CROSSENC_ACERTOU]  # os 7 reciprocos

# todos os modelos a testar. ajuste a lista se algum nome falhar.
MODELOS = [
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

FORMULACOES = {
    "contradicao": (
        "Considere as duas afirmações abaixo.\n"
        "Frase A: \"{A}\"\n"
        "Frase B: \"{B}\"\n"
        "Pergunta: as frases A e B se CONTRADIZEM, no sentido de que não podem ser "
        "ambas verdadeiras ao mesmo tempo? Responda apenas SIM ou NÃO."
    ),
    "papel_tematico": (
        "Considere as duas afirmações abaixo, que descrevem uma relação jurídica.\n"
        "Frase A: \"{A}\"\n"
        "Frase B: \"{B}\"\n"
        "Pergunta: a posição das partes (quem deve, quem paga, quem tem a obrigação "
        "ou a legitimidade) foi TROCADA entre A e B, ou cada parte conserva o mesmo "
        "papel? Responda apenas: TROCOU ou MANTEVE."
    ),
}

def tem(txt, palavra):
    return re.search(r"\b" + palavra + r"\b", txt, re.IGNORECASE) is not None

def classifica(fnome, txt):
    """Retorna True se identificou inversao, False se nao, None se ambiguo."""
    if fnome == "contradicao":
        sim, nao = tem(txt, "SIM"), tem(txt, r"N[ÃA]O")
        if sim and not nao: return True
        if nao and not sim: return False
        return None
    else:
        tr, mv = tem(txt, "TROCOU"), tem(txt, "MANTEVE")
        if tr and not mv: return True
        if mv and not tr: return False
        return None

def roda_modelo(client, modelo):
    """Retorna {formulacao: {pid: True/False/None}} e o texto bruto."""
    out = {f: {} for f in FORMULACOES}
    brutos = {}
    for pid, a, b in PARES:
        for fnome, tmpl in FORMULACOES.items():
            prompt = tmpl.format(A=a, B=b)
            for tentativa in range(3):
                try:
                    msg = client.messages.create(
                        model=modelo, max_tokens=256,
                        messages=[{"role":"user","content":prompt}])
                    txt = "".join(x.text for x in msg.content
                                  if getattr(x,"type","")=="text").strip()
                    break
                except Exception as e:
                    if tentativa == 2:
                        print(f"    [erro {pid}/{fnome}] {str(e)[:80]}")
                        txt = ""
                    else:
                        time.sleep(2)
            out[fnome][pid] = classifica(fnome, txt)
            brutos[f"{pid}__{fnome}"] = txt
    return out, brutos

def main():
    try:
        from anthropic import Anthropic
        client = Anthropic()
    except Exception as e:
        print("ERRO: pip install anthropic e defina ANTHROPIC_API_KEY.", e); sys.exit(1)

    resultados = {}
    todos_brutos = {}
    for modelo in MODELOS:
        print(f"\n[rodando] {modelo} (32 chamadas)...", flush=True)
        try:
            out, brutos = roda_modelo(client, modelo)
        except Exception as e:
            print(f"  [pulando {modelo}] {str(e)[:100]}")
            continue
        resultados[modelo] = out
        todos_brutos[modelo] = brutos
        c = sum(1 for v in out["contradicao"].values() if v is True)
        p = sum(1 for v in out["papel_tematico"].values() if v is True)
        print(f"  contradição: {c}/16   papel_temático: {p}/16", flush=True)

    json.dump({"resultados":resultados,"brutos":todos_brutos},
              open("resultado_multi.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- MATRIZ ----
    print("\n" + "="*60)
    print("MATRIZ: identificação da inversão de polo (de 16)")
    print("="*60)
    print(f"{'modelo':28s} {'contradição':>12s} {'papel temát.':>13s}")
    print(f"{'cross-encoder NLI (ref.)':28s} {'9':>12s} {'—':>13s}")
    for m in resultados:
        c = sum(1 for v in resultados[m]['contradicao'].values() if v is True)
        p = sum(1 for v in resultados[m]['papel_tematico'].values() if v is True)
        ca = [k for k,v in resultados[m]['contradicao'].items() if v is None]
        pa = [k for k,v in resultados[m]['papel_tematico'].items() if v is None]
        amb = ""
        if ca or pa: amb = f"  [ambíguos c={len(ca)} p={len(pa)}]"
        print(f"{m:28s} {c:>12d} {p:>13d}{amb}")

    # ---- ANALISE DE PADRAO (o que e robusto com n=16) ----
    print("\n" + "="*60)
    print("ANÁLISE DE PADRÃO (n=16: ler padrão, não placar)")
    print("="*60)
    for m in resultados:
        # dentro do modelo: ganho de papel_tematico sobre contradicao
        ganhos = [pid for pid,_,_ in PARES
                  if resultados[m]['papel_tematico'].get(pid) is True
                  and resultados[m]['contradicao'].get(pid) is not True]
        # recuperacao dos pares reciprocos que o cross-encoder errou
        rec = [pid for pid in CROSSENC_ERROU
               if resultados[m]['papel_tematico'].get(pid) is True]
        print(f"\n{m}:")
        print(f"  papel_temático acerta e contradição falha em: {len(ganhos)} pares")
        print(f"  recupera dos 7 recíprocos que o cross-encoder errou: {len(rec)}/7  {rec}")

    print("\n" + "="*60)
    print("LEITURA DA HIPÓTESE")
    print("="*60)
    print("Se TODOS os modelos (inclusive o mais barato) acertam sob papel_temático")
    print("e falham sob contradição, o ganho vem da FORMULAÇÃO, não da capacidade.")
    print("Se só o modelo maior acerta sob contradição, a capacidade compensa a")
    print("formulação ruim — tese se matiza. Reportar o padrão observado, não o placar.")

if __name__ == "__main__":
    main()
