from flask import Flask, render_template, request, session
import pandas as pd
import joblib

app = Flask(__name__)
app.secret_key = "secreta"

# === Carregar modelos ===
modelo = joblib.load("modelo_embeddings_cores.pkl")
kmeans = joblib.load("modelo_kmeans_cores.pkl")

# === Carregar rótulos ===
rotulos = pd.read_excel("cores_para_rotular.xlsx")
rotulos["cor_normalizada"] = rotulos["cor_normalizada"].str.strip().str.lower()

grupo_para_cor_pai = (
    rotulos.groupby("grupo_cor")["cor_pai"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Não definido")
    .to_dict()
)

def normalizar_texto(texto):
    return str(texto).strip().lower()

# === Função para ordenar tamanhos ===
def ordena_tamanho(x):
    ordem_padrao = ["PP", "P", "M", "G", "GG", "XG", "XGG"]
    s = str(x).strip().upper()
    if s in ordem_padrao:
        return ordem_padrao.index(s)
    try:
        return 100 + int(s)  # Tamanhos numéricos depois dos padrões
    except:
        return 999  # Outros no final

@app.route("/", methods=["GET", "POST"])
@app.route("/analise", methods=["GET", "POST"])
def analise():
    sugestao = None
    distribuicao_loja = {}
    total_itens = 0
    filtro_subgrupo = None
    filtro_secao = None

    dados = pd.read_excel("Tabela.xlsx")
    subgrupos_unicos = sorted(dados["nome_sub_grupo"].dropna().unique())
    secoes_unicas = sorted(dados["nome_secao"].dropna().unique())

    if request.method == "POST":
        filtro_subgrupo = request.form.get("subgrupo")
        filtro_secao = request.form.get("secao")
        session["subgrupo"] = filtro_subgrupo
        session["secao"] = filtro_secao

    dados_filtrados = dados
    if filtro_subgrupo:
        dados_filtrados = dados_filtrados[dados_filtrados["nome_sub_grupo"] == filtro_subgrupo]
    if filtro_secao:
        dados_filtrados = dados_filtrados[dados_filtrados["nome_secao"] == filtro_secao]

    if not dados_filtrados.empty:
        total_itens = dados_filtrados.shape[0]
        distribuicao_loja = dados_filtrados["codigo_loja"].value_counts().to_dict()
        lojas = dados_filtrados["codigo_loja"].unique()
        todas_combinacoes = dados_filtrados[["cor_1", "tamanho"]].drop_duplicates()

        faltas = []
        sobras = []
        sugestoes = []

        for _, row in todas_combinacoes.iterrows():
            cor = row["cor_1"]
            tam = row["tamanho"]
            lojas_com_item = dados_filtrados[
                (dados_filtrados["cor_1"] == cor) & (dados_filtrados["tamanho"] == tam)
            ]["codigo_loja"].value_counts().to_dict()

            for loja in lojas:
                qtd = lojas_com_item.get(loja, 0)
                if qtd == 0:
                    faltas.append({"cor": cor, "tamanho": tam, "loja": loja})
                elif qtd > 1:
                    sobras.append({"cor": cor, "tamanho": tam, "loja": loja, "qtd": qtd})

        for falta in faltas:
            for sobra in sobras:
                if (
                    falta["cor"] == sobra["cor"]
                    and falta["tamanho"] == sobra["tamanho"]
                    and sobra["qtd"] > 1
                ):
                    sugestoes.append({
                        "cor": falta["cor"],
                        "tamanho": falta["tamanho"],
                        "loja_origem": sobra["loja"],
                        "loja_destino": falta["loja"]
                    })
                    sobra["qtd"] -= 1
                    break

        sugestao = {
            "loja_prioritaria": "Baseada em redistribuição por faltas",
            "faltando": {"total": len(faltas)},
            "reposicoes": sugestoes
        }

    return render_template(
        "analise.html",
        subgrupos=subgrupos_unicos,
        secoes=secoes_unicas,
        filtro_subgrupo=filtro_subgrupo,
        filtro_secao=filtro_secao,
        total=total_itens,
        distribuicao=distribuicao_loja,
        sugestao=sugestao
    )

@app.route("/loja/<int:codigo_loja>")
def detalhes_loja(codigo_loja):
    dados = pd.read_excel("Tabela.xlsx")
    subgrupo = session.get("subgrupo")
    secao = session.get("secao")

    if subgrupo:
        dados = dados[dados["nome_sub_grupo"] == subgrupo]
    if secao:
        dados = dados[dados["nome_secao"] == secao]

    loja_dados = dados[dados["codigo_loja"] == codigo_loja]

    tamanhos_ordenados = sorted(
        loja_dados["tamanho"].dropna().unique(),
        key=ordena_tamanho
    )
    cores_ordenadas = sorted(loja_dados["cor_1"].dropna().unique())

    tabela_pivotada = (
        loja_dados.groupby(["cor_1", "tamanho"])
        .size()
        .reset_index(name="quantidade")
        .pivot(index="cor_1", columns="tamanho", values="quantidade")
        .reindex(index=cores_ordenadas, columns=tamanhos_ordenados)
        .fillna(0)
        .astype(int)
    )

    todas_combinacoes = dados[["cor_1", "tamanho"]].drop_duplicates()
    faltando = []
    sobrando = []
    for _, row in todas_combinacoes.iterrows():
        cor = row["cor_1"]
        tam = row["tamanho"]
        qtd = loja_dados[
            (loja_dados["cor_1"] == cor) & (loja_dados["tamanho"] == tam)
        ].shape[0]
        if qtd == 0:
            faltando.append({"cor": cor, "tamanho": tam})
        elif qtd > 1:
            sobrando.append({"cor": cor, "tamanho": tam, "quantidade": qtd})

    faltando = sorted(faltando, key=lambda x: (x["cor"], ordena_tamanho(x["tamanho"])))
    sobrando = sorted(sobrando, key=lambda x: (x["cor"], ordena_tamanho(x["tamanho"])))

    return render_template(
        "loja.html",
        codigo_loja=codigo_loja,
        tabela=tabela_pivotada.reset_index().to_dict(orient="records"),
        colunas=tamanhos_ordenados,
        faltando=faltando,
        sobrando=sobrando
    )

@app.route("/index", methods=["GET", "POST"])
def index():
    resultado = None
    if request.method == "POST":
        cor_digitada = request.form["cor"]
        cor_normalizada = normalizar_texto(cor_digitada)
        embedding = modelo.encode([cor_normalizada])
        grupo = kmeans.predict(embedding)[0]

        cor_pai_encontrada = rotulos.loc[
            rotulos["cor_normalizada"] == cor_normalizada, "cor_pai"
        ]
        if not cor_pai_encontrada.empty:
            cor_pai = cor_pai_encontrada.values[0]
        else:
            cor_pai = grupo_para_cor_pai.get(grupo, "Não definido")

        resultado = {
            "cor_digitada": cor_digitada,
            "cor_normalizada": cor_normalizada,
            "grupo": grupo,
            "cor_pai": cor_pai
        }

    return render_template("index.html", resultado=resultado)

if __name__ == "__main__":
    app.run(debug=True)
