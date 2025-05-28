import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, session, redirect, send_file
import openrouteservice
from urllib.parse import quote  # Coloque no topo do arquivo
from functools import lru_cache


app = Flask(__name__)
app.secret_key = "secreta"

# === Modelos de IA ===
modelo = joblib.load("modelo_embeddings_cores.pkl")
kmeans = joblib.load("modelo_kmeans_cores.pkl")

# === Rótulos ===
@lru_cache(maxsize=1)
def carregar_tabela_excel():
    return pd.read_excel("Tabela.xlsx")

@lru_cache(maxsize=1)
def carregar_rotulos_excel():
    rotulos = pd.read_excel("cores_para_rotular.xlsx")
    rotulos["cor_normalizada"] = rotulos["cor_normalizada"].str.strip().str.lower()
    return rotulos

rotulos = carregar_rotulos_excel()


def atualizar_grupo_para_cor_pai():
    return rotulos.groupby("grupo_cor")["cor_pai"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Não definido").to_dict()

grupo_para_cor_pai = atualizar_grupo_para_cor_pai()

# === Coordenadas ===
ORS_API_KEY = "5b3ce3597851110001cf6248ce13353e178941e1ab0c4eea7f18cfef"
ors_client = openrouteservice.Client(key=ORS_API_KEY)
coordenadas_lojas = {
    1: (-25.5332, -49.2222),
    2: (-25.3689, -49.2162),
    3: (-25.4427, -49.0623),
    4: (-25.3031, -49.0551),
    5: (-25.5912, -49.3533),
    6: (-25.6634, -49.3133),
    7: (-25.5862, -49.4102),
    9: (-25.6612, -49.3074),
    10: (-25.4711, -49.3355),
    11: (-25.0921, -50.1634),
    12: (-24.9151, -50.0982),
}


enderecos_lojas = {
    1: "Rua São José dos Pinhais, 107, Sítio Cercado, Curitiba/PR, 81910-010",
    2: "Rua Arquimedes, 18 Sala 05, Planta Maracanã, Colombo/PR, 83408-270",
    3: "Av. Getúlio Vargas, 894, Centro, Piraquara/PR, 83301-010",
    4: "Rua João Trevisan, 959, Jardim Paulista, Campina Grande do Sul/PR, 83430-000",
    5: "Rua Enette Dubard, 481, Tatuquara, Curitiba/PR, 81470-075",
    6: "Rua Cesar Carelli, 261, Pioneiros, Fazenda Rio Grande/PR, 83833-054",
    7: "Rua Carlos Cavalcanti, 69, Centro, Araucária/PR, 83702-470",
    9: "Rua Jacarandá, 208, Fazenda Rio Grande/PR, 83823-014",
    10: "Rua Raul Pompéia, 374, Fazendinha, Curitiba/PR, 81240-000",
    11: "Rua Fernandes Pinheiro, 74, Centro, Ponta Grossa/PR, 84010-135",
    12: "Avenida do Ouro, 128, Centro, Carambeí/PR, 84145-000"
}



def mapa(loja):
    endereco = enderecos_lojas.get(loja)
    if endereco:
        endereco_url = quote(endereco)  # codifica o texto para URL
        return f"https://www.google.com/maps/search/?api=1&query={endereco_url}"
    return "#"

def calcular_rota(loja_origem, loja_destino):
    try:
        origem = coordenadas_lojas[loja_origem]
        destino = coordenadas_lojas[loja_destino]
        rota = ors_client.directions(
            coordinates=[(origem[1], origem[0]), (destino[1], destino[0])],
            profile='driving-car', format='geojson'
        )
        segmento = rota['features'][0]['properties']['segments'][0]
        distancia_km = segmento['distance'] / 1000
        duracao_min = round(segmento['duration'] / 60)  # Arredonda para inteiro
        return round(distancia_km, 2), duracao_min
    except:
        return None, None


def normalizar_texto(texto):
    return str(texto).strip().lower()

def ordena_tamanho(x):
    prioridade = {"PP": 1, "P": 2, "M": 3, "G": 4, "GG": 5, "XG": 6, "XGG": 7}
    try:
        return prioridade.get(str(x).upper(), 1000 + int(x))
    except:
        return 9999

@app.route("/", methods=["GET", "POST"])
@app.route("/analise", methods=["GET", "POST"])

def analise():
    dados = carregar_tabela_excel().copy()
    sugestao = None
    distribuicao_loja = {}
    total_itens = 0
    filtro_subgrupo = filtro_secao = None

    dados = pd.read_excel("Tabela.xlsx")
    subgrupos_unicos = sorted(dados["nome_sub_grupo"].dropna().unique())
    secoes_unicas = sorted(dados["nome_secao"].dropna().unique())

    if request.method == "POST":
        filtro_subgrupo = request.form.get("subgrupo")
        filtro_secao = request.form.get("secao")
        session["subgrupo"] = filtro_subgrupo
        session["secao"] = filtro_secao

    if session.get("subgrupo"):
        dados = dados[dados["nome_sub_grupo"] == session["subgrupo"]]
    if session.get("secao"):
        dados = dados[dados["nome_secao"] == session["secao"]]

    if not dados.empty:
        total_itens = len(dados)
        distribuicao_loja = dados["codigo_loja"].value_counts().to_dict()
        lojas = dados["codigo_loja"].unique()
        todas_combinacoes = dados[["cor_1", "tamanho"]].drop_duplicates()

        faltas, sobras, sugestoes = [], [], []

        for _, row in todas_combinacoes.iterrows():
            cor, tam = row["cor_1"], row["tamanho"]
            lojas_com_item = dados[(dados["cor_1"] == cor) & (dados["tamanho"] == tam)]["codigo_loja"].value_counts().to_dict()
            for loja in lojas:
                qtd = lojas_com_item.get(loja, 0)
                if qtd == 0:
                    faltas.append({"cor": cor, "tamanho": tam, "loja": loja})
                elif qtd > 1:
                    sobras.append({"cor": cor, "tamanho": tam, "loja": loja, "quantidade": qtd})

        for falta in faltas:
            for sobra in sobras:
                if falta["cor"] == sobra["cor"] and falta["tamanho"] == sobra["tamanho"] and sobra["quantidade"] > 1:
                    distancia, tempo = calcular_rota(sobra["loja"], falta["loja"])
                    sugestoes.append({
                        "cor": falta["cor"],
                        "tamanho": falta["tamanho"],
                        "loja_origem": sobra["loja"],
                        "loja_destino": falta["loja"],
                        "endereco_origem": mapa(sobra["loja"]),
                        "endereco_destino": mapa(falta["loja"]),
                        "distancia_km": distancia,
                        "tempo_min": tempo
                    })
                    sobra["quantidade"] -= 1
                    break

        sugestao = {
            "loja_prioritaria": "Baseada em redistribuição por faltas",
            "faltando": {"total": len(faltas)},
            "reposicoes": sugestoes
        }

        pd.concat([pd.DataFrame(faltas), pd.DataFrame(sobras)]).to_csv("estoque_falta_sobra.csv", index=False)

    return render_template("analise.html", subgrupos=subgrupos_unicos, secoes=secoes_unicas,
                           filtro_subgrupo=session.get("subgrupo"), filtro_secao=session.get("secao"),
                           total=total_itens, distribuicao=distribuicao_loja, sugestao=sugestao)

@app.route("/download_csv")
def download_csv():
    path = "estoque_falta_sobra.csv"
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "Arquivo ainda não foi gerado.", 404

@app.route("/loja/<int:codigo_loja>")
def detalhes_loja(codigo_loja):
    dados = pd.read_excel("Tabela.xlsx")
    if session.get("subgrupo"):
        dados = dados[dados["nome_sub_grupo"] == session["subgrupo"]]
    if session.get("secao"):
        dados = dados[dados["nome_secao"] == session["secao"]]
    loja_dados = dados[dados["codigo_loja"] == codigo_loja]

    tamanhos_ordenados = sorted(loja_dados["tamanho"].dropna().unique(), key=ordena_tamanho)
    cores_ordenadas = sorted(loja_dados["cor_1"].dropna().unique())

    tabela_pivotada = (
        loja_dados.groupby(["cor_1", "tamanho"]).size().reset_index(name="quantidade")
        .pivot(index="cor_1", columns="tamanho", values="quantidade")
        .reindex(index=cores_ordenadas, columns=tamanhos_ordenados)
        .fillna(0).astype(int)
    )

    todas_combinacoes = dados[["cor_1", "tamanho"]].drop_duplicates()
    faltando, sobrando = [], []
    for _, row in todas_combinacoes.iterrows():
        cor, tam = row["cor_1"], row["tamanho"]
        qtd = loja_dados[(loja_dados["cor_1"] == cor) & (loja_dados["tamanho"] == tam)].shape[0]
        if qtd == 0:
            faltando.append({"cor": cor, "tamanho": tam})
        elif qtd > 1:
            sobrando.append({"cor": cor, "tamanho": tam, "quantidade": qtd})

    return render_template("loja.html",
        codigo_loja=codigo_loja,
        tabela=tabela_pivotada.reset_index().to_dict(orient="records"),
        colunas=tamanhos_ordenados,
        faltando=sorted(faltando, key=lambda x: (x["cor"], ordena_tamanho(x["tamanho"]))),
        sobrando=sorted(sobrando, key=lambda x: (x["cor"], ordena_tamanho(x["tamanho"])))
    )

@app.route("/index", methods=["GET", "POST"])
def index():
    resultado = None
    if request.method == "POST":
        cor_digitada = request.form["cor"]
        cor_normalizada = normalizar_texto(cor_digitada)
        embedding = modelo.encode([cor_normalizada])
        grupo = kmeans.predict(embedding)[0]

        cor_pai_encontrada = rotulos.loc[rotulos["cor_normalizada"] == cor_normalizada, "cor_pai"]
        cor_pai = cor_pai_encontrada.values[0] if not cor_pai_encontrada.empty else grupo_para_cor_pai.get(grupo, "Não definido")

        resultado = {
            "cor_digitada": cor_digitada,
            "cor_normalizada": cor_normalizada,
            "grupo": grupo,
            "cor_pai": cor_pai
        }

    return render_template("index.html", resultado=resultado)

@app.route("/rotular", methods=["GET", "POST"])
def rotular():
    cores_nao_rotuladas = rotulos[rotulos["cor_pai"].isnull()]["cor_normalizada"].dropna().unique()

    if request.method == "POST":
        cor = request.form["cor"]
        cor_pai = request.form["cor_pai"]
        rotulos.loc[rotulos["cor_normalizada"] == cor, "cor_pai"] = cor_pai
        rotulos.to_excel(rotulos_path, index=False)
        global grupo_para_cor_pai
        grupo_para_cor_pai = atualizar_grupo_para_cor_pai()
        return redirect("/rotular")

    return render_template("rotular.html", cores=sorted(cores_nao_rotuladas))

if __name__ == "__main__":
    app.run(debug=True)
