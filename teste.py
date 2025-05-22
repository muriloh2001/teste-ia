import os
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

# Função para normalizar nomes
def normalizar_texto(texto):
    return str(texto).strip().lower()

# === Carregar modelos ou treinar ===
if os.path.exists("modelo_kmeans_cores.pkl") and os.path.exists("modelo_embeddings_cores.pkl"):
    print("Carregando modelos existentes...")
    kmeans = joblib.load("modelo_kmeans_cores.pkl")
    modelo = joblib.load("modelo_embeddings_cores.pkl")
    df = pd.read_excel("Tabela.xlsx")
    df["cor_normalizada"] = df["cor_1"].apply(normalizar_texto)
    embeddings = modelo.encode(df["cor_normalizada"].tolist())
    df["grupo_cor"] = kmeans.predict(embeddings)
else:
    print("Modelos não encontrados. Treinando agora...")
    df = pd.read_excel("Tabela.xlsx")
    df = df.drop(columns=["cor_2", "cor_3", "%Total"], errors="ignore")
    df["cor_normalizada"] = df["cor_1"].apply(normalizar_texto)

    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = modelo.encode(df["cor_normalizada"].tolist())
    kmeans = KMeans(n_clusters=10, random_state=42)
    df["grupo_cor"] = kmeans.fit_predict(embeddings)

    agrupado = df[["cor_1", "cor_normalizada", "grupo_cor"]].drop_duplicates()
    agrupado.to_excel("cores_para_rotular.xlsx", index=False)
    joblib.dump(kmeans, "modelo_kmeans_cores.pkl")
    joblib.dump(modelo, "modelo_embeddings_cores.pkl")

# === Carregar rótulos manuais ===
try:
    rotulos = pd.read_excel("cores_para_rotular.xlsx")
    rotulos["cor_normalizada"] = rotulos["cor_normalizada"].apply(normalizar_texto)
    rotulos = rotulos[["cor_normalizada", "cor_pai"]].dropna().drop_duplicates()
    df = df.merge(rotulos, on="cor_normalizada", how="left")
    print("Rótulos carregados com sucesso.")
except FileNotFoundError:
    print("Arquivo 'cores_para_rotular.xlsx' não encontrado. Pule esta etapa.")

# === Função para sugerir cor pai ===
def sugerir_cor_pai(grupo, df):
    sugestoes = (
        df[df["grupo_cor"] == grupo]
        .dropna(subset=["cor_pai"])
        ["cor_pai"]
        .value_counts()
    )
    return sugestoes.index[0] if not sugestoes.empty else "Não definido", sugestoes.to_dict()

# === Histórico ===
historico = []

# === Interface interativa ===
while True:
    nova_cor = input("\nDigite uma cor para classificar (ou 'sair'): ")
    if nova_cor.lower() == "sair":
        break

    cor_normalizada = normalizar_texto(nova_cor)
    embedding = modelo.encode([cor_normalizada])
    grupo = kmeans.predict(embedding)[0]

    # Tenta obter cor pai direto dos rótulos
    cor_pai_manual = rotulos[rotulos["cor_normalizada"] == cor_normalizada]["cor_pai"]
    if not cor_pai_manual.empty:
        cor_pai = cor_pai_manual.values[0]
        print(f"✅ Cor pai identificada diretamente: {cor_pai}")
    else:
        # Sugestão com base no grupo
        cor_pai, sugestoes = sugerir_cor_pai(grupo, df)
        print(f"🤖 Cor pai inferida com base no grupo: {cor_pai}")
        if len(sugestoes) > 1:
            print("📋 Sugestões dentro do grupo:")
            for nome, freq in sugestoes.items():
                print(f"   - {nome} (freq: {freq})")
        
        # Salvar para revisão futura
        df_nova = pd.DataFrame([[nova_cor, cor_normalizada, grupo, cor_pai]], 
                               columns=["cor_1", "cor_normalizada", "grupo_cor", "cor_pai"])
        try:
            df_registro = pd.read_excel("novas_cores_para_rotular.xlsx")
            df_registro = pd.concat([df_registro, df_nova], ignore_index=True).drop_duplicates("cor_normalizada")
        except:
            df_registro = df_nova
        df_registro.to_excel("novas_cores_para_rotular.xlsx", index=False)
        print("💾 Cor registrada para revisão em 'novas_cores_para_rotular.xlsx'.")

    # Salvar no histórico
    historico.append({
        "entrada": nova_cor,
        "normalizada": cor_normalizada,
        "grupo": grupo,
        "cor_pai": cor_pai
    })

# === Salvar histórico ===
if historico:
    df_hist = pd.DataFrame(historico)
    df_hist.to_excel("historico_classificacoes.xlsx", index=False)
    print("📝 Histórico salvo em 'historico_classificacoes.xlsx'.")

# === Visualização com PCA ===
pca = PCA(n_components=2)
reducao = pca.fit_transform(embeddings)
df_plot = pd.DataFrame(reducao, columns=["x", "y"])
df_plot["grupo_cor"] = df["grupo_cor"]

plt.figure(figsize=(10, 7))
for grupo in sorted(df_plot["grupo_cor"].unique()):
    grupo_df = df_plot[df_plot["grupo_cor"] == grupo]
    plt.scatter(grupo_df["x"], grupo_df["y"], label=f"Grupo {grupo}", alpha=0.6)

plt.title("Visualização dos Grupos de Cores com PCA")
plt.legend()
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.tight_layout()
plt.savefig("visualizacao_clusters.png")
plt.show()
print("📊 Gráfico de clusters salvo como 'visualizacao_clusters.png'")
