# teste-ia

# 📦 Sistema de Redistribuição Inteligente de Estoque

Este é um sistema web desenvolvido com **Flask** que analisa a distribuição de produtos por loja com base em **cores**, **tamanhos**, **subgrupos** e **seções**, propondo uma **redistribuição inteligente** de peças para garantir **grades completas** nas lojas prioritárias.

---

## 🚀 Funcionalidades

- 📊 **Análise por Subgrupo e Seção**
- 🧠 **Agrupamento inteligente de cores** com *embeddings* de linguagem (modelo `all-MiniLM-L6-v2`)
- 🧩 Verificação de **grade completa** (cores e tamanhos)
- 🔁 Sugestão de **redistribuição entre lojas**, priorizando:
  - Lojas âncora (1 a 7)
  - Loja de atacado (50)
- 📈 Planejamento para integrar dados reais de venda (em desenvolvimento)

---

## 🖼️ Interface

- Página `/analise`: Interface com seleção de subgrupo e seção.
- Página `/index`: Classificação individual de cor digitada.
- Relatórios automáticos:
  - Cores faltando
  - Tamanhos ausentes
  - Sugestões de transferência entre lojas

---

## 🧠 Inteligência Artificial

- Agrupamento de cores com `KMeans`
- Vetorização semântica com `SentenceTransformer`
- Geração de "cor pai" para unificação de nomes similares

---

## 📁 Estrutura do Projeto

teste-ia/
├── app.py # Código principal do Flask
├── teste.py # Script para gerar e aplicar agrupamentos
├── modelo_kmeans_cores.pkl # Modelo KMeans treinado
├── modelo_embeddings_cores.pkl # Modelo de embeddings
├── cores_para_rotular.xlsx # Arquivo com cores e grupos
├── templates/
│ ├── analise.html # Página de análise
│ └── index.html # Página de classificação manual
├── .gitignore
├── README.md
└── .venv/ # Ambiente virtual (não incluído no Git)

yaml
Copiar
Editar

---

## 🛠️ Como Executar

1. **Clone o repositório**:

```bash
git clone https://github.com/muriloh2001/teste-ia.git
cd teste-ia
Crie e ative o ambiente virtual:

bash
Copiar
Editar
python -m venv .venv
.\.venv\Scripts\activate
Instale as dependências:

bash
Copiar
Editar
pip install -r requirements.txt
Crie um requirements.txt com: pip freeze > requirements.txt

Execute o app:

bash
Copiar
Editar
python app.py
📌 Requisitos
Python 3.10+

Flask

pandas

scikit-learn

sentence-transformers

joblib

📢 Em breve
Integração com dados de vendas reais

Relatórios gráficos

Painel com alertas automatizados de grade incompleta

Exportação de recomendações para Excel

👨‍💻 Autor
Murilo H.
GitHub
