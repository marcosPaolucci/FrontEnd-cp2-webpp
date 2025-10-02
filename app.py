import streamlit as st
import pandas as pd
from pycaret.clustering import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

# --- CONFIGURAÇÃO DA PÁGINA E CARREGAMENTO DOS DADOS ---

# Configuração inicial da página do Streamlit
st.set_page_config(
    page_title="Sistema de Recomendação de Filmes",
    page_icon="🎬",
    layout="wide"
)

# Carregando o modelo e o dataset
@st.cache_resource
def carregar_modelo_e_dados():
    """
    Carrega o modelo treinado e o CSV com os dados dos filmes.
    Também ajusta um TfidfVectorizer com base nas sinopses limpas.
    """
    try:
        modelo = load_model('final_model')
        df = pd.read_csv('filmes_com_clusters.csv')
        
        vectorizer = TfidfVectorizer(min_df=0.03, max_df=0.3, max_features=500)
        matriz_tfidf_completa = vectorizer.fit_transform(df['sinopse_limpa'])
        
        return modelo, df, vectorizer, matriz_tfidf_completa
    except FileNotFoundError:
        st.error("Erro: Os arquivos 'final_model.pkl' ou 'filmes_com_clusters.csv' não foram encontrados.")
        st.error("Por favor, certifique-se de que eles estão na mesma pasta que o arquivo 'app.py'.")
        return None, None, None, None

# Carrega os dados na inicialização do app
modelo, df, vectorizer, matriz_tfidf_completa = carregar_modelo_e_dados()

# --- INTERFACE DO USUÁRIO E LÓGICA DE RECOMENDAÇÃO ---

# Título da aplicação
st.title('🎬 Sistema de Recomendação de Filmes')

# --- BOTÃO DE NOVA INDICAÇÃO (MELHORIA APLICADA AQUI) ---
if st.button("🔄 Gerar Novas Indicações"):
    # Limpa o estado da sessão para forçar a geração de novas sugestões
    if 'sugestoes' in st.session_state:
        del st.session_state['sugestoes']
    if 'escolha_feita' in st.session_state:
        del st.session_state['escolha_feita']
    st.rerun() # Força a re-execução do script para atualizar a interface imediatamente

st.write("""
    Bem-vindo ao nosso sistema de recomendação! Para começar, leia as 4 sinopses abaixo e escolha a que mais lhe agrada.
    Com base na sua escolha, recomendaremos 5 filmes com temas similares.
""")

# Apenas continua se os arquivos foram carregados com sucesso
if df is not None:
    # Gerenciamento de estado da sessão para manter as sugestões e a escolha do usuário
    if 'sugestoes' not in st.session_state:
        clusters_disponiveis = df['Cluster'].unique()
        clusters_escolhidos = random.sample(list(clusters_disponiveis), 4)
        
        sugestoes_idx = []
        for cluster in clusters_escolhidos:
            filme_sugerido = df[df['Cluster'] == cluster].sample(1).index[0]
            sugestoes_idx.append(filme_sugerido)
            
        st.session_state['sugestoes'] = sugestoes_idx
        st.session_state['escolha_feita'] = None

    # Exibe as 4 sugestões em colunas
    st.subheader("Escolha uma das sinopses abaixo:")
    col1, col2, col3, col4 = st.columns(4)
    colunas = [col1, col2, col3, col4]

    for i, idx in enumerate(st.session_state['sugestoes']):
        with colunas[i]:
            with st.container(border=True):
                filme = df.loc[idx]
                st.write(f"**Sugestão #{i+1}**")
                st.write(f"_{filme['synopsis'][:250]}..._")
                if st.button("Escolher este", key=f"btn_{idx}"):
                    st.session_state['escolha_feita'] = idx
                    st.rerun() # Re-executa para mostrar as recomendações imediatamente

    # --- LÓGICA APÓS A ESCOLHA DO USUÁRIO ---
    if st.session_state.get('escolha_feita') is not None:
        st.divider()
        
        filme_escolhido_idx = st.session_state['escolha_feita']
        filme_escolhido = df.loc[filme_escolhido_idx]
        cluster_escolhido = filme_escolhido['Cluster']

        st.success(f"Você escolheu uma sinopse do filme: **{filme_escolhido['title']}**")
        st.write(f"Buscando os 5 filmes mais similares dentro do **Cluster {cluster_escolhido}**...")

        df_cluster = df[df['Cluster'] == cluster_escolhido].drop(index=filme_escolhido_idx)
        vetor_filme_escolhido = matriz_tfidf_completa[filme_escolhido_idx]
        indices_cluster = df_cluster.index
        matriz_tfidf_cluster = matriz_tfidf_completa[indices_cluster]
        
        similaridades = cosine_similarity(vetor_filme_escolhido, matriz_tfidf_cluster).flatten()
        
        # Garante que pegaremos no máximo 5 filmes, ou menos se o cluster for pequeno
        num_recomendacoes = min(5, len(similaridades))
        indices_top_similaridade = np.argsort(similaridades)[-num_recomendacoes:]
        
        indices_finais_no_df = df_cluster.index[indices_top_similaridade]
        df_recomendacoes = df.loc[indices_finais_no_df].sort_values(by="rating", ascending=False)
        
        st.subheader(f"Aqui estão os {len(df_recomendacoes)} filmes que recomendamos para você:")
        
        for _, filme_rec in df_recomendacoes.iterrows():
            with st.container(border=True):
                col_img, col_info = st.columns([1, 4])
                with col_img:
                    st.markdown(f"### 🍿")
                with col_info:
                    st.markdown(f"**{filme_rec['title']} ({filme_rec['year']})**")
                    st.markdown(f"**Nota IMDb:** {filme_rec['rating']} ⭐")
                    st.markdown(f"**Gêneros:** {filme_rec['genre']}")