import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from pycaret.clustering import setup, create_model, assign_model, save_model
import re
import unicodedata

# 3. Carregamento dos Dados
DATA_FILE = 'imdb_top_250_filmes_generos_filtrados.csv'
df = pd.read_csv(DATA_FILE)
print(f"Arquivo '{DATA_FILE}' carregado com sucesso.")

# 4. Pré-processamento de Texto
def limpar_texto(texto):
    if isinstance(texto, str):
        texto = texto.lower()
        texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
        texto = re.sub(r'[^\w\s]', '', texto)
        palavras = texto.split()
        stop_words = set(stopwords.words('portuguese'))
        palavras_sem_stopwords = [p for p in palavras if p not in stop_words]
        stemmer_pt = nltk.stem.RSLPStemmer()
        palavras_stemizadas = [stemmer_pt.stem(p) for p in palavras_sem_stopwords]
        return ' '.join(palavras_stemizadas)
    return ""

try:
    stopwords.words('portuguese')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('stemmers/rslp')
except LookupError:
    nltk.download('rslp')

print("Realizando limpeza e pré-processamento das sinopses...")
df['sinopse_limpa'] = df['synopsis'].apply(limpar_texto)
print("Limpeza concluída.")

# 5. Vetorização do Texto com TF-IDF
print("Vetorizando o texto com TF-IDF (com melhorias)...")
vectorizer = TfidfVectorizer(min_df=0.03, max_df=0.3, max_features=500)
matriz_tfidf = vectorizer.fit_transform(df['sinopse_limpa'])
tfidf_df = pd.DataFrame(matriz_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

# 6. Preparação do Dataset para o PyCaret
print("Preparando o DataFrame final para o PyCaret...")
# --- MELHORIA APLICADA AQUI ---
# 1. Separar os múltiplos gêneros em colunas dummy
# A função str.get_dummies faz isso perfeitamente, usando a vírgula como separador
generos_dummies = df['genre'].str.get_dummies(sep=', ')

# 2. Juntar tudo: os dummies de gênero, os vetores da sinopse e as colunas que vamos ignorar
df_para_pycaret = pd.concat([
    df[['title', 'rating']], 
    generos_dummies, 
    tfidf_df
], axis=1)

print("DataFrame preparado com a separação correta dos gêneros.")


# 7. Treinamento e Seleção de Modelo com PyCaret
print("\n--- Iniciando o processo com PyCaret ---")
# Agora, não precisamos mais declarar 'categorical_features', pois já transformamos os gêneros em colunas numéricas (0s e 1s)
exp_cluster = setup(data=df_para_pycaret,
                    ignore_features=['title', 'rating'],
                    session_id=123,
                    verbose=True)

print("Criando o modelo de clusterização K-Means com 6 clusters...")
model = create_model('kmeans', num_clusters=6)
print("\n--- Modelo Criado ---")
print(model)

# 8. Atribuição dos Clusters e Salvamento
print("\nAtribuindo os clusters aos filmes...")
df_com_clusters_temp = assign_model(model)
df_final = df.copy()
df_final['Cluster'] = df_com_clusters_temp['Cluster']

save_model(model, 'final_model')
df_final.to_csv('filmes_com_clusters.csv', index=False)
print("Modelo salvo como 'final_model.pkl' e dados com clusters salvos em 'filmes_com_clusters.csv'.")

# 9. Simulação da Lógica do WebApp (Método 1)
print("\n\n--- SIMULAÇÃO DA LÓGICA DO WEBAPP (MÉTODO 1) ---")

def obter_opcoes_iniciais(df_completo, n_opcoes=3):
    clusters_unicos = df_completo['Cluster'].unique()
    n_opcoes = min(n_opcoes, len(clusters_unicos))
    if n_opcoes == 0: return pd.DataFrame()
    clusters_selecionados = np.random.choice(clusters_unicos, n_opcoes, replace=False)

    opcoes = []
    for cluster in clusters_selecionados:
        filme_de_exemplo = df_completo[df_completo['Cluster'] == cluster].sample(1)
        opcoes.append(filme_de_exemplo.iloc[0])

    return pd.DataFrame(opcoes)

def recomendar_filmes(df_completo, id_filme_escolhido):
    try:
        filme_escolhido = df_completo.loc[id_filme_escolhido]
        cluster_alvo = filme_escolhido['Cluster']
        titulo_escolhido = filme_escolhido['title']
    except KeyError:
        return f"Erro: Filme com ID {id_filme_escolhido} não encontrado."

    print(f"\nUsuário escolheu a sinopse correspondente ao cluster: '{cluster_alvo}' (Filme base: '{titulo_escolhido}')")

    filmes_do_cluster = df_completo[df_completo['Cluster'] == cluster_alvo]
    filmes_do_cluster = filmes_do_cluster.drop(index=id_filme_escolhido, errors='ignore')

    recomendacoes = filmes_do_cluster.sort_values(by='rating', ascending=False).head(5)
    return recomendacoes[['title', 'rating', 'genre']]

# -- Executando a Simulação --
print("\nPasso 1: Gerando 3 opções de sinopses de clusters diferentes para o usuário...")
opcoes_iniciais = obter_opcoes_iniciais(df_final, n_opcoes=3)

if not opcoes_iniciais.empty:
    for index, row in opcoes_iniciais.iterrows():
        sinopse_original = row['synopsis']
        print(f"\nOpção ID {index}:")
        print(f"SINOPSE: \"{sinopse_original[:200]}...\"")

    filme_escolhido_id = opcoes_iniciais.index[0]
    print(f"\n\nPasso 2: O usuário (simulado) escolheu a sinopse da Opção ID {filme_escolhido_id}.")

    print("\nPasso 3: Gerando as recomendações com base na escolha...")
    recomendacoes_finais = recomendar_filmes(df_final, filme_escolhido_id)

    print("\n--- FILMES RECOMENDADOS PARA VOCÊ ---")
    print(recomendacoes_finais)
else:
    print("Não foi possível gerar opções iniciais.")