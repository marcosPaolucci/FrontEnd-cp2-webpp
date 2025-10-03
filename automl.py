import pandas as pd
from pycaret.clustering import *
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import unicodedata
import os

def run_automl_clustering():
    """
    Executa um pipeline de AutoML para encontrar o melhor modelo de clusterização,
    testando diferentes algoritros e números de clusters.
    """
    # --- 1. CARREGAMENTO E PRÉ-PROCESSAMENTO ---
    
    print(">>> 1. Carregando e processando os dados...")
    
    df = pd.read_csv('imdb_top_250_filmes_generos_filtrados.csv')

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
        nltk.download('rslp')

    df['sinopse_limpa'] = df['synopsis'].apply(limpar_texto)
    
    vectorizer = TfidfVectorizer(min_df=0.03, max_df=0.8, max_features=500)
    matriz_tfidf = vectorizer.fit_transform(df['sinopse_limpa'])
    tfidf_df = pd.DataFrame(matriz_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

    generos_dummies = df['genre'].str.get_dummies(sep=', ')

    df_para_pycaret = pd.concat([df[['title', 'rating']], generos_dummies, tfidf_df], axis=1)
    
    print(">>> Dados processados com sucesso.\n")

    # --- 2. SETUP DO PYCARET ---
    
    print(">>> 2. Configurando o ambiente PyCaret...")
    exp = setup(data=df_para_pycaret,
                ignore_features=['title', 'rating'],
                session_id=123,
                verbose=False)
    
    print(">>> Ambiente configurado.\n")
    
    # --- 3. BUSCA PELO MELHOR MODELO (AutoML) ---
    
    print(">>> 3. Iniciando busca AutoML...\n")
    
    # Parte A: Encontrar o melhor número de clusters (k) para o K-Means
    print("--- Parte A: Testando diferentes 'k' para o K-Means (4 a 15) ---")
    best_kmeans = None
    best_kmeans_score = -1
    best_k = 0

    for k in range(4, 16):
        kmeans_model = create_model('kmeans', num_clusters=k, verbose=False)
        metrics = pull()
        silhouette_score = metrics.loc[0, 'Silhouette']
        print(f"K-Means com k={k}: Silhouette Score = {silhouette_score:.4f}")
        
        if silhouette_score > best_kmeans_score:
            best_kmeans_score = silhouette_score
            best_kmeans = kmeans_model
            best_k = k
            
    print(f"\nMelhor K-Means encontrado: k={best_k} com Silhouette Score de {best_kmeans_score:.4f}\n")

    # --- CORREÇÃO APLICADA AQUI ---
    # Parte B: Comparar com outros algoritmos de clusterização
    print("--- Parte B: Comparando com outros algoritmos ---")
    
    # Lista de outros algoritmos para testar
    models_to_compare = ['hclust', 'birch', 'agglomerative']
    best_other_model = None
    best_other_score = -1
    best_other_model_name = ""

    for model_name in models_to_compare:
        try:
            model = create_model(model_name, verbose=False)
            metrics = pull()
            silhouette_score = metrics.loc[0, 'Silhouette']
            print(f"Modelo '{model_name}': Silhouette Score = {silhouette_score:.4f}")
            
            if silhouette_score > best_other_score:
                best_other_score = silhouette_score
                best_other_model = model
                best_other_model_name = model_name
        except Exception as e:
            print(f"Não foi possível treinar o modelo '{model_name}'. Erro: {e}")

    print(f"\nMelhor 'outro' modelo encontrado: {best_other_model_name} com Silhouette Score de {best_other_score:.4f}\n")
    
    # --- 4. SELEÇÃO E FINALIZAÇÃO DO MELHOR MODELO ---
    
    print(">>> 4. Selecionando o modelo final...")
    
    final_model = None
    if best_kmeans_score >= best_other_score:
        print(f"O CAMPEÃO É: K-Means com k={best_k} (Score: {best_kmeans_score:.4f})")
        final_model = best_kmeans
    else:
        print(f"O CAMPEÃO É: {best_other_model_name} (Score: {best_other_score:.4f})")
        final_model = best_other_model
        
    # --- 5. SALVANDO OS RESULTADOS ---
    
    print("\n>>> 5. Gerando e salvando os resultados finais...")
    
    df_com_clusters_temp = assign_model(final_model)
    df_final = df.copy()
    df_final['Cluster'] = df_com_clusters_temp['Cluster']

    save_model(final_model, 'automl_best_model')
    df_final.to_csv('automl_filmes_com_clusters.csv', index=False)
    
    print("\nProcesso concluído!")
    print("Modelo salvo como 'automl_best_model.pkl'")
    print("CSV com clusters salvo como 'automl_filmes_com_clusters.csv'")

# Executa a função principal
if __name__ == "__main__":
    run_automl_clustering()