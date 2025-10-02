import pandas as pd
from collections import Counter

def analisar_clusters():
    """
    Carrega o arquivo de filmes com clusters e imprime uma análise detalhada
    de cada cluster, incluindo gêneros, palavras-chave e exemplos.
    """
    try:
        df = pd.read_csv('filmes_com_clusters.csv')
    except FileNotFoundError:
        print("ERRO: O arquivo 'filmes_com_clusters.csv' não foi encontrado.")
        print("Por favor, certifique-se de que este script está na mesma pasta que o seu arquivo CSV.")
        return

    # Função para extrair palavras-chave da sinopse limpa
    def get_keywords(text_series):
        all_words = ' '.join(text_series.dropna()).split()
        if not all_words:
            return []
        return Counter(all_words).most_common(10)

    # Função para extrair e contar gêneros
    def get_genres(genre_series):
        all_genres = []
        for g_list in genre_series.dropna():
            # Trata gêneros separados por ', '
            all_genres.extend(g_list.split(', '))
        if not all_genres:
            return []
        return Counter(all_genres).most_common(5)

    # Analisar cada cluster
    cluster_ids = sorted(df['Cluster'].unique())
    print("--- Análise Detalhada e Correta dos Clusters ---\n")

    for cluster_id in cluster_ids:
        print(f"=================================================")
        print(f" Cluster {cluster_id}")
        print(f"=================================================\n")

        cluster_df = df[df['Cluster'] == cluster_id]
        
        print(f"Total de Filmes: {len(cluster_df)}\n")

        # Análise de Gêneros
        top_genres = get_genres(cluster_df['genre'])
        print("Gêneros Mais Comuns:")
        for genre, count in top_genres:
            print(f"- {genre}: {count} filmes")
        
        print("\nPalavras-Chave Mais Comuns na Sinopse:")
        # Análise de Palavras-chave
        keywords = get_keywords(cluster_df['sinopse_limpa'])
        for word, count in keywords:
            print(f"- {word}: {count} vezes")

        print("\nExemplos de Filmes no Cluster:")
        # Amostra de Títulos
        # Usamos .head() para uma amostra consistente em vez de aleatória
        sample_titles = cluster_df['title'].head(min(len(cluster_df), 7)).tolist()
        for title in sample_titles:
            print(f"- {title}")
        print("\n")

# Executa a função de análise
if __name__ == "__main__":
    analisar_clusters()