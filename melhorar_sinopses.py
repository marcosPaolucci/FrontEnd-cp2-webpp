import pandas as pd
import time
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from openai import OpenAI
from dotenv import load_dotenv
import os
import csv

# Certifique-se de que as dependências NLTK estão disponíveis
try:
    stopwords.words('portuguese')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('stemmers/rslp')
except LookupError:
    nltk.download('rslp')

# Função de limpeza igual ao cluster.py
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

# Função para chamar o LLM da OpenAI para melhorar a sinopse
def melhorar_sinopse(sinopse, client):
    def prompt_com_titulo(sinopse, titulo):
        return (
            f"Melhore e amplie a seguinte sinopse do filme '{titulo}', tornando-a mais detalhada, envolvente e informativa, sem perder o contexto original. "
            "Não invente fatos ou personagens, apenas complemente com informações conhecidas sobre o filme. "
            "Limite a sinopse melhorada a 1 parágrafo de no máximo 800 caracteres. "
            "Sinopse original: "
            f"{sinopse}"
        )

    # Recebe o título do filme do contexto do loop
    import inspect
    frame = inspect.currentframe().f_back
    titulo = frame.f_locals.get('row', {}).get('title', '')
    prompt = prompt_com_titulo(sinopse, titulo)
    try:
        result = client.responses.create(
            model="gpt-5-mini",
            input=prompt,
            reasoning={"effort": "low"},
            text={"verbosity": "medium"}
        )
        return result.output_text.strip()
    except Exception as e:
        print(f"Erro ao chamar o modelo OpenAI: {e}")
        return sinopse

# Caminho do CSV original e do novo CSV de teste
CSV_PATH = 'filmes_com_clusters.csv'
CSV_TESTE_PATH = 'filmes_com_clusters_teste.csv'

# Inicializa o cliente OpenAI
load_dotenv()  # Carrega variáveis de ambiente do arquivo .env, se existir
API_KEY = os.getenv("OPENAI_API_KEY")

# Inicializa o cliente OpenAI
client = OpenAI(api_key=API_KEY)

df = pd.read_csv(CSV_PATH).copy()

sinopses_melhoradas = []
sinopses_melhoradas_limpa = []

for idx, row in df.iterrows():
    sinopse_original = row['synopsis']
    print(f"Melhorando sinopse do filme: {row['title']} ({row['year']})")
    sinopse_melhorada = melhorar_sinopse(sinopse_original, client)
    sinopses_melhoradas.append(sinopse_melhorada)
    sinopse_melhorada_limpa = limpar_texto(sinopse_melhorada)
    sinopses_melhoradas_limpa.append(sinopse_melhorada_limpa)
    time.sleep(1)  # Evita rate limit

# Adiciona as novas colunas ao DataFrame
df['sinopse_melhorada'] = sinopses_melhoradas
df['sinopse_melhorada_limpa'] = sinopses_melhoradas_limpa

# Salva o novo CSV principal com todas as colunas
CSV_RESULTADO = 'filmes_com_cluster_melhorados.csv'
df.to_csv(CSV_RESULTADO, index=False, encoding='utf-8-sig')
print(f"Arquivo gerado: {CSV_RESULTADO}")
