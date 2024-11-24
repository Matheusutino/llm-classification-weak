import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sentence_transformers import SentenceTransformer

class TextClusterSilhouette:
    def __init__(self, df, model_name='all-MiniLM-L6-v2'):
        """
        Inicializa a classe com um DataFrame e carrega o modelo SentenceTransformer.
        
        Parâmetros:
        df (pd.DataFrame): DataFrame contendo uma coluna 'text' com textos e uma coluna 'class' com rótulos.
        model_name (str): Nome do modelo SentenceTransformer para gerar os embeddings.
        """
        self.df = df
        self.embeddings = None
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self):
        """
        Gera embeddings usando SentenceTransformer para cada texto na coluna 'text' do DataFrame.
        """
        texts = self.df['text'].tolist()
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        print("Embeddings gerados com sucesso.")
    
    def cluster_and_filter(self, n, random_state = 42):
        """
        Realiza o clustering com KMeans, calcula as pontuações de silhueta e retorna as instâncias 
        com as maiores e menores pontuações.
        
        Parâmetros:
        n (int): Número de instâncias a retornar para as menores e maiores pontuações de silhueta.
        
        Retorno:
        pd.DataFrame: DataFrame filtrado com as instâncias de acordo com as pontuações de silhueta.
        """
        # Obtendo o número de clusters a partir da coluna 'class'
        n_clusters = self.df['class'].nunique()

        # Aplicando o KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(self.embeddings)

        # Calculando a medida de silhueta
        silhouette_scores = silhouette_samples(self.embeddings, labels)

        # Encontrando os n índices das maiores e menores pontuações de silhueta
        lowest_silhouette_indices = np.argsort(silhouette_scores)[:int(n/2)]
        highest_silhouette_indices = np.argsort(silhouette_scores)[-int(n/2):]

        # Filtrando o DataFrame original com base nos índices selecionados
        filtered_df = pd.concat([self.df.iloc[lowest_silhouette_indices], self.df.iloc[highest_silhouette_indices]])
        filtered_df['silhouette_score'] = np.concatenate([silhouette_scores[lowest_silhouette_indices], silhouette_scores[highest_silhouette_indices]])

        return filtered_df
