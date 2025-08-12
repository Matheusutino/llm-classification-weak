import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

class TextClusterSilhouette:
    def __init__(self, df, model_name='all-MiniLM-L12-v2'):
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
    
    def cluster_and_filter(self, percentage, random_state=42):
        """
        Realiza o clustering com KMeans, calcula as pontuações de silhueta e retorna os conjuntos de treino e teste.
        
        Parâmetros:
        percentage (float): Porcentagem de instâncias a selecionar com base nas pontuações de silhueta.
        
        Retorno:
        X_train, y_train, X_test, y_test
        """
        # Gerando os embeddings antes de realizar o clustering
        self.generate_embeddings()

        # Obtendo o número de clusters a partir da coluna 'class'
        n_clusters = self.df['class'].nunique()

        # Aplicando o KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(self.embeddings)

        # Calculando a medida de silhueta
        silhouette_scores = silhouette_samples(self.embeddings, labels)

        # Aplicar módulo para garantir que os valores sejam todos positivos
        silhouette_scores_abs = np.abs(silhouette_scores)

        # Determinando a quantidade de amostras a selecionar
        n = int(len(self.df) * percentage)

        # Ordenar os índices dos valores mais próximos de 0, com base no valor absoluto de silhouette_scores
        closest_to_zero_indices = np.argsort(silhouette_scores_abs)[:int(n)]

        # Criando DataFrame filtrado
        filtered_df = self.df.iloc[closest_to_zero_indices]
        filtered_df['silhouette_score'] = silhouette_scores[closest_to_zero_indices]
        
        # Selecionando as instâncias de treino (baseadas nas instâncias de melhor silhueta)
        X_train_embeddings = self.embeddings[closest_to_zero_indices]
        y_train_real = filtered_df['class'].tolist()
        y_train_llm = filtered_df['predicted_class'].tolist()

        # Selecionando as instâncias de teste (demais instâncias)
        test_indices = np.setdiff1d(np.arange(len(self.df)), closest_to_zero_indices)
        X_test_embeddings = self.embeddings[test_indices]
        y_test_real = self.df.iloc[test_indices]['class'].tolist()
        y_test_llm = self.df.iloc[test_indices]['predicted_class'].tolist()

        return X_train_embeddings, y_train_real, y_train_llm, X_test_embeddings, y_test_real, y_test_llm

