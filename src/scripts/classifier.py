import pandas as pd
import numpy as np
from src.core.text_cluster_silhouette import TextClusterSilhouette
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score

path = "datasets/llm_predict/zero_shot/CSTR/gpt-oss-20b.csv"
model_name = "knn"

df = pd.read_csv(path)
df = df[df['predicted_class'] != "other"]

# Lista para armazenar os resultados de F1-score
results = []

# Testar diferentes percentuais de dados rotulados
for percentage in np.arange(0.1, 0.9, 0.1):
    print(f"Testando com {percentage*100}% de dados rotulados...")

    # Realizar a clusterização e filtragem com base na porcentagem
    text_cluster_silhouette = TextClusterSilhouette(df)
    X_train, y_train_real, y_train_llm, X_test, y_test_real, y_test_llm = text_cluster_silhouette.cluster_and_filter(percentage)

    # Treinamento do modelo KNN com y_train_real
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train_real)

    # Avaliar o modelo com y_test_real
    y_pred_real = knn.predict(X_test)
    f1_real = f1_score(y_test_real, y_pred_real, average='weighted')

    # Treinamento do modelo KNN com y_train_llm
    knn.fit(X_train, y_train_llm)

    # Avaliar o modelo com y_test_real
    y_pred_llm = knn.predict(X_test)
    f1_llm_knn = f1_score(y_test_real, y_pred_llm, average='weighted')

    # Avaliar desempenho usando y_test_llm (LLM "bruto")
    f1_llm = f1_score(y_test_real, y_test_llm, average='weighted')

    # Salvar os resultados em uma lista
    results.append({
        'percentage': percentage,
        'f1_real': f1_real,
        'f1_llm_knn': f1_llm_knn,
        'f1_llm': f1_llm
    })

# Criar um DataFrame com os resultados
df_results = pd.DataFrame(results)

# Salvar o DataFrame em um arquivo CSV
df_results.to_csv('results_knn_f1_score.csv', index=False)

# Exibir os resultados
print("Resultados salvos no arquivo 'results_knn_f1_score.csv'")
print(df_results)
