# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import time

# ===============================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ===============================================================
# Carrega a base de dados Iris (substitua o caminho abaixo pelo caminho correto)
data = pd.read_csv("Iris.csv")

# Visualize as 5 primeiras linhas para confirmar
print(data.head())

# Separa as features e o target (variável que queremos prever)
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = data[features].values
y = data['Species'].values

# Divisão entre conjunto de treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===============================================================
# 2. IMPLEMENTAÇÃO DO KNN DO ZERO (CLASSIFICADOR CUSTOMIZADO)
# ===============================================================

# Função para calcular a distância Euclidiana entre dois pontos
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Função que implementa o algoritmo KNN customizado para classificar um ponto de teste
def knn_classify(X_train, y_train, test_point, k):
    distances = []
    for i in range(len(X_train)):
        distance = euclidean_distance(test_point, X_train[i])
        distances.append((distance, y_train[i]))
    # Ordena as distâncias (do menor para o maior) e seleciona os k vizinhos mais próximos
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    # Realiza a votação pelos rótulos dos vizinhos
    votes = [label for _, label in neighbors]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

# Função para avaliar o classificador customizado
def evaluate_knn(X_train, y_train, X_test, y_test, k):
    predictions = []
    for test_point in X_test:
        pred = knn_classify(X_train, y_train, test_point, k)
        predictions.append(pred)
    cm = confusion_matrix(y_test, predictions, labels=np.unique(y))
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='weighted')
    rec = recall_score(y_test, predictions, average='weighted')
    return predictions, cm, acc, prec, rec

# Função aprimorada para plotar a matriz de confusão com anotações
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    classes = np.unique(y)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Adiciona os valores de cada célula
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > cm.max()/2. else 'black')
    
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.tight_layout()
    plt.show()

# Avaliação para diferentes valores de k
k_values = [1, 3, 5, 7]
results_custom = {}

print("===== KNN Customizado (do zero) =====")
for k in k_values:
    start_time = time.time()
    predictions, cm, acc, prec, rec = evaluate_knn(X_train, y_train, X_test, y_test, k)
    elapsed_time = time.time() - start_time
    results_custom[k] = {'confusion_matrix': cm,
                         'accuracy': acc,
                         'precision': prec,
                         'recall': rec,
                         'time': elapsed_time}
    print(f"\nK = {k}:")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Revocação: {rec:.4f}")
    print(f"Tempo de execução: {elapsed_time:.4f} s")
    print("Matriz de Confusão:")
    print(cm)

# Plot da matriz de confusão para o k com melhor acurácia no classificador customizado
best_k_custom = max(results_custom, key=lambda k: results_custom[k]['accuracy'])
cm_best_custom = results_custom[best_k_custom]['confusion_matrix']
plot_confusion_matrix(cm_best_custom, f'Matriz de Confusão (Customizado) para k = {best_k_custom}')

# ===============================================================
# 3. IMPLEMENTAÇÃO COM SKLEARN (CLASSIFICADOR COM LIBRARY)
# ===============================================================
from sklearn.neighbors import KNeighborsClassifier

results_sklearn = {}
print("\n===== KNN com Scikit-Learn =====")
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    start_time = time.time()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    elapsed_time = time.time() - start_time
    cm = confusion_matrix(y_test, predictions, labels=np.unique(y))
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='weighted')
    rec = recall_score(y_test, predictions, average='weighted')
    results_sklearn[k] = {'confusion_matrix': cm,
                          'accuracy': acc,
                          'precision': prec,
                          'recall': rec,
                          'time': elapsed_time}
    print(f"\n(KNN Sklearn) K = {k}:")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Revocação: {rec:.4f}")
    print(f"Tempo de execução: {elapsed_time:.4f} s")
    print("Matriz de Confusão:")
    print(cm)

# Plot da matriz de confusão para o k com melhor acurácia no classificador do sklearn
best_k_sklearn = max(results_sklearn, key=lambda k: results_sklearn[k]['accuracy'])
cm_best_sklearn = results_sklearn[best_k_sklearn]['confusion_matrix']
plot_confusion_matrix(cm_best_sklearn, f'Matriz de Confusão (Sklearn) para k = {best_k_sklearn}')

# ===============================================================
# 4. ANÁLISE DE DESEMPENHO
# ===============================================================
print("\n===== Análise Comparativa =====")
for k in k_values:
    print(f"\nPara k = {k}:")
    print("Classificador Customizado:")
    print(f"  Acurácia: {results_custom[k]['accuracy']:.4f}")
    print(f"  Precisão: {results_custom[k]['precision']:.4f}")
    print(f"  Revocação: {results_custom[k]['recall']:.4f}")
    print(f"  Tempo de execução: {results_custom[k]['time']:.4f} s")
    
    print("Classificador Sklearn:")
    print(f"  Acurácia: {results_sklearn[k]['accuracy']:.4f}")
    print(f"  Precisão: {results_sklearn[k]['precision']:.4f}")
    print(f"  Revocação: {results_sklearn[k]['recall']:.4f}")
    print(f"  Tempo de execução: {results_sklearn[k]['time']:.4f} s")
    
# ===============================================================
# 5. VISUALIZAÇÃO ADICIONAL DE MÉTRICAS
# ===============================================================
# Extra: Comparação de métricas e tempo de execução entre as implementações Customizada e Sklearn

# Preparando os dados para os gráficos
k_vals = k_values  # [1, 3, 5, 7]
custom_acc = [results_custom[k]['accuracy'] for k in k_vals]
sklearn_acc = [results_sklearn[k]['accuracy'] for k in k_vals]

custom_prec = [results_custom[k]['precision'] for k in k_vals]
sklearn_prec = [results_sklearn[k]['precision'] for k in k_vals]

custom_rec = [results_custom[k]['recall'] for k in k_vals]
sklearn_rec = [results_sklearn[k]['recall'] for k in k_vals]

custom_time = [results_custom[k]['time'] for k in k_vals]
sklearn_time = [results_sklearn[k]['time'] for k in k_vals]

# Função para criar gráfico comparativo de barras
def plot_comparison(metric_custom, metric_sklearn, metric_name, ylabel):
    x = np.arange(len(k_vals))
    width = 0.35
    plt.figure(figsize=(8,6))
    plt.bar(x - width/2, metric_custom, width, label='Customizado')
    plt.bar(x + width/2, metric_sklearn, width, label='Sklearn')
    plt.xlabel('Valores de k')
    plt.ylabel(ylabel)
    plt.title(f'Comparação de {metric_name} por k')
    plt.xticks(x, k_vals)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    # Anota os valores acima de cada barra
    for i in x:
        plt.text(i - width/2, metric_custom[i] + 0.01, f'{metric_custom[i]:.2f}', ha='center')
        plt.text(i + width/2, metric_sklearn[i] + 0.01, f'{metric_sklearn[i]:.2f}', ha='center')
    plt.tight_layout()
    plt.show()

# Plot de Acurácia
plot_comparison(custom_acc, sklearn_acc, 'Acurácia', 'Acurácia')

# Plot de Precisão
plot_comparison(custom_prec, sklearn_prec, 'Precisão', 'Precisão')

# Plot de Revocação
plot_comparison(custom_rec, sklearn_rec, 'Revocação', 'Revocação')

# Plot de Tempo de Execução
plot_comparison(custom_time, sklearn_time, 'Tempo de Execução', 'Tempo (s)')

print("\nResumo Final:")
print(f"Melhor k para Customizado: {best_k_custom} com Acurácia {results_custom[best_k_custom]['accuracy']:.4f}")
print(f"Melhor k para Sklearn: {best_k_sklearn} com Acurácia {results_sklearn[best_k_sklearn]['accuracy']:.4f}")