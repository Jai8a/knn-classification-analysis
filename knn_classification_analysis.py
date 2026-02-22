import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import numpy as np

train_data = pd.read_csv("data/data3_train.csv", header=None)
test_data = pd.read_csv("data/data3_test.csv", header=None)

X_train = train_data.iloc[:, :-1].values  
y_train = train_data.iloc[:, -1].values   
X_test = test_data.iloc[:, :-1].values    
y_test = test_data.iloc[:, -1].values     

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

feature_names = ["Długość działki kielicha [cm]","Szerokość działki kielicha [cm]","Długość płatka [cm]","Szerokość płatka [cm]"]
class_mapping = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

def evaluate_knn(X_train, y_train, X_test, y_test, max_k=15):
    accuracies = []
    confusion_matrices = {}

    for k in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        
        for i in range(len(X_test)):
            neighbors_indices = knn.kneighbors(X_test[i].reshape(1, -1), n_neighbors=k)[1][0]
            neighbors_labels = y_train[neighbors_indices]

            if np.sum(neighbors_labels == y_pred[i]) == 0:
                y_pred[i] = neighbors_labels[0]

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc * 100)
        confusion_matrices[k] = confusion_matrix(y_test, y_pred)

    return accuracies, confusion_matrices

def plot_accuracies(accuracies, feature_set, all_features=False, suffix=""):
    plt.figure(figsize=(12, 6))
    k_values = range(1, 16)
    plt.bar(k_values, accuracies, edgecolor='black', width=0.7)
    
    if all_features:
        title = 'Dokładność klasyfikacji dla różnych wartości k (Wszystkie cechy)'
    else:
        feature_str = '\n'.join(feature_set)
        title = f'Dokładność klasyfikacji dla różnych wartości k\n({feature_str})'
    
    plt.title(title, fontsize="20")
    plt.xlabel('Liczba sąsiadów (k)', fontsize="18")
    plt.ylabel('Dokładność (%)', fontsize="18")
    plt.xticks(k_values)
    plt.ylim(50, 105)  
    plt.yticks(range(50, 101, 10),fontsize="14")
    plt.xticks(fontsize="14")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(matrix, k, feature_names, all_features=False):
    df_cm = pd.DataFrame(matrix,index=[class_mapping[i] for i in range(len(class_mapping))],columns=[class_mapping[i] for i in range(len(class_mapping))])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', edgecolor='black', linewidths=1, linecolor='black')
    
    if all_features:
        title = f"Macierz pomyłek dla k={k}\n(Wszystkie cechy)"
    else:
        feature_str = "\n".join(feature_names)  
        title = f"Macierz pomyłek dla k={k}\n({feature_str})"
    
    plt.title(title,fontsize="16")
    plt.xlabel('Przewidywana klasa',fontsize="16")
    plt.ylabel('Rzeczywista klasa',fontsize="16")
    plt.xticks(fontsize="14")
    plt.yticks(fontsize="14")
    plt.tight_layout()
    plt.show()
    plt.xticks

accuracies, confusion_matrices = evaluate_knn(X_train, y_train, X_test, y_test)
plot_accuracies(accuracies, feature_names, all_features=True, suffix="(wszystkie cechy)")

best_k = accuracies.index(max(accuracies)) + 1
print(f"Najlepsza wartość k (wszystkie cechy): {best_k}")
plot_confusion_matrix(confusion_matrices[best_k], best_k, feature_names, all_features=True)

feature_combinations = list(combinations(range(len(feature_names)), 2))

for combo in feature_combinations:
    X_train_subset = X_train[:, combo]
    X_test_subset = X_test[:, combo]

    accuracies, confusion_matrices = evaluate_knn(X_train_subset, y_train, X_test_subset, y_test)
    combo_names = [feature_names[i] for i in combo]
    plot_accuracies(accuracies, combo_names, suffix=f"({', '.join(combo_names)})")

    best_k = accuracies.index(max(accuracies)) + 1
    print(f"Najlepsza wartość k ({', '.join(combo_names)}): {best_k}")
    plot_confusion_matrix(confusion_matrices[best_k], best_k, combo_names) 