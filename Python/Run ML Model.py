# Import wymaganych bibliotek

import pandas as pd
import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import Image, display

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
import shutil

# Pobierz najnowszą wersję zbioru danych
path = kagglehub.dataset_download("alexandrepetit881234/fake-bills")

# Wczytaj dane do ramki danych (DataFrame) przy użyciu pandas
df = pd.read_csv(os.path.join(path, os.listdir(path)[0]), header = 0, delimiter = ";")

# Load the CSV file with semicolon separator
#file_path = '/export/viya/homes/szymon.czuszek@edu.uekat.pl/casuser/ML/fake_bills.csv'
#df = pd.read_csv(file_path, sep=';')

print(path)

# Display the first few rows
print(df.head())

# Wyświetl podstawowe statystyki opisowe dla danych
print(df.describe())

# Sprawdź, czy w zbiorze danych występują wartości NaN (brakujące dane)
print(df.isna().sum())

# Wyświetl wiersze, w których występują brakujące dane w kolumnie "margin_low"
print(df[df["margin_low"].isna() == True])

# Oblicz średnią wartość dla kolumny "margin_low"
mean_value = df["margin_low"].mean()

# Wypełnij brakujące wartości w kolumnie "margin_low" średnią wartością
df["margin_low"] = df["margin_low"].fillna(mean_value)

# Zakoduj kolumnę docelową "is_genuine" jako zmienną binarną
df['is_genuine'] = df['is_genuine'].astype(int)

# Zdefiniuj cechy (X) i zmienną docelową (y)
X = df[['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']]
y = df['is_genuine']

# Podziel dane na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Normalizacja danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Rozkład zmiennej docelowej
plt.figure(figsize=(6,4))
df['is_genuine'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Rozkład zmiennej docelowej (is_genuine)')
plt.xticks(ticks=[0,1], labels=['Fałszywy', 'Prawdziwy'], rotation=0)
plt.ylabel('Liczba próbek')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

plt.savefig("Rozklad.png")
shutil.copy("Rozklad.png", "/export/viya/homes/szymon.czuszek@edu.uekat.pl/casuser/ML/")

# Macierz korelacji
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Macierz korelacji cech')
plt.tight_layout()
plt.show()

plt.savefig("Korelacja.png")
shutil.copy("Korelacja.png", "/export/viya/homes/szymon.czuszek@edu.uekat.pl/casuser/ML/")

import shutil

# Diagonal
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='diagonal', hue='is_genuine', kde=True, palette=['red', 'green'], bins=30)
plt.title('Rozkład cechy "diagonal" względem klasy')
plt.tight_layout()
plt.savefig('diagonal.png')
shutil.copy("diagonal.png", "/export/viya/homes/szymon.czuszek@edu.uekat.pl/casuser/ML/")
plt.show()

# Margin low
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='margin_low', hue='is_genuine', kde=True, palette=['red', 'green'], bins=30)
plt.title('Rozkład cechy "margin_low" względem klasy')
plt.tight_layout()
plt.savefig('margin_low.png')
shutil.copy("margin_low.png", "/export/viya/homes/szymon.czuszek@edu.uekat.pl/casuser/ML/")
plt.show()

# Length
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='length', hue='is_genuine', kde=True, palette=['red', 'green'], bins=30)
plt.title('Rozkład cechy "length" względem klasy')
plt.tight_layout()
plt.savefig('length.png')
shutil.copy("length.png", "/export/viya/homes/szymon.czuszek@edu.uekat.pl/casuser/ML/")
plt.show()

# Wizualizacja dokładności dla różnych k
accuracies = []
k_values = range(1, 11)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Rysowanie wykresu
plt.figure(figsize = (10, 6))
plt.plot(k_values, accuracies, marker = 'o', linestyle = '-', color = 'b')
plt.title("Dokładność w zależności od liczby sąsiadów (k)")
plt.xlabel("Liczba sąsiadów (k)")
plt.ylabel("Dokładność")
plt.grid(alpha = 0.5)
plt.tight_layout()
plt.show()

plt.savefig("knn_accuracy.png")
shutil.copy("knn_accuracy.png", "/export/viya/homes/szymon.czuszek@edu.uekat.pl/casuser/ML/")

# Najlepsza dokładność
print(max(accuracies))

# Musimy dodać 1 bo indexy w Python liczy się od zera
k = accuracies.index(max(accuracies)) + 1
print(k)

### K-Nearest Neighbors (KNN) ###
print("\nAlgorytm K-Nearest Neighbors (KNN):")

# Inicjalizacja modelu KNN
model_knn = KNeighborsClassifier(n_neighbors = k)
model_knn.fit(X_train, y_train)

# Predykcja dla zbioru testowego (KNN)
y_pred_knn = model_knn.predict(X_test)
y_pred_proba_knn = model_knn.predict_proba(X_test)[:, 1]

# Ocena modelu (KNN)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Dokładność (KNN, k={k}): {accuracy_knn:.2f}")
print("\nRaport klasyfikacji (KNN):\n", classification_report(y_test, y_pred_knn))
print("\nMacierz pomyłek (KNN):\n", confusion_matrix(y_test, y_pred_knn))

# Krzywa ROC (KNN)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_proba_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

### Regresja logistyczna ###
print("Regresja logistyczna:")
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

# Predykcja dla zbioru testowego (Regresja logistyczna)
y_pred_lr = model_lr.predict(X_test)
y_pred_proba_lr = model_lr.predict_proba(X_test)[:, 1]

# Ocena modelu (Regresja logistyczna)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Dokładność (Regresja logistyczna): {accuracy_lr:.2f}")
print("\nRaport klasyfikacji (Regresja logistyczna):\n", classification_report(y_test, y_pred_lr))
print("\nMacierz pomyłek (Regresja logistyczna):\n", confusion_matrix(y_test, y_pred_lr))

# Krzywa ROC (Regresja logistyczna)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

### Wizualizacja krzywych ROC dla obu modeli ###
plt.figure(figsize = (8, 6))

# Regresja logistyczna
plt.plot(fpr_lr,
         tpr_lr,
         color = 'blue',
         lw = 2,
         label = f'Regresja Logistyczna (AUC = {roc_auc_lr:.2f})'
        )

# KNN
plt.plot(fpr_knn,
         tpr_knn,
         color = 'green',
         lw = 2,
         linestyle = '-.',
         label = f'KNN (k={k}, AUC = {roc_auc_knn:.2f})'
        )

# Linia zgadywania
plt.plot([0, 1],
         [0, 1],
         color = 'red',
         linestyle = '--',
         label = 'Zgadywanie'
        )

# Dostosowanie wykresu
plt.xlabel('Wskaźnik fałszywych alarmów \n(False Positive Rate)',
           fontsize = 14,
           fontweight = 'bold'
          )

plt.ylabel('Wskaźnik prawdziwych alarmów \n(True Positive Rate)',
           fontsize = 14,
           fontweight = 'bold'
          )

plt.title('Porównanie krzywych ROC',
          fontsize = 16,
          fontweight = 'bold'
         )

plt.legend(loc = "lower right",
           fontsize = 12
          )

plt.grid(True,
         linestyle = '--',
         alpha = 0.6
        )

plt.tight_layout()

plt.show()

plt.savefig("ROC.png")
shutil.copy("ROC.png", "/export/viya/homes/szymon.czuszek@edu.uekat.pl/casuser/ML/")

# Send the modified DataFrame back to SAS
SAS.df2sd(df, _output1)