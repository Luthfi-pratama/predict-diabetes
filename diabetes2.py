import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import dump, load
import time  # Modul untuk menghitung waktu komputasi

# Load data
df = pd.read_csv('diabetes.csv')

# Features and labels
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 1. KNN ---
start_time_knn = time.time()  # Mulai waktu komputasi
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
end_time_knn = time.time()  # Selesai waktu komputasi
time_knn = end_time_knn - start_time_knn

# --- 2. Naive Bayes ---
start_time_nb = time.time()  # Mulai waktu komputasi
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
end_time_nb = time.time()  # Selesai waktu komputasi
time_nb = end_time_nb - start_time_nb

# --- 3. K-Means Clustering ---
start_time_kmeans = time.time()  # Mulai waktu komputasi
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_scaled)
kmeans_labels = kmeans.predict(X_test_scaled)
end_time_kmeans = time.time()  # Selesai waktu komputasi
time_kmeans = end_time_kmeans - start_time_kmeans

# Convert K-Means labels to match `y_test`
kmeans_labels_adjusted = np.where(kmeans_labels == 0, 1, 0)
silhouette_avg = silhouette_score(X_test_scaled, kmeans_labels)

# Simpan model
dump(nb, 'naive_bayes_model.joblib')

# --- Output ---
print("="*50)
print("Machine Learning Model Accuracy and Time Results".center(50))
print("="*50)
print(f"KNN Model Accuracy:           {accuracy_knn*100:.2f}%")
print(f"KNN Computation Time:         {time_knn:.4f} seconds")
print(f"Naive Bayes Model Accuracy:   {accuracy_nb*100:.2f}%")
print(f"Naive Bayes Computation Time: {time_nb:.4f} seconds")
print(f"K-Means Silhouette Score:     {silhouette_avg:.2f}")
print(f"K-Means Computation Time:     {time_kmeans:.4f} seconds")
print("="*50)
