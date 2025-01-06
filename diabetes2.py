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
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# --- 2. Naive Bayes ---
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# --- 3. K-Means Clustering ---
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_scaled)
kmeans_labels = kmeans.predict(X_test_scaled)

# Convert K-Means labels to match `y_test`
kmeans_labels_adjusted = np.where(kmeans_labels == 0, 1, 0)
silhouette_avg = silhouette_score(X_test_scaled, kmeans_labels)


# Simpan model
dump(nb, 'naive_bayes_model.joblib')

# --- Output ---
print("="*50)
print("Machine Learning Model Accuracy Results".center(50))
print("="*50)
print(f"KNN Model Accuracy:           {accuracy_knn*100:.2f}%")
print(f"Naive Bayes Model Accuracy:   {accuracy_nb*100:.2f}%")
print(f"K-Means Silhouette Score:     {silhouette_avg:.2f}")
print("="*50)
