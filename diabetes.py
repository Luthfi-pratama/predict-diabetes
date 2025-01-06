import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# --- 3. Perceptron ---
perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
perceptron.fit(X_train_scaled, y_train)
y_pred_perceptron = perceptron.predict(X_test_scaled)
accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)

# --- 4. Backpropagation (Neural Network) ---
model = Sequential([
    Dense(16, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=50, batch_size=8, verbose=0)

loss, accuracy_nn = model.evaluate(X_test_scaled, y_test, verbose=0)


# --- Simpan Model ---
model.save('best_model.h5')  # Menyimpan model neural network

# --- Output ---
print("="*50)
print("Machine Learning Model Accuracy Results".center(50))
print("="*50)
print(f"KNN Model Accuracy:           {accuracy_knn*100:.2f}%")
print(f"Naive Bayes Model Accuracy:   {accuracy_nb*100:.2f}%")
print(f"Perceptron Model Accuracy:    {accuracy_perceptron*100:.2f}%")
print(f"Neural Network Accuracy:      {accuracy_nn*100:.2f}%")
print("="*50)
