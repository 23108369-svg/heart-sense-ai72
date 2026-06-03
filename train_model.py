# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np

# FIX FOR MATPLOTLIB/TKINTER ERROR
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

import joblib

# =========================
# 2. LOAD DATASET
# =========================
df = pd.read_csv("dataset/raw_merged_heart_dataset.csv")

# =========================
# 3. DATA CLEANING
# =========================
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.drop_duplicates()

print("Shape after cleaning:", df.shape)

# =========================
# 4. EDA (SAVE GRAPHS)
# =========================

# Target Distribution Plot
sns.countplot(x="target", data=df)
plt.title("Target Distribution")
plt.savefig("target_distribution.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

print("Graphs saved successfully!")

# =========================
# 5. SPLIT DATA
# =========================
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# 6. SCALING
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("Scaler saved successfully!")

# =========================
# 7. BUILD ANN MODEL
# =========================
model = Sequential()

# Input Layer
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Hidden Layer
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Hidden Layer
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

# =========================
# 8. COMPILE MODEL
# =========================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully!")

# =========================
# 9. EARLY STOPPING
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# =========================
# 10. TRAIN MODEL
# =========================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# 11. EVALUATION
# =========================
loss, acc = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", acc)

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.close()

print("Confusion matrix saved successfully!")

# =========================
# 12. SAVE MODEL
# =========================
model.save("ann_model.h5")

print("Model saved successfully!")
print("Training completed successfully!")