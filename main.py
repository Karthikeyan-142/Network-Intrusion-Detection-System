import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ===============================
# CREATE WEBSITE FOLDER
# ===============================
os.makedirs("website/static", exist_ok=True)

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv("dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()

# ===============================
# LABEL
# ===============================
label_col = "Label"
df[label_col] = df[label_col].apply(
    lambda x: 0 if str(x).upper() == "BENIGN" else 1
)

X = df.drop(label_col, axis=1)
y = df[label_col]
X = X.select_dtypes(include=[np.number])

# ===============================
# SCALE + RESHAPE
# ===============================
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(X.shape[0], X.shape[1], 1)

# ===============================
# SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_train = to_categorical(y_train, 2)

# ===============================
# MODEL
# ===============================
model = Sequential([
    Conv1D(64, 3, activation="relu", input_shape=(X.shape[1], 1)),
    MaxPooling1D(2),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# TRAIN
# ===============================
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=256,
    validation_split=0.1
)

# ===============================
# PREDICT
# ===============================
y_pred = np.argmax(model.predict(X_test), axis=1)

# ===============================
# METRICS
# ===============================
accuracy = accuracy_score(y_test, y_pred)

precision_micro = precision_score(y_test, y_pred, average="micro")
precision_macro = precision_score(y_test, y_pred, average="macro")
precision_weighted = precision_score(y_test, y_pred, average="weighted")

recall_micro = recall_score(y_test, y_pred, average="micro")
recall_macro = recall_score(y_test, y_pred, average="macro")
recall_weighted = recall_score(y_test, y_pred, average="weighted")

f1_micro = f1_score(y_test, y_pred, average="micro")
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

# ===============================
# SAVE GRAPHS
# ===============================
plt.figure()
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.legend()
plt.title("Accuracy vs Epoch")
plt.savefig("website/static/accuracy.png")
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.legend()
plt.title("Loss vs Epoch")
plt.savefig("website/static/loss.png")
plt.close()

disp = ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred),
    display_labels=["Benign", "Attack"]
)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("website/static/confusion_matrix.png")
plt.close()

# ===============================
# METRICS TABLE (COLUMN FORMAT)
# ===============================
metrics_html = f"""
<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" href="style.css">
</head>
<body>

<h2 style="text-align:center">Model Evaluation Metrics</h2>

<table class="metrics-table">
<tr>
    <th>Accuracy</th>
    <th>Precision (Micro)</th>
    <th>Precision (Macro)</th>
    <th>Precision (Weighted)</th>
    <th>Recall (Micro)</th>
    <th>Recall (Macro)</th>
    <th>Recall (Weighted)</th>
    <th>F1 (Micro)</th>
    <th>F1 (Macro)</th>
    <th>F1 (Weighted)</th>
</tr>
<tr>
    <td>{accuracy:.4f}</td>
    <td>{precision_micro:.4f}</td>
    <td>{precision_macro:.4f}</td>
    <td>{precision_weighted:.4f}</td>
    <td>{recall_micro:.4f}</td>
    <td>{recall_macro:.4f}</td>
    <td>{recall_weighted:.4f}</td>
    <td>{f1_micro:.4f}</td>
    <td>{f1_macro:.4f}</td>
    <td>{f1_weighted:.4f}</td>
</tr>
</table>

</body>
</html>
"""

with open("website/metrics.html", "w") as f:
    f.write(metrics_html)

print("DONE: graphs + column-metrics table generated")
