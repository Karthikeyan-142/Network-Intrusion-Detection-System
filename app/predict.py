import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# =====================================================
# 1. LOAD TRAINED MODEL
# =====================================================
model = load_model("model/cnn_lstm_model.h5")
print("Model loaded successfully")

# =====================================================
# 2. LOAD DATASET (same CSV)
# =====================================================
data_path = "dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(data_path)

# Fix column name spaces
df.columns = df.columns.str.strip()

# =====================================================
# 3. CLEAN DATA (same as training)
# =====================================================
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.drop_duplicates()

# =====================================================
# 4. REMOVE LABEL COLUMN SAFELY
# =====================================================
possible_labels = ["Label", "label", "Attack", "Class"]

for col in possible_labels:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)
        print(f"Removed label column: {col}")
        break

# =====================================================
# 5. REMOVE NON-NUMERIC COLUMNS
# =====================================================
non_numeric = df.select_dtypes(include=["object"]).columns
df.drop(non_numeric, axis=1, inplace=True)

# =====================================================
# 6. TAKE SINGLE SAMPLE
# =====================================================
sample = df.iloc[0].values.reshape(1, -1)

# =====================================================
# 7. SCALE (NOTE: demo-safe)
# =====================================================
scaler = StandardScaler()
sample = scaler.fit_transform(sample)

# Reshape for CNN + LSTM
sample = sample.reshape(sample.shape[0], sample.shape[1], 1)

# =====================================================
# 8. PREDICT
# =====================================================
prediction = model.predict(sample)
result = np.argmax(prediction)

print("\nPrediction Result:")
if result == 0:
    print("✅ BENIGN (Normal Traffic)")
else:
    print("🚨 ATTACK (Malicious Traffic)")
