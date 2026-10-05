# importar pacotes
import glob
import json
import os

import joblib
import kagglehub
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

OUTPUT_DIR = os.environ.get("MODEL_DIR", "artefatos")
FEATURES = ["Open", "High", "Low", "Close", "Volume"]

# Download do dataset
path = kagglehub.dataset_download("varpit94/ethereum-data")
csv_path = glob.glob(os.path.join(path, "*.csv"))[0]
print("Dataset:", csv_path)

# Carregar e limpar
df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date")
df = df.dropna(subset=FEATURES).reset_index(drop=True)

# Alvo: fechamento do dia seguinte (t+1), a partir das features do dia atual
df["target"] = df["Close"].shift(-1)
df = df.dropna(subset=["target"])

X = df[FEATURES]
y = df["target"]

# Série temporal: split cronológico (shuffle=False) para evitar vazamento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Treino
model = LinearRegression()
model.fit(X_train, y_train)

# Avaliação
pred = model.predict(X_test)
metrics = {
    "mae": float(mean_absolute_error(y_test, pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    "r2": float(r2_score(y_test, pred)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}
print("Métricas:", metrics)

# Exportar artefatos
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(model, os.path.join(OUTPUT_DIR, "model.joblib"))
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump({"features": FEATURES, "target": "Close(t+1)", "metrics": metrics}, f, indent=2)
print(f"Modelo salvo em {OUTPUT_DIR}/model.joblib")
