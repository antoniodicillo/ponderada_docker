# importar pacotes
import json
import os

import joblib
import kagglehub
import numpy as np
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = os.environ.get("MODEL_DIR", "artefatos")


def build_features(close: pd.Series) -> pd.DataFrame:
    """Features do dia t a partir dos fechamentos até t."""
    lr = np.log(close).diff()  # retorno logarítmico diário
    X = pd.DataFrame({"ret_1": lr})
    for n in (3, 7, 14):  # quanto o preço se moveu nos últimos n dias
        X[f"ret_{n}"] = np.log(close / close.shift(n))
    X["vol_7"] = lr.rolling(7).std()  # desvio padrão dos retornos
    X["vol_30"] = lr.rolling(30).std()
    X["ma30_dist"] = close / close.rolling(30).mean() - 1  # distância da média de 30 dias
    return X


# Download e carga do dataset (preços diários)
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "prasoonkottarathil/ethereum-historical-dataset",
    "ETH_day.csv",
    pandas_kwargs={"parse_dates": ["Date"]},
)
df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)

# Alvo: retorno logarítmico do dia seguinte. Preço previsto = Close(t) * exp(retorno)
feats = build_features(df["Close"])
y = np.log(df["Close"].shift(-1) / df["Close"])
valid = feats.notna().all(axis=1) & y.notna()
X, y, close = feats[valid], y[valid], df["Close"][valid]

# Série temporal: split cronológico (shuffle=False) para evitar vazamento
X_train, X_test, y_train, y_test, _, close_test = train_test_split(
    X, y, close, test_size=0.2, shuffle=False
)

# Baseline ingênuo: preço de amanhã = preço de hoje (retorno previsto = 0)
price_test = close_test * np.exp(y_test)
naive_mae = float(mean_absolute_error(price_test, close_test))
print(f"Baseline (repetir o dia anterior): MAE = {naive_mae:.2f}")

# Modelos bem regularizados: as previsões ficam perto de 0 (perto do ingênuo)
models = {
    f"Ridge(alpha={a})": make_pipeline(StandardScaler(), Ridge(alpha=a))
    for a in (10, 100, 1000, 10000)
}
models["GradientBoosting(raso)"] = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.02,
    max_depth=2,
    min_samples_leaf=30,
    subsample=0.7,
    random_state=42,
)

# Escolha do modelo por validação cruzada temporal só no treino (teste fica intocado)
cv_splits = TimeSeriesSplit(5)
zero_cv = float(y_train.abs().mean())  # MAE do retorno se previssemos sempre 0
print(f"CV MAE do retorno prevendo 0 (ingênuo): {zero_cv:.5f}")

results = {}
for name, m in models.items():
    cv = -cross_val_score(m, X_train, y_train, cv=cv_splits, scoring="neg_mean_absolute_error").mean()
    m.fit(X_train, y_train)
    pred_ret = m.predict(X_test)
    pred_price = close_test * np.exp(pred_ret)
    results[name] = {
        "cv_mae_return": float(cv),
        "test_mae": float(mean_absolute_error(price_test, pred_price)),
        "test_rmse": float(np.sqrt(mean_squared_error(price_test, pred_price))),
        "mean_abs_pred_return": float(np.abs(pred_ret).mean()),
        # acerto de direção (sobe/desce), que é onde dá para ganhar do ingênuo
        "direction_acc": float((np.sign(pred_ret) == np.sign(y_test)).mean()),
    }
    print(name, results[name])

best = min(results, key=lambda n: results[n]["cv_mae_return"])
print("Melhor modelo (validação cruzada):", best)

# Modelo final: treina com todos os dados
final_model = models[best].fit(X, y)

# Exportar artefatos
os.makedirs(OUTPUT_DIR, exist_ok=True)
# O model.joblib é autossuficiente: leva o modelo e a tabela de features de cada dia,
# então a inferência só carrega o arquivo e procura a data pedida.
data = feats.dropna().assign(Close=df["Close"])
data.index = df["Date"][data.index]
joblib.dump(
    {"model": final_model, "features": list(X.columns), "data": data},
    os.path.join(OUTPUT_DIR, "model.joblib"),
)
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(
        {
            "model": best,
            "features": list(X.columns),
            "target": "Close(t+1) = Close(t) * exp(retorno log)",
            "metrics": {
                "naive_mae": naive_mae,
                "naive_cv_mae_return": zero_cv,
                "models": results,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
            },
        },
        f,
        indent=2,
    )
print(f"Modelo salvo em {OUTPUT_DIR}/model.joblib")
