"""
Pronóstico de demanda diaria por sucursal y producto.

Horizonte: 1 día (la producción de panadería y cocina se decide la tarde anterior),
por eso se usan rezagos desde t-1.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = [
    "sucursal_cod", "producto_cod", "dia_semana", "mes", "dia_anio_sin", "dia_anio_cos",
    "en_promocion", "es_feriado", "lag_1", "lag_7", "lag_14", "media_7", "media_28",
    "media_mismo_dia_4sem",
]
CATEGORICAS = ["sucursal_cod", "producto_cod", "dia_semana", "mes"]


def crear_features(ventas: pd.DataFrame) -> pd.DataFrame:
    df = ventas.sort_values(["sucursal", "producto", "fecha"]).copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["sucursal_cod"] = df["sucursal"].astype("category").cat.codes
    df["producto_cod"] = df["producto"].astype("category").cat.codes
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["mes"] = df["fecha"].dt.month
    doy = df["fecha"].dt.dayofyear
    df["dia_anio_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["dia_anio_cos"] = np.cos(2 * np.pi * doy / 365.25)

    g = df.groupby(["sucursal", "producto"])["unidades"]
    df["lag_1"] = g.shift(1)
    df["lag_7"] = g.shift(7)
    df["lag_14"] = g.shift(14)
    df["media_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["media_28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
    # Heurística "manual" que usaba el equipo: promedio de los últimos 4 mismos días de la semana
    df["media_mismo_dia_4sem"] = (
        df.groupby(["sucursal", "producto", "dia_semana"])["unidades"]
        .transform(lambda s: s.shift(1).rolling(4).mean())
    )
    return df.dropna(subset=["lag_14", "media_28", "media_mismo_dia_4sem"]).reset_index(drop=True)


def entrenar(train: pd.DataFrame, semilla: int = 42) -> HistGradientBoostingRegressor:
    modelo = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.06,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=[FEATURES.index(c) for c in CATEGORICAS],
        random_state=semilla,
    )
    modelo.fit(train[FEATURES], train["unidades"])
    return modelo


def metricas(real: pd.Series, pred: pd.Series) -> dict:
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    error = pred - real
    return {
        "MAE": float(np.mean(np.abs(error))),
        "WAPE_%": float(100 * np.sum(np.abs(error)) / np.sum(real)),
        "sesgo_%": float(100 * np.sum(error) / np.sum(real)),
    }
