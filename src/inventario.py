"""
Simulación de política de producción para productos perecederos (vida útil de 1 día).

Lo que no se vende al cierre se descarta (merma). Lo que falta es venta perdida (quiebre).
Se comparan dos políticas a IGUAL nivel de servicio para que la comparación sea justa:

- Manual:  producir = promedio de los últimos 4 mismos días de la semana x (1 + colchón)
- Modelo:  producir = pronóstico + z x desvío esperado del error para ese producto y volumen

Supuesto: en los datos sintéticos la demanda no está censurada (vendido = demandado).
"""
import numpy as np
import pandas as pd


def evaluar(produccion: np.ndarray, demanda: np.ndarray, costo: np.ndarray) -> dict:
    produccion = np.round(np.maximum(produccion, 0))
    vendido = np.minimum(produccion, demanda)
    merma = produccion - vendido
    return {
        "nivel_servicio_%": float(100 * vendido.sum() / demanda.sum()),
        "merma_unidades": float(merma.sum()),
        "merma_%_producido": float(100 * merma.sum() / produccion.sum()),
        "merma_costo_ARS": float((merma * costo).sum()),
    }


def calibrar_error(validacion: pd.DataFrame) -> pd.DataFrame:
    """Mide el error del pronóstico por producto en un período de validación.

    El error de demanda crece aproximadamente con la raíz del volumen, así que se guarda
    el desvío normalizado a un volumen de referencia (la media del producto) y luego se
    reescala para cada sucursal y día.
    """
    ref = validacion.groupby("producto")["pronostico"].mean().rename("volumen_ref")
    v = validacion.join(ref, on="producto")
    escala = np.sqrt(v["volumen_ref"] / np.maximum(v["pronostico"], 1))
    sigma = ((v["unidades"] - v["pronostico"]) * escala).groupby(v["producto"]).std().rename("sigma_ref")
    return pd.concat([sigma, ref], axis=1)


def desvio_esperado(df: pd.DataFrame, error: pd.DataFrame) -> np.ndarray:
    e = df[["producto", "pronostico"]].join(error, on="producto")
    return (e["sigma_ref"] * np.sqrt(np.maximum(e["pronostico"], 1) / e["volumen_ref"])).to_numpy()


def curva_manual(df: pd.DataFrame, colchones=np.linspace(0, 0.6, 31)) -> pd.DataFrame:
    filas = []
    for c in colchones:
        prod = df["media_mismo_dia_4sem"].to_numpy() * (1 + c)
        filas.append({"parametro": c, **evaluar(prod, df["unidades"].to_numpy(), df["costo_unitario"].to_numpy())})
    return pd.DataFrame(filas)


def curva_modelo(df: pd.DataFrame, error: pd.DataFrame, zs=np.linspace(-0.5, 2.5, 31)) -> pd.DataFrame:
    sigma = desvio_esperado(df, error)
    filas = []
    for z in zs:
        prod = df["pronostico"].to_numpy() + z * sigma
        filas.append({"parametro": z, **evaluar(prod, df["unidades"].to_numpy(), df["costo_unitario"].to_numpy())})
    return pd.DataFrame(filas)


def a_nivel_de_servicio(curva: pd.DataFrame, servicio_objetivo: float, columna: str) -> float:
    """Valor de `columna` de una política al nivel de servicio pedido (interpolando su curva)."""
    c = curva.sort_values("nivel_servicio_%")
    return float(np.interp(servicio_objetivo, c["nivel_servicio_%"], c[columna]))
