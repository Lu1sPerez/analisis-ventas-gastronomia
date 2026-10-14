"""Orquesta el flujo completo: datos -> features -> modelo -> evaluación -> simulación de merma."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import generar_datos, inventario, modelo

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"

INICIO_VALIDACION = "2024-07-01"
INICIO_TEST = "2024-09-01"
COLCHON_MANUAL = 0.20  # el clásico "producí 20% más por las dudas"


@dataclass
class Resultado:
    ventas: pd.DataFrame
    catalogo: pd.DataFrame
    test: pd.DataFrame
    metricas: pd.DataFrame
    curva_manual: pd.DataFrame
    curva_modelo: pd.DataFrame
    resumen: dict
    modelo: object


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (RAW / "ventas_diarias.csv").exists():
        generar_datos.main()
    ventas = pd.read_csv(RAW / "ventas_diarias.csv", parse_dates=["fecha"])
    catalogo = pd.read_csv(RAW / "catalogo_productos.csv")
    return ventas, catalogo


def ejecutar() -> Resultado:
    ventas, catalogo = cargar_datos()
    df = modelo.crear_features(ventas)

    # 1) Modelo de calibración: aprende hasta junio, mide su error en julio-agosto
    m_cal = modelo.entrenar(df[df["fecha"] < INICIO_VALIDACION])
    val = df[(df["fecha"] >= INICIO_VALIDACION) & (df["fecha"] < INICIO_TEST)].copy()
    val["pronostico"] = m_cal.predict(val[modelo.FEATURES])

    # 2) Modelo final: aprende hasta agosto, se evalúa en sep-dic (datos nunca vistos)
    m = modelo.entrenar(df[df["fecha"] < INICIO_TEST])
    test = df[df["fecha"] >= INICIO_TEST].copy()
    test["pronostico"] = m.predict(test[modelo.FEATURES])

    metricas = pd.DataFrame({
        "Promedio 4 mismos días (manual)": modelo.metricas(test["unidades"], test["media_mismo_dia_4sem"]),
        "Mismo día semana pasada": modelo.metricas(test["unidades"], test["lag_7"]),
        "Gradient Boosting (modelo)": modelo.metricas(test["unidades"], test["pronostico"]),
    }).T.round(2)

    # 3) Simulación de producción para perecederos de 1 día
    perecederos = catalogo.loc[catalogo["vida_util_dias"] == 1, "producto"]
    error = inventario.calibrar_error(val[val["producto"].isin(perecederos)])
    t = test[test["producto"].isin(perecederos)]
    c_manual = inventario.curva_manual(t)
    c_modelo = inventario.curva_modelo(t, error)

    ref = inventario.evaluar(
        t["media_mismo_dia_4sem"].to_numpy() * (1 + COLCHON_MANUAL),
        t["unidades"].to_numpy(), t["costo_unitario"].to_numpy(),
    )
    servicio = ref["nivel_servicio_%"]
    merma_modelo = inventario.a_nivel_de_servicio(c_modelo, servicio, "merma_unidades")
    costo_modelo = inventario.a_nivel_de_servicio(c_modelo, servicio, "merma_costo_ARS")
    meses_test = t["fecha"].dt.to_period("M").nunique()

    resumen = {
        "periodo_test": f"{INICIO_TEST} a {t['fecha'].max():%Y-%m-%d}",
        "productos_simulados": list(perecederos),
        "nivel_servicio_%": round(servicio, 1),
        "merma_manual_unidades": round(ref["merma_unidades"]),
        "merma_modelo_unidades": round(merma_modelo),
        "reduccion_merma_%": round(100 * (1 - merma_modelo / ref["merma_unidades"]), 1),
        "ahorro_costo_mensual_ARS": round((ref["merma_costo_ARS"] - costo_modelo) / meses_test),
        "mejora_WAPE_pp": round(
            metricas.loc["Promedio 4 mismos días (manual)", "WAPE_%"]
            - metricas.loc["Gradient Boosting (modelo)", "WAPE_%"], 1),
    }
    return Resultado(ventas, catalogo, test, metricas, c_manual, c_modelo, resumen, m)
