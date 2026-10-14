"""Gráficos del análisis. Cada función devuelve la figura para poder mostrarla o guardarla."""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from src.estilo import AZUL, NARANJA, TEXTO_SEC, SERIES, aplicar_estilo, miles

aplicar_estilo()
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def ingresos_mensuales_por_categoria(ventas: pd.DataFrame) -> plt.Figure:
    m = (ventas.assign(mes=ventas["fecha"].dt.to_period("M").dt.to_timestamp())
         .pivot_table(index="mes", columns="categoria", values="ingreso", aggfunc="sum"))
    m = m[m.sum().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, col in enumerate(m.columns):
        ax.plot(m.index, m[col], color=SERIES[i], label=col)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(miles))
    ax.set_title("Ingresos mensuales por categoría (ARS)")
    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=8.5)
    return fig


def efecto_dia_semana(ventas: pd.DataFrame) -> plt.Figure:
    d = ventas.groupby([ventas["fecha"], ventas["fecha"].dt.dayofweek.rename("dow")])["unidades"].sum()
    prom = d.groupby("dow").mean()
    indice = 100 * prom / prom.mean()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    barras = ax.bar(DIAS, indice, color=[NARANJA if v == indice.max() else AZUL for v in indice], width=0.6)
    ax.axhline(100, color=TEXTO_SEC, lw=1, ls="--")
    ax.bar_label(barras, fmt="%.0f", padding=3, fontsize=9)
    ax.set_ylim(0, indice.max() * 1.15)
    ax.set_title("Índice de unidades vendidas por día de la semana (promedio = 100)")
    return fig


def estacionalidad_bebidas(ventas: pd.DataFrame) -> plt.Figure:
    b = ventas[ventas["categoria"].str.startswith("Bebida")]
    s = (b.groupby([b["fecha"].dt.month.rename("mes"), "categoria"])["unidades"].sum().unstack())
    s = 100 * s / s.mean()
    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for col, color in zip(["Bebida caliente", "Bebida fría"], [NARANJA, AZUL]):
        ax.plot(meses, s[col], color=color, marker="o", ms=5, label=col)
    ax.axhline(100, color=TEXTO_SEC, lw=1, ls="--")
    ax.set_title("Estacionalidad anual: bebidas frías vs calientes (índice, media anual = 100)")
    ax.legend(loc="upper center")
    return fig


def efecto_promociones(ventas: pd.DataFrame) -> plt.Figure:
    v = ventas.copy()
    v["dow"] = v["fecha"].dt.dayofweek
    v["mes"] = v["fecha"].dt.to_period("M")
    claves = ["producto", "sucursal", "mes", "dow"]
    # comparar contra días sin promo del mismo producto, sucursal, mes y día de semana
    base = v[v["en_promocion"] == 0].groupby(claves)["unidades"].mean().rename("base")
    p = v[v["en_promocion"] == 1].join(base, on=claves).dropna(subset=["base"])
    uplift = (p.groupby("categoria")["unidades"].sum() / p.groupby("categoria")["base"].sum() - 1) * 100
    uplift = uplift.sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    barras = ax.barh(uplift.index, uplift.values, color=AZUL, height=0.55)
    ax.bar_label(barras, fmt="+%.0f%%", padding=4, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, uplift.max() * 1.25)
    ax.set_title("Aumento de unidades en días de promoción (precio -15%)")
    return fig


def pronostico_vs_real(test: pd.DataFrame, sucursal="Centro", producto="Medialuna") -> plt.Figure:
    t = test[(test["sucursal"] == sucursal) & (test["producto"] == producto)].set_index("fecha")
    t = t.loc[t.index[:56]]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(t.index, t["unidades"], color=AZUL, label="Venta real")
    ax.plot(t.index, t["pronostico"], color=NARANJA, label="Pronóstico del modelo")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.set_title(f"{producto} — sucursal {sucursal}: pronóstico a 1 día vs venta real (8 semanas de test)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    return fig


def frontera_merma(c_manual: pd.DataFrame, c_modelo: pd.DataFrame, resumen: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for c, color, lbl in [(c_manual, NARANJA, "Política manual (promedio + colchón)"),
                          (c_modelo, AZUL, "Política con modelo (pronóstico + stock de seguridad)")]:
        c = c[(c["nivel_servicio_%"] >= 85) & (c["nivel_servicio_%"] <= 99)]
        ax.plot(c["nivel_servicio_%"], c["merma_unidades"], color=color, label=lbl, marker="o", ms=4)
    s = resumen["nivel_servicio_%"]
    ax.vlines(s, resumen["merma_modelo_unidades"], resumen["merma_manual_unidades"], color=TEXTO_SEC, lw=1.2, ls=":")
    ax.annotate(f"−{resumen['reduccion_merma_%']:.0f}% de merma\ncon el mismo nivel de servicio ({s:.0f}%)",
                (s, (resumen["merma_modelo_unidades"] + resumen["merma_manual_unidades"]) / 2),
                xytext=(s + 2.2, resumen["merma_modelo_unidades"] * 0.72), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=TEXTO_SEC, lw=1))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(miles))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_xlabel("Nivel de servicio (% de la demanda atendida)")
    ax.set_ylabel("Unidades descartadas (sep–dic)")
    ax.set_title("Merma vs nivel de servicio: cada punto es una regla de producción distinta")
    ax.legend(loc="upper left", fontsize=8.5)
    return fig


def importancia_variables(modelo, test: pd.DataFrame, features: list[str], n=4000, semilla=42) -> plt.Figure:
    from sklearn.inspection import permutation_importance
    muestra = test.sample(min(n, len(test)), random_state=semilla)
    r = permutation_importance(modelo, muestra[features], muestra["unidades"], n_repeats=3,
                               random_state=semilla, scoring="neg_mean_absolute_error")
    imp = pd.Series(r.importances_mean, index=features).sort_values().tail(8)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(imp.index, imp.values, color=AZUL, height=0.55)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Aumento del error (MAE) al desordenar la variable")
    ax.set_title("Variables más importantes del modelo")
    return fig
