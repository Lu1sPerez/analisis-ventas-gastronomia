"""
Generador de datos sintéticos de ventas para una cadena gastronómica (café / deli).

Los datos son SINTÉTICOS pero modelan patrones reales del rubro:
- 4 sucursales con distinto volumen
- estacionalidad semanal (fines de semana) y anual (bebidas frías en verano, calientes en invierno)
- feriados nacionales de Argentina
- promociones con aumento de demanda
- productos perecederos con vida útil de 1 a 3 días

Uso:
    python src/generar_datos.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 42
FECHA_INICIO = "2023-01-01"
FECHA_FIN = "2024-12-31"

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "data" / "raw"

SUCURSALES = {
    # nombre: factor de volumen
    "Centro": 1.30,
    "City Bell": 1.00,
    "Gonnet": 0.85,
    "Los Hornos": 0.70,
}

# producto: (categoría, demanda base diaria, precio ARS, costo ARS, vida útil en días, sensibilidad al calor)
# sensibilidad al calor > 0: vende más en verano; < 0: vende más en invierno
PRODUCTOS = {
    "Café con leche":      ("Bebida caliente", 95, 3200, 900, 0, -0.25),
    "Té en hebras":        ("Bebida caliente", 40, 2900, 600, 0, -0.30),
    "Chocolate caliente":  ("Bebida caliente", 18, 3600, 1100, 0, -0.60),
    "Limonada":            ("Bebida fría", 22, 3400, 800, 1, 0.70),
    "Té helado":           ("Bebida fría", 16, 3300, 700, 1, 0.75),
    "Medialuna":           ("Panadería", 160, 900, 280, 1, -0.05),
    "Scon":                ("Panadería", 45, 1100, 350, 2, -0.10),
    "Budín":               ("Pastelería", 14, 2600, 900, 3, 0.00),
    "Cheesecake":          ("Pastelería", 12, 4800, 1700, 2, 0.10),
    "Tostado jamón y queso": ("Salado", 38, 5200, 1900, 1, -0.05),
    "Wrap de pollo":       ("Salado", 20, 6400, 2500, 1, 0.20),
    "Ensalada bowl":       ("Salado", 15, 7200, 2800, 1, 0.45),
}

# Feriados nacionales de Argentina (inamovibles + trasladables principales) 2023-2024
FERIADOS = pd.to_datetime([
    "2023-01-01", "2023-02-20", "2023-02-21", "2023-03-24", "2023-04-02", "2023-04-07",
    "2023-05-01", "2023-05-25", "2023-06-17", "2023-06-20", "2023-07-09", "2023-08-21",
    "2023-10-16", "2023-11-20", "2023-12-08", "2023-12-25",
    "2024-01-01", "2024-02-12", "2024-02-13", "2024-03-24", "2024-03-29", "2024-04-02",
    "2024-05-01", "2024-05-25", "2024-06-17", "2024-06-20", "2024-07-09", "2024-08-17",
    "2024-10-12", "2024-11-18", "2024-12-08", "2024-12-25",
])

# Efecto del día de la semana (lunes=0 ... domingo=6)
EFECTO_DIA = np.array([0.88, 0.92, 0.95, 0.98, 1.08, 1.25, 1.12])


def efecto_temperatura(fechas: pd.DatetimeIndex) -> np.ndarray:
    """Índice de 'calor' entre -1 (pleno invierno) y +1 (pleno verano) para el hemisferio sur."""
    dia_anio = fechas.dayofyear.to_numpy()
    # máximo de calor ~ 15 de enero (día 15), mínimo ~ mediados de julio
    return np.cos(2 * np.pi * (dia_anio - 15) / 365.25)


def generar(semilla: int = SEMILLA) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D")
    calor = efecto_temperatura(fechas)
    es_feriado = fechas.isin(FERIADOS)
    # Vacaciones de enero: la ciudad universitaria se vacía y baja el movimiento
    vacaciones = (fechas.month == 1).astype(float)
    tendencia = np.linspace(1.0, 1.12, len(fechas))  # crecimiento de 12% en 2 años

    # Calendario de promociones: ~1 de cada 12 días, por categoría
    categorias = sorted({p[0] for p in PRODUCTOS.values()})
    promo = {c: rng.random(len(fechas)) < 1 / 12 for c in categorias}

    filas = []
    for sucursal, factor_suc in SUCURSALES.items():
        # cada sucursal tiene su propio ruido diario de tráfico (clima, eventos)
        trafico = rng.normal(1.0, 0.06, len(fechas))
        for producto, (cat, base, precio, costo, vida, sens) in PRODUCTOS.items():
            mu = (
                base
                * factor_suc
                * EFECTO_DIA[fechas.dayofweek]
                * (1 + sens * calor)
                * np.where(es_feriado, 0.80, 1.0)
                * (1 - 0.18 * vacaciones)
                * tendencia
                * trafico
            )
            en_promo = promo[cat]
            mu = mu * np.where(en_promo, rng.uniform(1.25, 1.45, len(fechas)), 1.0)
            # Binomial negativa: sobre-dispersión típica de ventas reales
            k = 12.0
            demanda = rng.negative_binomial(k, k / (k + mu))
            filas.append(pd.DataFrame({
                "fecha": fechas,
                "sucursal": sucursal,
                "producto": producto,
                "categoria": cat,
                "unidades": demanda,
                "precio_unitario": np.where(en_promo, round(precio * 0.85), precio),
                "costo_unitario": costo,
                "en_promocion": en_promo.astype(int),
                "es_feriado": es_feriado.astype(int),
            }))

    ventas = pd.concat(filas, ignore_index=True)
    ventas["ingreso"] = ventas["unidades"] * ventas["precio_unitario"]

    catalogo = pd.DataFrame([
        {"producto": p, "categoria": c, "precio_lista": pr, "costo_unitario": co,
         "vida_util_dias": v, "perecedero": int(v >= 1)}
        for p, (c, _, pr, co, v, _) in PRODUCTOS.items()
    ])
    return ventas, catalogo


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    ventas, catalogo = generar()
    ventas.to_csv(SALIDA / "ventas_diarias.csv", index=False)
    catalogo.to_csv(SALIDA / "catalogo_productos.csv", index=False)
    print(f"ventas_diarias.csv: {len(ventas):,} filas")
    print(f"catalogo_productos.csv: {len(catalogo)} productos")


if __name__ == "__main__":
    main()
