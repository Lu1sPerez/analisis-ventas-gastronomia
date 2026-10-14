"""
Ejecuta el análisis completo y guarda gráficos y resultados en reports/.

    python main.py
"""
import json
from pathlib import Path

from src import graficos as g
from src import modelo, pipeline
from src.estilo import guardar

FIG = Path(__file__).parent / "reports" / "figures"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    r = pipeline.ejecutar()

    guardar(g.ingresos_mensuales_por_categoria(r.ventas), FIG / "01_ingresos_por_categoria.png")
    guardar(g.efecto_dia_semana(r.ventas), FIG / "02_dia_semana.png")
    guardar(g.estacionalidad_bebidas(r.ventas), FIG / "03_estacionalidad_bebidas.png")
    guardar(g.efecto_promociones(r.ventas), FIG / "04_efecto_promociones.png")
    guardar(g.pronostico_vs_real(r.test), FIG / "05_pronostico_vs_real.png")
    guardar(g.importancia_variables(r.modelo, r.test, modelo.FEATURES), FIG / "06_importancia_variables.png")
    guardar(g.frontera_merma(r.curva_manual, r.curva_modelo, r.resumen), FIG / "07_merma_vs_servicio.png")

    salida = Path(__file__).parent / "reports"
    r.metricas.to_csv(salida / "metricas_pronostico.csv")
    (salida / "resumen.json").write_text(json.dumps(r.resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nPrecisión del pronóstico (test sep–dic 2024):")
    print(r.metricas.to_string())
    print("\nSimulación de merma:")
    for k, v in r.resumen.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
