"""Estilo gráfico común: paleta validada para daltonismo, ejes discretos, sin bordes innecesarios."""
import matplotlib as mpl
import matplotlib.pyplot as plt

AZUL = "#2a78d6"
NARANJA = "#eb6834"
AQUA = "#1baf7a"
AMARILLO = "#eda100"
MAGENTA = "#e87ba4"
VERDE = "#008300"
VIOLETA = "#6250d6"
ROJO = "#e34948"
SERIES = [AZUL, NARANJA, AQUA, AMARILLO, MAGENTA, VERDE, VIOLETA, ROJO]

TEXTO = "#0b0b0b"
TEXTO_SEC = "#52514e"
GRILLA = "#e4e3df"
FONDO = "#fcfcfb"


def aplicar_estilo() -> None:
    mpl.rcParams.update({
        "figure.facecolor": FONDO,
        "axes.facecolor": FONDO,
        "savefig.facecolor": FONDO,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": TEXTO_SEC,
        "axes.edgecolor": GRILLA,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRILLA,
        "grid.linewidth": 0.8,
        "xtick.color": TEXTO_SEC,
        "ytick.color": TEXTO_SEC,
        "text.color": TEXTO,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "lines.linewidth": 2,
        "legend.frameon": False,
    })


def miles(x, _pos=None) -> str:
    """Formateador de ejes: 12500 -> 12,5 mil."""
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f} M".replace(".", ",")
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f} mil"
    return f"{x:.0f}"


def guardar(fig: plt.Figure, ruta) -> None:
    fig.savefig(ruta)
    plt.close(fig)
