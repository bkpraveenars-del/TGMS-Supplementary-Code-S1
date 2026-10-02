import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import os

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#e87ba4", green="#008300", violet="#4a3aa7", red="#e34948",
         ink="#0b0b0b", ink2="#52514e", muted="#8a8984", grid="#e6e5e0", surf="#ffffff")

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.edgecolor": C["ink2"],
    "axes.labelcolor": C["ink"],
    "xtick.color": C["ink2"],
    "ytick.color": C["ink2"],
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": C["grid"],
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "savefig.dpi": 600,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
MM = 1 / 25.4


def panel(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left", color=C["ink"])


def save(fig, name):
    png = os.path.join(OUT, name + ".png")
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    im = Image.open(png).convert("RGB")
    im.save(os.path.join(OUT, name + ".tif"), dpi=(600, 600), compression="tiff_lzw")
    plt.close(fig)
    print(name, im.size)
