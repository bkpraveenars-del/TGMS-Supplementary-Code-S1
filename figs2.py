import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle
from style import C, save, MM
import model as M

W = 180 * MM


def solar_spectrum(nm):
    """Analytic stand-in for the AM1.5G global spectrum: a 5778 K Planck curve
    attenuated by a Rayleigh-type short-wavelength term and by the principal
    water-vapour, oxygen and carbon-dioxide bands.  Normalised to 1000 W m-2;
    the resulting share above 700 nm is close to the tabulated 0.52."""
    lam = nm * 1e-9
    h, c, kB, T = 6.626e-34, 2.998e8, 1.381e-23, 5778.0
    planck = 2 * h * c ** 2 / lam ** 5 / (np.exp(h * c / (lam * kB * T)) - 1)
    atm = np.exp(-0.42 * (550.0 / nm) ** 4) * (1 - 0.55 * np.exp(-((nm - 330) ** 2) / (2 * 60 ** 2)))
    x = nm / 1000.0
    for cen, w, dpth in ((0.72, 0.012, 0.22), (0.76, 0.008, 0.55), (0.82, 0.02, 0.28),
                         (0.94, 0.035, 0.70), (1.13, 0.045, 0.78), (1.40, 0.06, 0.97),
                         (1.87, 0.07, 0.99), (2.05, 0.05, 0.55), (2.5, 0.08, 0.8)):
        atm = atm * (1 - dpth * np.exp(-((x - cen) ** 2) / (2 * w ** 2)))
    atm = atm * np.exp(-0.75 * np.clip((nm - 700.0) / 1800.0, 0, None))
    body = planck * atm
    body[nm < 300] = 0.0
    return body / np.trapezoid(body, nm) * 1000.0


def sheath_absorptance(nm):
    """Illustrative absorptance of a green flag-leaf sheath: strong in the
    visible through chlorophyll, low across the NIR plateau, rising again in
    the water bands.  Shape follows published leaf optical measurements."""
    a = np.full_like(nm, 0.12, dtype=float)
    a += 0.80 * np.exp(-((nm - 450) ** 2) / (2 * 55 ** 2))
    a += 0.72 * np.exp(-((nm - 670) ** 2) / (2 * 40 ** 2))
    a += 0.30 * np.exp(-((nm - 550) ** 2) / (2 * 60 ** 2))
    a += 0.55 / (1 + np.exp(-(nm - 1500) / 110.0))
    a += 0.25 * np.exp(-((nm - 1200) ** 2) / (2 * 60 ** 2))
    return np.clip(a, 0.05, 0.96)


# ---------------------------------------------------------------- Fig 4
def fig4():
    fig = plt.figure(figsize=(W, 78 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.45, 1],
                          hspace=0.55, wspace=0.28, left=0.07, right=0.985, top=0.9, bottom=0.11)
    nm = np.linspace(300, 2500, 2400)
    S = solar_spectrum(nm)
    a = sheath_absorptance(nm)

    ax = fig.add_subplot(gs[:, 0])
    ax.fill_between(nm, 0, S, color="#f0eee7", lw=0, label="solar irradiance (AM1.5G, 1000 W m$^{-2}$)")
    ax.fill_between(nm, 0, S * a, color=C["green"], alpha=0.35, lw=0, label="already absorbed by the organ")
    ax.plot(nm, S, color=C["ink2"], lw=0.7)
    nirmask = nm >= 700
    head = np.trapezoid(S[nirmask] * (1 - a[nirmask]), nm[nirmask])
    ax.fill_between(nm[nirmask], S[nirmask] * a[nirmask], S[nirmask], color=C["orange"], alpha=0.30,
                    lw=0, label="headroom a photothermal coating can take (%.0f W m$^{-2}$)" % head)
    ax.axvline(700, color=C["muted"], lw=0.6, ls=":")
    ax.text(760, S.max() * 0.52, "NIR, 700-2500 nm\n%.0f %% of the incident flux"
            % (100 * np.trapezoid(S[nirmask], nm[nirmask]) / 1000.0), fontsize=6.3, color=C["ink2"])
    ax.set_ylim(0, S.max() * 1.42)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("spectral irradiance (W m$^{-2}$ nm$^{-1}$)")
    ax.set_title("(a)  the entire optical budget available to a coating", loc="left")
    ax.legend(loc="upper right")
    ax.set_xlim(300, 2500)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(nm, a, color=C["green"], lw=1.1, label="uncoated sheath")
    ax.plot(nm, np.clip(a + 0.45 * (1 - a) * (nm > 700), 0, 1), color=C["violet"], lw=1.1,
            label="with a broadband NIR absorber")
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("absorptance")
    ax.set_ylim(0, 1.05); ax.set_xlim(300, 2500)
    ax.set_title("(b)  where the increment must come from", loc="left")
    ax.legend(loc="lower right")

    ax = fig.add_subplot(gs[1, 1])
    eta = np.linspace(0.3, 1.0, 200)
    for da, col in ((0.10, C["yellow"]), (0.25, C["aqua"]), (0.45, C["blue"])):
        ax.plot(100 * eta, M.dT_steady(eta * M.absorbed_increment(da, 900.0), 1.0),
                color=col, lw=1.1, label="$\\Delta a_{\\mathrm{NIR}}$ = %.2f" % da)
    ax.axvspan(80, 99, color="#f0eee7", lw=0, zorder=0)
    ax.text(89.5, 0.12, "range reported for\n17 nanoheaters", ha="center", fontsize=6.0, color=C["ink2"])
    ax.set_xlabel("photothermal conversion efficiency $\\eta$ (%)")
    ax.set_ylabel("$\\Delta T$ at clear-sky noon (K)")
    ax.set_title("(c)  $\\eta$ is not the limiting quantity", loc="left")
    ax.legend(loc="upper left")
    save(fig, "Fig4")
    return head


# ---------------------------------------------------------------- Fig 5
def fig5():
    fig = plt.figure(figsize=(W, 66 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.065, right=0.975, top=0.86, bottom=0.17)

    ax = fig.add_subplot(gs[0, 0])
    da = np.linspace(0, 0.6, 200)
    for S, col, lab in ((900, C["blue"], "clear sky, 900 W m$^{-2}$"),
                        (500, C["aqua"], "hazy, 500 W m$^{-2}$"),
                        (180, C["orange"], "overcast, 180 W m$^{-2}$"),
                        (0, C["ink2"], "night, 0 W m$^{-2}$")):
        ax.plot(da, M.dT_steady(M.absorbed_increment(da, S), 1.0), color=col, lw=1.1, label=lab)
    ax.set_xlabel("increase in NIR absorptance, $\\Delta a_{\\mathrm{NIR}}$")
    ax.set_ylabel("steady-state $\\Delta T$ (K)")
    ax.set_title("(a)  gain against sky condition", loc="left")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[0, 1])
    u = np.linspace(0.1, 4.0, 200)
    for d, col, lab in ((0.006, C["blue"], "d = 6 mm"), (0.010, C["aqua"], "d = 10 mm"),
                        (0.020, C["orange"], "d = 20 mm")):
        ax.plot(u, M.dT_steady(M.absorbed_increment(0.25, 900.0), u, d), color=col, lw=1.1, label=lab)
    ax.set_xlabel("wind speed at the panicle (m s$^{-1}$)")
    ax.set_ylabel("steady-state $\\Delta T$ (K)")
    ax.set_title("(b)  convection erases the gain", loc="left")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[0, 2])
    DA, U = np.meshgrid(np.linspace(0, 0.6, 160), np.linspace(0.1, 3.0, 160))
    Z = M.dT_steady(M.absorbed_increment(DA, 900.0), U)
    cs = ax.contourf(DA, U, Z, levels=np.arange(0, 4.1, 0.25), cmap="YlGnBu")
    cl = ax.contour(DA, U, Z, levels=[0.5, 1.0, 1.5, 2.0, 3.0], colors="w", linewidths=0.6)
    ax.clabel(cl, fmt="%.1f K", fontsize=5.6)
    ax.plot([0.45], [0.5], marker="*", ms=8, color=C["red"], zorder=5)
    ax.text(0.44, 0.65, "most favourable case\nconsidered here", fontsize=6.0, color="w", ha="right")
    fig.colorbar(cs, ax=ax, pad=0.02, label="$\\Delta T$ (K)")
    ax.set_xlabel("$\\Delta a_{\\mathrm{NIR}}$")
    ax.set_ylabel("wind speed (m s$^{-1}$)")
    ax.set_title("(c)  the whole feasible envelope at noon", loc="left")
    save(fig, "Fig5")
    return float(M.dT_steady(M.absorbed_increment(0.45, 900.0), 0.5))


# ---------------------------------------------------------------- Fig 6
def fig6():
    fig = plt.figure(figsize=(W, 70 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.35,
                          left=0.065, right=0.98, top=0.86, bottom=0.16)
    h = np.linspace(0, 24, 1441)
    tair = M.diurnal_air_temperature(h, 19.0, 26.0)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(h, tair, color=C["ink2"], lw=1.0, label="air temperature")
    for cloud, col, lab in ((1.0, C["blue"], "clear"), (0.45, C["aqua"], "hazy"), (0.2, C["orange"], "overcast")):
        S = M.diurnal_irradiance(h, 900.0) * cloud
        ax.plot(h, tair + M.dT_steady(M.absorbed_increment(0.25, S), 1.0), color=col, lw=1.0,
                label="coated organ, %s" % lab)
    ax.axhline(23.5, color=C["red"], lw=0.8, ls="--", label="CSIT")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
    ax.set_xlabel("hour of day"); ax.set_ylabel("temperature ($^\\circ$C)")
    ax.set_title("(a)  the heating is in phase with the warmest hours", loc="left")
    ax.legend(loc="lower center", ncol=2)

    ax = fig.add_subplot(gs[0, 1])
    clouds = np.linspace(0.1, 1.0, 60)
    means, nights = [], []
    for c in clouds:
        S = M.diurnal_irradiance(h, 900.0) * c
        dT = M.dT_steady(M.absorbed_increment(0.25, S), 1.0)
        means.append(dT.mean())
        nights.append(dT[(h < 6) | (h > 18)].mean())
    ax.plot(100 * clouds, means, color=C["blue"], lw=1.2, label="24 h mean gain")
    ax.plot(100 * clouds, nights, color=C["orange"], lw=1.2, label="night-time gain")
    ax.set_xlabel("fraction of clear-sky irradiance (%)")
    ax.set_ylabel("$\\Delta T$ averaged over the period (K)")
    ax.set_title("(b)  what survives daily averaging", loc="left")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[0, 2])
    lab = ["daytime\npeak", "daylight\nmean", "24 h\nmean", "night\nmean"]
    S = M.diurnal_irradiance(h, 900.0) * 0.55
    dT = M.dT_steady(M.absorbed_increment(0.25, S), 1.0)
    day = (h >= 6) & (h <= 18)
    vals = [dT.max(), dT[day].mean(), dT.mean(), dT[~day].mean()]
    ax.bar(lab, vals, color=[C["blue"], C["aqua"], C["yellow"], C["ink2"]], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.025, "%.2f K" % v, ha="center", fontsize=6.4, color=C["ink2"])
    ax.set_ylabel("$\\Delta T$ (K)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("(c)  a representative cool, part-cloudy day", loc="left")
    save(fig, "Fig6")
    return vals


# ---------------------------------------------------------------- Fig 7
def fig8():
    fig = plt.figure(figsize=(W, 74 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1.25], wspace=0.38,
                          left=0.07, right=0.985, top=0.86, bottom=0.26)

    ax = fig.add_subplot(gs[0, 0])
    depth = np.linspace(0, 25, 200)
    ax.plot(depth, M.water_offset(depth), color=C["blue"], lw=1.2)
    ax.axvspan(15, 20, color="#eef4fc", lw=0, zorder=0)
    ax.text(17.5, 0.45, "depth used in the\nJapanese practice", ha="center", fontsize=6.0, color=C["ink2"])
    ax.set_xlabel("irrigation depth above the growing point (cm)")
    ax.set_ylabel("offset applied to the young panicle (K)")
    ax.set_title("(a)  deep water, modelled", loc="left")
    ax.text(0.03, 0.93, "saturating curve fitted to the\nqualitative field literature;\nnot a measured dataset",
            transform=ax.transAxes, fontsize=5.9, color=C["ink2"], va="top")

    ax = fig.add_subplot(gs[0, 1])
    names = ["photothermal\ncoating", "deep water\n15-20 cm", "cold water\n19-21 $^\\circ$C",
             "sowing date\nshift"]
    day = [1.23, 2.46, -3.0, 0.0]
    night = [0.0, 2.46, -3.0, 0.0]
    x = np.arange(len(names))
    ax.bar(x - 0.19, day, 0.36, color=C["blue"], label="daytime")
    ax.bar(x + 0.19, night, 0.36, color=C["ink2"], label="night-time")
    ax.axhline(0, color=C["ink2"], lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.0)
    ax.set_ylabel("temperature offset achieved (K)")
    ax.set_title("(b)  magnitude and duty cycle", loc="left")
    ax.legend(loc="lower left")
    ax.text(3, 0.25, "shifts the window,\ndoes not move $T$", ha="center", fontsize=5.9, color=C["ink2"])

    ax = fig.add_subplot(gs[0, 2])
    crit = ["gain at\nnight", "works in\ncloud", "cost per\nhectare", "regulatory\npath",
            "residue in\nthe paddy"]
    mat = np.array([[0.0, 0.15, 0.55, 0.20, 0.25],
                    [1.0, 1.00, 0.70, 1.00, 1.00],
                    [1.0, 1.00, 0.35, 1.00, 1.00],
                    [0.5, 0.60, 0.95, 1.00, 1.00]])
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(crit))); ax.set_xticklabels(crit, fontsize=5.9)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([n.replace("\n", " ") for n in names], fontsize=6.2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "%.2f" % mat[i, j], ha="center", va="center", fontsize=5.8,
                    color="#222222")
    fig.colorbar(im, ax=ax, pad=0.02, label="favourability (1 = best)")
    ax.set_title("(c)  a like-for-like scorecard", loc="left")
    save(fig, "Fig8")


if __name__ == "__main__":
    print("headroom", fig4())
    print("best case dT", fig5())
    print("diurnal", fig6())
    fig8()
