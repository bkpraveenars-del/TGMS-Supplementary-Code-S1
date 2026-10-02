"""Reproduce every computed number and every figure in the manuscript.

Usage:  python run_all.py
Outputs: figures/Fig1..Fig10 (PNG 600 dpi + LZW TIFF) and computed_values.csv
"""
import sys, os, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import model as M
import figs1, figs2, figs3

rows = []


def rec(name, value, unit, where):
    rows.append(dict(quantity=name, value=round(float(value), 4), unit=unit, appears_in=where))


# --- Section 5: energy balance -------------------------------------------
rec("radiative conductance g_r at 22 C", M.g_radiative(), "mol m-2 s-1", "Eq. 3, Table 5")
for u in (0.3, 1.0, 3.0):
    rec("total conductance at u=%.1f m/s, d=10 mm" % u,
        M.g_boundary(u, 0.010) + M.g_radiative(), "mol m-2 s-1", "Section 5.2")
    rec("thermal resistance at u=%.1f m/s" % u,
        M.CP * (M.g_boundary(u, 0.010) + M.g_radiative()), "W m-2 K-1", "Section 5.2")

cases = [("clear-sky noon", 900, 1.0, 0.25), ("clear-sky noon, still air", 900, 0.3, 0.25),
         ("clear-sky noon, breezy", 900, 3.0, 0.25), ("clear-sky noon, maximal coating", 900, 0.5, 0.45),
         ("hazy", 500, 1.0, 0.25), ("overcast", 180, 1.0, 0.25), ("night", 0, 1.0, 0.25)]
for lab, S, u, da in cases:
    rec("dT %s" % lab, M.dT_steady(M.absorbed_increment(da, S), u), "K", "Table 4")

# --- Section 4: optical headroom -----------------------------------------
nm = np.linspace(300, 2500, 2400)
S = figs2.solar_spectrum(nm)
a = figs2.sheath_absorptance(nm)
m = nm >= 700
rec("NIR share of global shortwave flux", np.trapezoid(S[m], nm[m]) / 1000.0, "fraction", "Section 4.2")
rec("unexploited NIR flux at 1000 W/m2", np.trapezoid(S[m] * (1 - a[m]), nm[m]), "W m-2", "Section 4.2")

# --- Section 5.3: diurnal ------------------------------------------------
h = np.linspace(0, 24, 1441)
Sd = M.diurnal_irradiance(h, 900.0) * 0.55
dT = M.dT_steady(M.absorbed_increment(0.25, Sd), 1.0)
day = (h >= 6) & (h <= 18)
for lab, v in (("peak", dT.max()), ("daylight mean", dT[day].mean()),
               ("24 h mean", dT.mean()), ("night mean", dT[~day].mean())):
    rec("dT cool part-cloudy day, %s" % lab, v, "K", "Fig. 6c, Table 4")

hours, mean_spell, dt_water = figs1.fig2()
rec("dT mean over the 12-day cool spell", mean_spell, "K", "Section 5.3, Table 4")
rec("hours below CSIT, untreated", hours[0], "h", "Fig. 2e")
rec("hours below CSIT, coating", hours[1], "h", "Fig. 2e")
rec("hours below CSIT, deep water", hours[2], "h", "Fig. 2e")
rec("modelled deep-water offset at 17 cm", dt_water, "K", "Section 7.1, Table 6")

# --- Section 8: simulated multi-environment trial ------------------------
mean, obs, env_T, lg, se, lg2, se2 = figs3.simulate_met()
ipca_g, ipca_e, expl, asv, resid = figs3.ammi(mean)
rec("IPCA1 share of interaction SS (simulated)", expl[0], "fraction", "Section 8.3, Fig. 10e")
rec("IPCA2 share of interaction SS (simulated)", expl[1], "fraction", "Section 8.3, Fig. 10e")
sd_e = np.sqrt(2 * (0.55 ** 2 / 3 + 0.35 ** 2) / 12)
rec("smallest detectable difference, 12 environments", 1.96 * sd_e, "logit units", "Section 8.3, Fig. 10f")

figs1.fig1(); figs1.fig3()
figs2.fig4(); figs2.fig5(); figs2.fig6(); figs2.fig8()
figs3.fig7(); figs3.fig9(); figs3.fig10()

with open(os.path.join(HERE, "computed_values.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["quantity", "value", "unit", "appears_in"])
    w.writeheader(); w.writerows(rows)
print("wrote computed_values.csv with %d entries" % len(rows))
