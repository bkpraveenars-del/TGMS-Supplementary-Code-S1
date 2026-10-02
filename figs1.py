import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from style import C, save, MM
import model as M

W = 180 * MM


def box(ax, x, y, w, h, text, fc, ec=None, fs=7.0, tc=None, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=%f" % r,
                                fc=fc, ec=ec or fc, lw=0.7, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or C["ink"], zorder=3, linespacing=1.35)


def arrow(ax, p0, p1, color=None, style="-|>", lw=0.8, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=7,
                                 color=color or C["ink2"], lw=lw, zorder=1, linestyle=ls,
                                 connectionstyle="arc3,rad=%f" % rad,
                                 shrinkA=1.5, shrinkB=1.5))


# ---------------------------------------------------------------- Fig 1
def fig1():
    fig, ax = plt.subplots(figsize=(W, 108 * MM))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    pale = "#eef4fc"; pale2 = "#fdeee8"; pale3 = "#e9f7f1"; pale4 = "#f6f2fd"

    ax.text(0.5, 0.975, "A photothermal intervention in two-line hybrid rice: where it could act and what bounds it",
            ha="center", fontsize=8.2, color=C["ink"], fontweight="bold")

    box(ax, 0.02, 0.80, 0.27, 0.11, "Hybrid seed production block\nTGMS female sown with pollinator\nsterility must hold for the whole\nflowering window", pale)
    box(ax, 0.36, 0.80, 0.27, 0.11, "Cold snap during the\nsensitive stage (stages IV-VI\nof panicle differentiation)\n$T$ falls below the CSIT", pale2)
    box(ax, 0.70, 0.80, 0.28, 0.11, "Partial fertility restoration\nselfed seed set, loss of\nhybrid purity below the\n96 % statutory floor", pale2)
    arrow(ax, (0.29, 0.855), (0.36, 0.855)); arrow(ax, (0.63, 0.855), (0.70, 0.855))

    box(ax, 0.02, 0.60, 0.30, 0.13, "Proposed intervention\nNIR-absorbing coating on the\nflag-leaf sheath enclosing the\nyoung panicle; sunlight is\nconverted to local heat", pale3)
    box(ax, 0.37, 0.60, 0.26, 0.13, "Energy balance of the\nenclosed organ\n$\\Delta T=\\Delta R/(c_p g_{\\mathrm{tot}})$\nSection 5, Eq. 6-9", pale)
    box(ax, 0.68, 0.60, 0.30, 0.13, "Quantitative ceiling\n$\\approx$1.3 K at clear-sky noon,\n$\\approx$0.3 K under overcast,\n0 K at night (Fig. 5, Fig. 6)", pale)
    arrow(ax, (0.32, 0.665), (0.37, 0.665)); arrow(ax, (0.63, 0.665), (0.68, 0.665))

    ax.add_patch(Rectangle((0.02, 0.30), 0.96, 0.25, fc="#fbfaf7", ec=C["grid"], lw=0.7, zorder=0))
    ax.text(0.5, 0.525, "Four independent constraints, each resolved in a separate section", ha="center",
            fontsize=7.6, color=C["ink"], fontweight="bold")
    labs = [("Genetic\nthe tms5 allele is a null;\nno optical route to a\nprotein that is absent\n(Section 3)", pale4),
            ("Physical\nday-only gain, bounded by\nthe unused NIR absorptance\nand by convection\n(Sections 4-5)", pale),
            ("Delivery\nthe target tissue sits inside\nthe sheath; particles stay\nin the apoplast\n(Section 6)", pale3),
            ("Statistical\nsterility is a bounded,\nhighly G x E trait; power\ndemands large networks\n(Section 8)", pale2)]
    for i, (t, c) in enumerate(labs):
        box(ax, 0.035 + i * 0.2425, 0.325, 0.225, 0.175, t, c, fs=6.7)

    box(ax, 0.02, 0.145, 0.30, 0.12, "Benchmarks that already work\ndeep water 15-20 cm or\ncold-water irrigation at\n19-21 degC, applied day and night", "#eef4fc", ec=C["blue"])
    box(ax, 0.36, 0.145, 0.27, 0.12, "Residual niche for the\nnanomaterial: not heating,\nbut sensing and decision\nsupport (Section 9)", pale3, ec=C["aqua"])
    box(ax, 0.68, 0.145, 0.30, 0.12, "Testable predictions and a\nminimum reporting standard\n(Table 9, Table 10)", "#f7f6f2", ec=C["muted"])
    arrow(ax, (0.17, 0.325), (0.17, 0.265)); arrow(ax, (0.49, 0.325), (0.49, 0.265))
    arrow(ax, (0.83, 0.325), (0.83, 0.265))

    ax.text(0.02, 0.075, "Boxes in blue are quantities computed in this review; boxes in orange are the failure modes they bound.\n"
                         "CSIT, critical sterility-inducing temperature; NIR, near-infrared (700-2500 nm).",
            fontsize=6.5, color=C["ink2"], va="center")
    save(fig, "Fig1")


# ---------------------------------------------------------------- Fig 2 (operating envelope) (molecular pathway)
def fig3():
    fig, ax = plt.subplots(figsize=(W, 112 * MM))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    hot = "#fdeee8"; cool = "#eef4fc"; gene = "#f6f2fd"

    ax.text(0.5, 0.975, "What the tms5 lesion actually does, and why an optical switch cannot be grafted onto it",
            ha="center", fontsize=8.2, fontweight="bold", color=C["ink"])

    box(ax, 0.30, 0.86, 0.40, 0.075, "TMS5  =  RNase Z$^{\\mathrm{S1}}$,  a tRNA 2',3'-cyclic phosphatase\n"
                                     "loss-of-function in essentially all commercial TGMS lines", gene, ec=C["violet"])

    ax.text(0.25, 0.805, "Permissive temperature  (below the CSIT)", ha="center", fontsize=7.4,
            color=C["blue"], fontweight="bold")
    ax.text(0.75, 0.805, "Restrictive temperature  (above the CSIT)", ha="center", fontsize=7.4,
            color=C["orange"], fontweight="bold")
    ax.plot([0.5, 0.5], [0.07, 0.79], color=C["grid"], lw=0.8, zorder=0)

    box(ax, 0.04, 0.68, 0.42, 0.095, "Substrate flux is low\ncP-tRNA and $UbL40$ transcripts stay\nwithin the capacity of back-up enzymes", cool)
    box(ax, 0.54, 0.68, 0.42, 0.095, "Substrate flux rises steeply\ncP-$\\Delta$CCA-tRNA accumulates; $UbL40$\nmRNAs are no longer processed", hot)

    box(ax, 0.04, 0.545, 0.42, 0.10, "Ribosome traffic is orderly\nOsHel2 / OsRqc1-OsVms1 clear the few\nstalled complexes; translation continues", cool)
    box(ax, 0.54, 0.545, 0.42, 0.10, "Ribosome-associated quality control is\nsaturated; CSIT1 and CSIT2 ubiquitinate\n80S subunits and misfolded products", hot)

    box(ax, 0.04, 0.41, 0.42, 0.10, "R-loops at the $TMS5$ locus stay low;\nOsLS1 keeps transcription-replication\nconflict in check", cool)
    box(ax, 0.54, 0.41, 0.42, 0.10, "R-loops accumulate over the locus,\namplifying the transcriptional defect\n(Zhu et al. 2026)", hot)

    box(ax, 0.04, 0.285, 0.42, 0.09, "Tapetal programmed cell death runs\non schedule; catalase activity holds\nreactive oxygen species in balance", cool)
    box(ax, 0.54, 0.285, 0.42, 0.09, "Premature tapetal cell death,\nROS burst, pollen abortion", hot)

    box(ax, 0.04, 0.165, 0.42, 0.085, "Pollen is viable\nthe line selfs and can be multiplied", cool, ec=C["blue"])
    box(ax, 0.54, 0.165, 0.42, 0.085, "Complete male sterility\nthe line is used as a female parent", hot, ec=C["orange"])
    for y0, y1 in ((0.68, 0.645), (0.545, 0.51), (0.41, 0.375), (0.285, 0.25)):
        arrow(ax, (0.25, y0), (0.25, y1), color=C["blue"])
        arrow(ax, (0.75, y0), (0.75, y1), color=C["orange"])
    arrow(ax, (0.42, 0.86), (0.25, 0.775), color=C["violet"], rad=0.15)
    arrow(ax, (0.58, 0.86), (0.75, 0.775), color=C["violet"], rad=-0.15)

    box(ax, 0.04, 0.025, 0.92, 0.105,
        "Why the draft mechanism fails:  a photothermally generated hot carrier can only act on a molecule that exists.  In tms5 lines the\n"
        "transcript is truncated or the protein is absent, so there is no mutant RNase Z$^{\\mathrm{S1}}$ left to misfold.  Every step above is a flux\n"
        "problem set by bulk temperature, and the only physical handle on it is the temperature of the tissue itself (Sections 4-5).",
        "#fbfaf7", ec=C["muted"], fs=6.6)
    save(fig, "Fig3")


# ---------------------------------------------------------------- Fig 3
def fig2():
    fig = plt.figure(figsize=(W, 82 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.52, wspace=0.34,
                          left=0.065, right=0.985, top=0.9, bottom=0.11)

    # (a) sterility response and CSIT definition
    ax = fig.add_subplot(gs[0, 0])
    T = np.linspace(19, 29, 400)
    for csit, col, lab in ((22.0, C["blue"], "low-CSIT line, 22.0 $^\\circ$C"),
                           (23.5, C["aqua"], "screening standard, 23.5 $^\\circ$C"),
                           (25.0, C["orange"], "drifted line, 25.0 $^\\circ$C")):
        ax.plot(T, 100 * M.sterility_logistic(T, csit), color=col, lw=1.1, label=lab)
    ax.axhline(99.5, color=C["muted"], lw=0.6, ls=":")
    ax.text(19.2, 95.5, "99.5 % floor for\ncommercial use", fontsize=6.0, color=C["ink2"])
    ax.set_xlabel("mean temperature over the sensitive window ($^\\circ$C)")
    ax.set_ylabel("pollen sterility (%)")
    ax.set_title("(a)  the sterility response is steep", loc="left")
    ax.legend(loc="lower right")

    # (b) screening outcomes reported for 97 Chinese dual-purpose lines
    ax = fig.add_subplot(gs[0, 1])
    cats = ["sterile after\n6 d at 23.5", "sterile after\n10 d at 23.5",
            ">70 % fertile\nafter 10 d at 19", "safe for both\noperations"]
    vals = [42, 12, 47, 6]
    ax.bar(cats, vals, color=[C["aqua"], C["blue"], C["yellow"], C["orange"]], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, str(v), ha="center", fontsize=6.6, color=C["ink2"])
    ax.set_ylabel("lines out of 97 screened")
    ax.set_ylim(0, 56)
    ax.tick_params(axis="x", labelsize=6.0)
    ax.set_title("(b)  how few lines pass both tests", loc="left")
    ax.text(0.02, 0.93, "data of Chen et al. (2022)", transform=ax.transAxes,
            fontsize=6.0, color=C["ink2"])

    # (c) purity consequence
    ax = fig.add_subplot(gs[0, 2])
    f = np.linspace(0, 0.14, 300)
    for ss, col, lab in ((0.12, C["blue"], "selfed set 0.12"),
                          (0.25, C["aqua"], "selfed set 0.25"),
                          (0.40, C["orange"], "selfed set 0.40")):
        ax.plot(100 * f, 100 * M.purity(f, ss), color=col, lw=1.1, label=lab)
    ax.axhline(96, color=C["red"], lw=0.7, ls="--")
    ax.text(0.4, 96.6, "96 % statutory purity", fontsize=6.0, color=C["red"])
    ax.set_xlabel("florets regaining male function (%)")
    ax.set_ylabel("hybrid purity (%)")
    ax.set_ylim(84, 101)
    ax.set_title("(c)  a few per cent is already fatal", loc="left")
    ax.legend(loc="lower left")

    # (d) a cool spell resolved hour by hour, with the coating applied only when the sun shines
    ax = fig.add_subplot(gs[1, :2])
    d = np.linspace(0, 12, 12 * 24 + 1)
    hour = (d * 24) % 24
    snap = 26.4 - 0.1 * d - 4.6 * np.exp(-0.5 * ((d - 5.4) / 1.5) ** 2)
    tair = snap + 3.1 * np.sin(2 * np.pi * (d - 0.35))
    # overcast during the depression, clear at the margins of the window
    cloud = 0.25 + 0.75 * np.clip(np.abs(d - 5.4) / 4.0, 0, 1)
    S = M.diurnal_irradiance(hour, S_peak=900.0) * cloud
    dT = M.dT_steady(M.absorbed_increment(0.25, S), u=1.0)
    ax.fill_between(d, tair, 23.5, where=tair < 23.5, color="#dce8f8", lw=0, zorder=0)
    ax.plot(d, tair, color=C["ink2"], lw=0.55, label="air temperature, hourly")
    ax.plot(d, tair + dT, color=C["aqua"], lw=0.55, ls="-",
            label="with the coating ($\\Delta a_{\\mathrm{NIR}}$ = 0.25)")
    ax.plot(d, snap, color=C["blue"], lw=1.3, label="daily mean, uncoated")
    ax.axhline(23.5, color=C["red"], lw=0.8, ls="--", label="CSIT 23.5 $^\\circ$C")
    ax.set_xlabel("day of the temperature-sensitive window")
    ax.set_ylabel("temperature ($^\\circ$C)")
    ax.set_title("(d)  an hour-resolved cool spell: the coating is dark exactly when the air is coldest", loc="left")
    ax.legend(loc="lower left", ncol=2)
    ax.set_xlim(0, 12)

    # (e) cumulative exposure below the threshold
    ax = fig.add_subplot(gs[1, 2])
    labels = ["no\ntreatment", "photothermal\ncoating", "deep water\n15-20 cm"]
    dt_water = M.water_offset(17.0)
    hours = []
    for off in (np.zeros_like(d), dT, np.full_like(d, dt_water)):
        hours.append(float(np.trapezoid((tair + off) < 23.5, d) * 24))
    ax.bar(labels, hours, color=[C["orange"], C["yellow"], C["blue"]], width=0.62)
    for i, h in enumerate(hours):
        ax.text(i, h + 2.5, "%.0f h" % h, ha="center", fontsize=6.6, color=C["ink2"])
    ax.set_ylabel("hours below the CSIT\nover the 12-day window")
    ax.set_title("(e)  exposure actually removed", loc="left")
    ax.set_ylim(0, max(hours) * 1.25)
    save(fig, "Fig2")
    return hours, float(dT.mean()), float(dt_water)


if __name__ == "__main__":
    fig1(); fig3(); print(fig2())
