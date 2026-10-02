import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse
from style import C, save, MM
import model as M

W = 180 * MM
rng = np.random.default_rng(20260401)


def box(ax, x, y, w, h, text, fc, ec=None, fs=6.8, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=%f" % r,
                                fc=fc, ec=ec or fc, lw=0.7, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=C["ink"], zorder=3, linespacing=1.35)


# ---------------------------------------------------------------- Fig 8
def fig7():
    fig = plt.figure(figsize=(W, 76 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.16,
                          left=0.03, right=0.985, top=0.9, bottom=0.07)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("(a)  the barriers between a sprayed droplet and the young panicle", loc="left", fontsize=7.6)
    layers = [("spray droplet on the sheath surface", "#eef4fc"),
              ("epicuticular wax and cuticle, 0.1-5 $\\mu$m;\nan aqueous pore of roughly 2 nm", "#f0eee7"),
              ("epidermal cell wall, pore 5-20 nm", "#f0eee7"),
              ("apoplast of the sheath; most particles stop here", "#fdeee8"),
              ("plasmodesmata, 3-50 nm; symplastic entry is\nselective and largely size-excluded", "#fdeee8"),
              ("tapetal cell layer enclosing the microspores", "#e9f7f1")]
    y = 0.86
    for i, (t, c) in enumerate(layers):
        box(ax, 0.06, y, 0.88, 0.095, t, c, ec=C["muted"] if i else C["blue"])
        if i < len(layers) - 1:
            ax.add_patch(FancyArrowPatch((0.5, y), (0.5, y - 0.045), arrowstyle="-|>",
                                         mutation_scale=7, color=C["ink2"], lw=0.8))
        y -= 0.14
    ax.text(0.5, 0.085, "No published study demonstrates uptake of an engineered particle into the rice tapetum.\n"
                        "Reports of pollen magnetofection have not been reproduced independently;\n"
                        "the one direct attempt found no evidence for it.",
            ha="center", fontsize=6.3, color=C["red"], linespacing=1.4)

    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("(b)  the size and charge window for foliar entry", loc="left", fontsize=7.6)
    ax.add_patch(Rectangle((0, 15), 20, 35, fc="#e9f7f1", ec=C["aqua"], lw=0.8, zorder=0))
    ax.text(10, 46, "window reported for\nstomatal and cuticular entry", ha="center", fontsize=6.2,
            color=C["ink2"])
    pts = [(5, 32, "quantum dot"), (11, 24, "small Au"), (18, 20, "silica"),
           (45, 8, "polyaniline particle"), (80, -12, "Fe$_3$O$_4$ cluster"),
           (140, -25, "LDH platelet"), (300, -30, "nano-biochar"), (55, 28, "coated Au")]
    for x, yv, lab in pts:
        inside = (x <= 20) and (yv >= 15)
        ax.scatter([x], [yv], s=22, color=C["aqua"] if inside else C["orange"], zorder=3,
                   edgecolor="white", linewidth=0.4)
        ax.annotate(lab, (x, yv), textcoords="offset points", xytext=(4, 4), fontsize=5.9,
                    color=C["ink2"])
    ax.axhline(0, color=C["muted"], lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(2, 600); ax.set_ylim(-40, 52)
    ax.set_xlabel("hydrodynamic diameter (nm)")
    ax.set_ylabel("zeta potential (mV)")
    ax.text(0.98, 0.03, "window after Hu et al. (2020); particle positions are nominal\nvalues for the"
                        " material classes discussed in Section 6",
            transform=ax.transAxes, ha="right", fontsize=5.8, color=C["ink2"])
    save(fig, "Fig7")


# ---------------------------------------------------------------- Fig 9
def fig9():
    fig = plt.figure(figsize=(W, 72 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.36, left=0.065, right=0.98, top=0.86, bottom=0.16)

    ax = fig.add_subplot(gs[0, 0])
    T = np.linspace(20.5, 26.5, 300)
    for dT, col, lab in ((0.0, C["orange"], "untreated"), (0.31, C["yellow"], "coating, 24 h mean +0.31 K"),
                         (2.46, C["blue"], "deep water, +2.46 K")):
        f = 1 - M.sterility_logistic(T + dT, 23.5)
        ax.plot(T, 100 * M.purity(np.clip(f, 0, 0.5), 0.25), color=col, lw=1.2, label=lab)
    ax.axhline(96, color=C["red"], lw=0.7, ls="--")
    ax.set_xlabel("mean air temperature in the sensitive window ($^\\circ$C)")
    ax.set_ylabel("predicted hybrid purity (%)")
    ax.set_title("(a)  purity recovered by each option", loc="left")
    ax.legend(loc="lower right")

    ax = fig.add_subplot(gs[0, 1])
    margin = np.linspace(-2.0, 2.0, 200)
    for dT, col, lab in ((0.31, C["yellow"], "coating"), (2.46, C["blue"], "deep water")):
        gain = 100 * (M.sterility_logistic(23.5 + margin + dT, 23.5)
                      - M.sterility_logistic(23.5 + margin, 23.5))
        ax.plot(margin, gain, color=col, lw=1.2, label=lab)
    ax.axvline(0, color=C["muted"], lw=0.6, ls=":")
    ax.set_xlabel("air temperature relative to the CSIT (K)")
    ax.set_ylabel("percentage points of sterility gained")
    ax.set_title("(b)  where an offset is worth most", loc="left")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[0, 2])
    cats = ["genetic\nmargin\n(low CSIT)", "site and\nsowing\ndate", "water\nmanagement",
            "photothermal\ncoating"]
    vals = [3.0, 2.2, 2.46, 0.31]
    cols = [C["violet"], C["green"], C["blue"], C["yellow"]]
    ax.barh(cats, vals, color=cols, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.06, i, "%.2f K" % v, va="center", fontsize=6.4, color=C["ink2"])
    ax.set_xlabel("effective temperature margin delivered (K)")
    ax.set_xlim(0, 3.8)
    ax.set_title("(c)  the levers ranked by what they buy", loc="left")
    ax.text(0.98, 0.04, "the first two are measured in the cited field studies;\nthe last two are computed here",
            transform=ax.transAxes, ha="right", fontsize=5.8, color=C["ink2"])
    save(fig, "Fig9")


# ---------------------------------------------------------------- Fig 10
def simulate_met(n_gen=9, n_env=12, reps=3, seed=7):
    """Simulate a genotype x environment trial of pollen sterility on the logit
    scale, with AMMI decomposition.  Entirely synthetic: used only to size
    future experiments, never as evidence about any real line."""
    r = np.random.default_rng(seed)
    gen = np.array([-1.1, -0.8, -0.45, -0.2, 0.0, 0.25, 0.5, 0.9, 1.3])[:n_gen]
    env_T = np.linspace(21.5, 26.0, n_env)
    mu = 2.2
    env = 0.9 * (env_T - env_T.mean())
    # two interaction axes: sensitivity to temperature and to photoperiod proxy
    load_g = np.array([1.3, 1.0, 0.7, 0.3, 0.0, -0.3, -0.6, -1.0, -1.4])[:n_gen]
    score_e = (env_T - env_T.mean()) / 3.4
    load_g2 = r.normal(0, 0.75, n_gen)
    score_e2 = r.normal(0, 0.85, n_env)
    ge = np.outer(load_g, score_e) + np.outer(load_g2, score_e2)
    mean = mu + gen[:, None] + env[None, :] + ge
    obs = mean[:, :, None] + r.normal(0, 0.55, (n_gen, n_env, reps))
    return mean, obs, env_T, load_g, score_e, load_g2, score_e2


def ammi(mean):
    g, e = mean.shape
    gm = mean.mean(1, keepdims=True); em = mean.mean(0, keepdims=True); om = mean.mean()
    resid = mean - gm - em + om
    U, s, Vt = np.linalg.svd(resid, full_matrices=False)
    ipca_g = U[:, :2] * np.sqrt(s[:2])
    ipca_e = (Vt[:2].T * np.sqrt(s[:2]))
    expl = s ** 2 / np.sum(s ** 2)
    asv = np.sqrt((expl[0] / expl[1] * ipca_g[:, 0]) ** 2 + ipca_g[:, 1] ** 2)
    return ipca_g, ipca_e, expl, asv, resid


def fig10():
    mean, obs, env_T, lg, se, lg2, se2 = simulate_met()
    ipca_g, ipca_e, expl, asv, resid = ammi(mean)
    perf = mean.mean(1)
    rank_perf = np.argsort(np.argsort(-perf)) + 1
    rank_asv = np.argsort(np.argsort(asv)) + 1
    ysi = rank_perf + rank_asv
    names = ["G%d" % (i + 1) for i in range(mean.shape[0])]

    fig = plt.figure(figsize=(W, 118 * MM))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.36, left=0.07, right=0.98,
                          top=0.915, bottom=0.085)

    ax = fig.add_subplot(gs[0, 0])
    for i in range(mean.shape[0]):
        ax.plot(env_T, 100 / (1 + np.exp(-mean[i])), lw=0.9,
                color=plt.cm.viridis(i / (mean.shape[0] - 1)), label=names[i])
    ax.axvline(23.5, color=C["red"], lw=0.7, ls="--")
    ax.set_xlabel("mean temperature of the environment ($^\\circ$C)")
    ax.set_ylabel("pollen sterility (%)")
    ax.set_title("(a)  simulated reaction norms", loc="left")
    ax.legend(ncol=2, loc="lower right", fontsize=5.6)

    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0, color=C["grid"], lw=0.6); ax.axvline(0, color=C["grid"], lw=0.6)
    ax.scatter(perf, ipca_g[:, 0], s=20, color=C["blue"], zorder=3)
    for i, n in enumerate(names):
        ax.annotate(n, (perf[i], ipca_g[i, 0]), textcoords="offset points", xytext=(3, 3), fontsize=5.8)
    ax.scatter(mean.mean(0), ipca_e[:, 0], s=14, marker="s", color=C["orange"], zorder=3)
    ax.set_xlabel("main effect (logit sterility)")
    ax.set_ylabel("IPCA1 score (%.0f %% of the interaction)" % (100 * expl[0]))
    ax.set_title("(b)  AMMI1 biplot, entries and environments", loc="left")

    ax = fig.add_subplot(gs[0, 2])
    order = np.argsort(ysi)[::-1]
    ax.barh([names[i] for i in order], [ysi[i] for i in order], color=C["aqua"], height=0.62)
    for k, i in enumerate(order):
        ax.text(ysi[i] + 0.25, k, "ASV %.2f" % asv[i], va="center", fontsize=5.8, color=C["ink2"])
    ax.set_xlabel("yield-stability index (rank sum; lower is better)")
    ax.set_xlim(0, max(ysi) * 1.3)
    ax.set_title("(c)  ASV and YSI on the simulated set", loc="left")

    ax = fig.add_subplot(gs[1, 0])
    for nenv, col in ((4, C["orange"]), (8, C["yellow"]), (12, C["aqua"]), (20, C["blue"])):
        d = np.linspace(0, 1.4, 60)
        se_d = np.sqrt(2 * (0.55 ** 2 / 3 + 0.35 ** 2) / nenv)
        power = 1 - norm.cdf(1.96 - d / se_d) + norm.cdf(-1.96 - d / se_d)
        ax.plot(d, 100 * power, color=col, lw=1.1, label="%d environments" % nenv)
    ax.axhline(80, color=C["red"], lw=0.7, ls="--")
    ax.set_xlabel("true difference between entries (logit units)")
    ax.set_ylabel("power (%)")
    ax.set_title("(d)  power to separate two entries", loc="left")
    ax.legend(loc="lower right")

    ax = fig.add_subplot(gs[1, 1])
    k = np.arange(1, 7)
    ax.bar(k, 100 * expl[:6], color=C["violet"], width=0.6)
    ax.plot(k, 100 * np.cumsum(expl[:6]), color=C["orange"], marker="o", ms=3, lw=1.0)
    ax.set_xlabel("interaction principal component")
    ax.set_ylabel("variance explained (%)")
    ax.set_title("(e)  how much structure AMMI1-2 capture", loc="left")

    ax = fig.add_subplot(gs[1, 2])
    sd_levels = np.linspace(0.2, 1.2, 40)
    detect = []
    for sd in sd_levels:
        se_d = np.sqrt(2 * (sd ** 2 / 3 + 0.35 ** 2) / 12)
        detect.append(1.96 * se_d)
    ax.plot(sd_levels, detect, color=C["blue"], lw=1.2)
    ax.set_xlabel("within-environment residual SD (logit units)")
    ax.set_ylabel("smallest detectable difference (logit)")
    ax.set_title("(f)  resolution of a 12-environment network", loc="left")

    fig.suptitle("Simulated multi-environment data, generated by the deposited code: no panel reports an observation "
                 "from a real sterile line or a real nanomaterial treatment",
                 fontsize=6.8, color=C["red"], y=0.995)
    save(fig, "Fig10")
    return dict(expl=expl[:3].round(3).tolist(), asv=asv.round(2).tolist(),
                ysi=ysi.tolist(), perf=perf.round(2).tolist())


if __name__ == "__main__":
    fig7(); fig9(); print(fig10())
