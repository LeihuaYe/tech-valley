"""
The Variance Tax — code + figures for Tech Valley issue 1.

Each "# ---- block N" section is one simulation; plotting code follows it.

  1. The variance tax: required n scales linearly with sigma^2
  2. CUPED: var(Y) vs var(Y_CUPED)
  3. Regression adjustment: confidence-interval shrinkage
  4. Trap 1: a post-treatment covariate collapses the effect
  5. Trap 2: chance imbalance (CUPED fixes it) vs a pre-window that overlaps launch
  6. Trap 3: a thin pre-period shrinks the gain; it does not add noise
  7. Trap 4: stale covariate (rho -> 0) -> no variance reduction

Run:  python analysis.py        (numpy, scipy, matplotlib; seeded)
Out:  figures/01..07.png and the numbers quoted in the post
"""

from pathlib import Path

import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 11})
BLUE, RED, GREEN = "#4c72b0", "#c44e52", "#55a868"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


# ---- block 1
import numpy as np
from scipy import stats

def required_n(sigma, mde, alpha=0.05, power=0.8):
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return 2 * ((z_a + z_b) * sigma / mde) ** 2

print(f"sigma^2 = 4.0 -> n = {required_n(2.0, 0.1):,.0f} per arm")
print(f"sigma^2 = 2.0 -> n = {required_n(np.sqrt(2), 0.1):,.0f} per arm")
# ---- end

v = np.linspace(0.25, 9, 100)
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(v, required_n(np.sqrt(v), 0.1), linewidth=2, color=BLUE)
for var, c in ((4.0, RED), (2.0, GREEN)):
    n_ = required_n(np.sqrt(var), 0.1)
    ax.plot([var], [n_], "o", color=c)
    ax.annotate(f"σ² = {var:.0f}: {n_:,.0f} per arm", (var, n_), textcoords="offset points",
                xytext=(12, -4), color=c)
ax.set_xlabel("Outcome variance (σ²)")
ax.set_ylabel("Required users per arm")
ax.set_title("Required sample size is proportional to variance\n(MDE 0.1, α 0.05, power 80%)")
ax.grid(alpha=0.3)
save(fig, "01_variance_tax.png")

print()
# ---- block 2
rng = np.random.default_rng(1)
n, rho = 100_000, 0.7
X = rng.normal(size=n)                                   # pre-period metric
Y = rho * X + np.sqrt(1 - rho**2) * rng.normal(size=n)   # in-experiment metric

theta = np.cov(X, Y)[0, 1] / np.var(X, ddof=1)
Y_cuped = Y - theta * X
print(f"var(Y)       = {np.var(Y):.3f}")
print(f"var(Y_CUPED) = {np.var(Y_cuped):.3f}")
print(f"reduction    = {1 - np.var(Y_cuped) / np.var(Y):.1%}   (theory: rho^2 = {rho**2:.0%})")
# ---- end

fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
bins = np.linspace(-3.6, 3.6, 61)
ax.hist(Y_cuped, bins=bins, color="#d1495b", alpha=0.85, linewidth=0)
ax.hist(Y, bins=bins, histtype="step", linewidth=2, color="#2e4057")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#888888")
ax.tick_params(axis="y", length=0, colors="#666666")
ax.tick_params(axis="x", colors="#444444")
ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
ax.set_axisbelow(True)
ax.set_xlim(-3.6, 3.6)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0"))
ax.set_xlabel("Metric value (standardized)", color="#444444")
ax.set_ylabel("Users", color="#666666")
ax.annotate(f"After CUPED\nvariance {np.var(Y_cuped):.2f}", xy=(0.55, 5600), xytext=(1.25, 6300),
            color="#d1495b", fontsize=11, fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="-", color="#d1495b", linewidth=1))
ax.annotate(f"Raw metric Y\nvariance {np.var(Y):.2f}", xy=(-1.45, 1900), xytext=(-3.4, 4300),
            color="#2e4057", fontsize=11, fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="-", color="#2e4057", linewidth=1))
fig.text(0.02, 0.97, f"CUPED cuts the metric's variance by {1 - np.var(Y_cuped) / np.var(Y):.0%}",
         fontsize=14, fontweight="bold", color="#222222", va="top")
fig.text(0.02, 0.90, f"100,000 simulated users, pre-period correlation ρ = {rho}; theory predicts ρ² = {rho**2:.0%}",
         fontsize=10.5, color="#666666", va="top")
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(FIG_DIR / "02_cuped_histogram.png", facecolor="white")
plt.close(fig)

print()
# ---- block 3
def ols(y, *cols):
    """OLS with an intercept. Returns (coefficients, standard errors)."""
    A = np.column_stack([np.ones(len(y)), *cols])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    cov = resid @ resid / (len(y) - A.shape[1]) * np.linalg.inv(A.T @ A)
    return beta, np.sqrt(np.diag(cov))

rng = np.random.default_rng(2)
n, rho, effect = 5_000, 0.6, 0.05
T = rng.integers(0, 2, n)
X = rng.normal(size=n)
Y = effect * T + rho * X + np.sqrt(1 - rho**2) * rng.normal(size=n)

_, se_raw = ols(Y, T)        # Y ~ T
_, se_adj = ols(Y, T, X)     # Y ~ T + X
print(f"95% CI width, Y ~ T     : {2 * 1.96 * se_raw[1]:.3f}")
print(f"95% CI width, Y ~ T + X : {2 * 1.96 * se_adj[1]:.3f}")
# ---- end

# Figure 3: three models. X keeps rho = 0.6 as in block 3; two more pre-treatment
# covariates (Z1, Z2) explain another 15% of Y's variance. Total var(Y) stays 1.
b1, b2 = 0.30, 0.25
noise_sd = np.sqrt(1 - rho**2 - b1**2 - b2**2)
def sim3(r):
    T_ = r.integers(0, 2, n)
    X_, Z1, Z2 = r.normal(size=(3, n))
    Y_ = effect * T_ + rho * X_ + b1 * Z1 + b2 * Z2 + noise_sd * r.normal(size=n)
    fits = [ols(Y_, T_), ols(Y_, T_, X_), ols(Y_, T_, X_, Z1, Z2)]
    return [(b[1], se[1]) for b, se in fits]

r3 = np.random.default_rng(3)
widths = np.array([[2 * 1.96 * se for _, se in sim3(r3)] for _ in range(500)]).mean(axis=0)
one = sim3(np.random.default_rng(30))
print("Figure 3 mean CI widths (500 tests):", np.round(widths, 3),
      "narrower vs Y ~ T:", [f"{1 - w / widths[0]:.0%}" for w in widths[1:]])
print("Figure 3 example test (est, lo, hi):", [(round(b, 3), round(b - 1.96 * se, 3), round(b + 1.96 * se, 3)) for b, se in one])

INK, MUTED, NAVY, CRIMSON, TEAL = "#222222", "#666666", "#2e4057", "#d1495b", "#00798c"
labels = ["Y ~ T", "Y ~ T + X", "Y ~ T + X + other covariates"]
notes = ["no adjustment", "pre-period metric (CUPED)", "plus 2 more pre-treatment predictors"]
colors = [NAVY, CRIMSON, TEAL]
fig, ax = plt.subplots(figsize=(8, 4.4), dpi=200)
ys = [2, 1, 0]
for y, (b, se), c, w, lab, note in zip(ys, one, colors, widths, labels, notes):
    ax.plot([b - 1.96 * se, b + 1.96 * se], [y, y], color=c, linewidth=6, solid_capstyle="round", alpha=0.9)
    ax.plot([b], [y], "o", color="white", markersize=6, markeredgecolor=c, markeredgewidth=2)
    ax.text(-0.04, y + 0.06, lab, fontsize=11.5, fontweight="bold", color=c, ha="right", va="bottom", clip_on=False)
    ax.text(-0.04, y - 0.06, note, fontsize=9.5, color=MUTED, ha="right", va="top", clip_on=False)
    pct = "" if w == widths[0] else f", {1 - w / widths[0]:.0%} narrower"
    ax.text(b + 1.96 * se + 0.006, y, f"width {w:.3f}{pct}", fontsize=10, color=c, va="center")
ax.plot([0, 0], [-0.5, 2.45], color="#999999", linewidth=1)
ax.plot([effect, effect], [-0.5, 2.45], color=INK, linewidth=1, linestyle=(0, (4, 3)))
ax.text(effect, 2.5, "true effect 0.05", fontsize=9.5, color=INK, ha="center", va="bottom")
ax.text(0, 2.5, "zero", fontsize=9.5, color=MUTED, ha="center", va="bottom")
ax.set_xlim(-0.03, 0.17)
ax.set_ylim(-0.6, 2.85)
ax.set_yticks([])
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#888888")
ax.tick_params(axis="x", colors="#444444")
ax.set_xlabel("Estimated treatment effect with 95% CI", color="#444444")
fig.text(0.02, 0.97, "Each added pre-treatment covariate narrows the confidence interval",
         fontsize=13.5, fontweight="bold", color=INK, va="top")
fig.text(0.02, 0.905, "One simulated test, n = 5,000, ρ(X, Y) = 0.6; widths are averages over 500 tests",
         fontsize=10, color=MUTED, va="top")
fig.subplots_adjust(left=0.36, right=0.98, top=0.8, bottom=0.15)
fig.savefig(FIG_DIR / "03_ci_shrinkage.png", facecolor="white")
plt.close(fig)

print()
# ---- block 4
rng = np.random.default_rng(3)
correct, wrong = [], []
for _ in range(500):
    T = rng.integers(0, 2, 2_000).astype(float)
    M = 0.5 * T + rng.normal(size=2_000)     # M is measured after treatment starts
    Y = 0.5 * M + rng.normal(size=2_000)     # T affects Y only through M
    correct.append(ols(Y, T)[0][1])          # Y ~ T
    wrong.append(ols(Y, T, M)[0][1])         # Y ~ T + M
print("true total effect : 0.250")
print(f"Y ~ T             : {np.mean(correct):.3f}")
print(f"Y ~ T + M         : {np.mean(wrong):.3f}")
# ---- end

fig, ax = plt.subplots(figsize=(7.5, 4))
bins = np.linspace(-0.2, 0.5, 50)
ax.hist(correct, bins=bins, color=GREEN, alpha=0.8, label=f"Y ~ T      (mean {np.mean(correct):.3f})")
ax.hist(wrong, bins=bins, color=RED, alpha=0.8, label=f"Y ~ T + M (mean {np.mean(wrong):.3f})")
ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
ax.axvline(0.25, color="black", linestyle="--", linewidth=1.2)
ax.text(0.255, ax.get_ylim()[1] * 0.92, "true effect 0.25", fontsize=9)
ax.set_xlabel("Estimated treatment effect")
ax.set_ylabel("Simulated tests")
ax.set_title("Controlling for a post-treatment variable removes the effect")
ax.legend(frameon=False, loc="upper left", fontsize=9)
save(fig, "04_trap1_post_treatment.png")

print()
# ---- block 5
rng = np.random.default_rng(4)
n, rho, effect = 2_000, 0.7, 0.10
imb, raw, adj = [], [], []
for _ in range(5_000):
    T = rng.integers(0, 2, n)
    X = rng.normal(size=n)
    Y = effect * T + rho * X + np.sqrt(1 - rho**2) * rng.normal(size=n)
    Y_cuped = Y - np.cov(X, Y)[0, 1] / np.var(X, ddof=1) * X
    imb.append(X[T == 1].mean() - X[T == 0].mean())
    raw.append(Y[T == 1].mean() - Y[T == 0].mean())
    adj.append(Y_cuped[T == 1].mean() - Y_cuped[T == 0].mean())
imb, raw, adj = map(np.array, (imb, raw, adj))

lucky = imb > np.quantile(imb, 0.9)   # treatment arm drew the most active users
print(f"all 5,000 tests     : raw {raw.mean():.3f}   CUPED {adj.mean():.3f}   (truth 0.100)")
print(f"top 10% imbalanced  : raw {raw[lucky].mean():.3f}   CUPED {adj[lucky].mean():.3f}")
# ---- end

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.scatter(imb, raw, s=4, alpha=0.25, color=BLUE, label="raw difference in means")
ax.scatter(imb, adj, s=4, alpha=0.25, color=RED, label="CUPED")
ax.axhline(effect, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Pre-period imbalance (treatment mean X − control mean X)")
ax.set_ylabel("Estimated effect (truth 0.10)")
ax.set_title("Chance imbalance moves the raw estimate. CUPED corrects it.\n"
             "(5,000 randomizations, ρ = 0.7)")
ax.legend(frameon=False, markerscale=4, loc="upper left")
save(fig, "05_trap2_imbalance.png")

print()
# ---- block 6
rng = np.random.default_rng(5)
n, effect = 200_000, 0.10
T = rng.integers(0, 2, n)
user = rng.normal(size=n)                    # each user's stable activity level
for overlap in (0, 3, 7, 14):                # days of the 14-day "pre" window after launch
    X = user + rng.normal(scale=0.7, size=n) + effect * T * overlap / 14
    Y = user + rng.normal(scale=0.7, size=n) + effect * T
    Y_cuped = Y - np.cov(X, Y)[0, 1] / np.var(X, ddof=1) * X
    est = Y_cuped[T == 1].mean() - Y_cuped[T == 0].mean()
    print(f"{overlap:>2} of 14 pre-window days after launch: CUPED {est:.3f}   (truth 0.100)")
# ---- end

# LinkedIn-only figure (not in the post): the overlap sweep from block 6, every day 0-14
rng = np.random.default_rng(55)
days, ests = np.arange(15), []
for overlap in days:                         # mean of 20 replications per point
    reps = []
    for _ in range(20):
        X = user + rng.normal(scale=0.7, size=n) + effect * T * overlap / 14
        Y = user + rng.normal(scale=0.7, size=n) + effect * T
        Y_cuped = Y - np.cov(X, Y)[0, 1] / np.var(X, ddof=1) * X
        reps.append(Y_cuped[T == 1].mean() - Y_cuped[T == 0].mean())
    ests.append(np.mean(reps))
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.axhline(effect, color="grey", linestyle="--", linewidth=1)
ax.text(0.2, effect + 0.004, "true effect 0.10", fontsize=9, color="grey")
ax.plot(days, ests, "o-", color=RED, markersize=4)
ax.set_ylim(0, 0.12)
ax.set_xlabel("Days of the 14-day \"pre-period\" that fall after launch")
ax.set_ylabel("CUPED estimate")
ax.set_title("A covariate window that overlaps launch eats the effect")
ax.grid(alpha=0.3)
save(fig, "08_launch_overlap.png")

print()
# ---- block 7
rng = np.random.default_rng(6)
n = 50_000
level = rng.normal(size=n)                             # each user's typical daily activity

def daily_mean(days):
    return (level[:, None] + rng.normal(scale=3, size=(n, days))).mean(axis=1)

Y = daily_mean(14)                                     # 14-day experiment metric
for d in (1, 3, 7, 14, 28):
    X = daily_mean(d)                                  # d-day pre-period
    theta = np.cov(X, Y)[0, 1] / np.var(X, ddof=1)
    r = np.corrcoef(X, Y)[0, 1]
    print(f"{d:>2}-day pre-period: rho {r:.2f}, variance reduction {1 - np.var(Y - theta * X) / np.var(Y):.1%}")
# ---- end

ds = np.arange(1, 43)
red = []
for d in ds:
    X = daily_mean(d)
    red.append(1 - np.var(Y - np.cov(X, Y)[0, 1] / np.var(X, ddof=1) * X) / np.var(Y))
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(ds, red, "o-", markersize=3, linewidth=1.5, color=BLUE)
ax.axhline(0, color="grey", linewidth=1)
ax.axvline(14, color="grey", linestyle=":", linewidth=1)
ax.text(14.5, 0.02, "pre-period = experiment length", fontsize=9, color="grey")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v_, _: f"{v_:.0%}"))
ax.set_ylim(-0.03, 0.55)
ax.set_xlabel("Pre-period length (days)")
ax.set_ylabel("Variance reduction")
ax.set_title("A short pre-period gives a smaller gain, not a loss\n(14-day test, noisy daily activity)")
ax.grid(alpha=0.3)
save(fig, "06_trap3_pre_period.png")

print()
# ---- block 8
rng = np.random.default_rng(7)
n = 20_000
for rho in (0.1, 0.3, 0.5, 0.7):
    X = rng.normal(size=n)
    Y = rho * X + np.sqrt(1 - rho**2) * rng.normal(size=n)
    theta = np.cov(X, Y)[0, 1] / np.var(X, ddof=1)
    print(f"rho = {rho}: variance reduction {1 - np.var(Y - theta * X) / np.var(Y):.1%}")
# ---- end

rhos = np.linspace(0, 0.9, 19)
emp = []
for r_ in rhos:
    X = rng.normal(size=n)
    Y = r_ * X + np.sqrt(1 - r_**2) * rng.normal(size=n)
    emp.append(1 - np.var(Y - np.cov(X, Y)[0, 1] / np.var(X, ddof=1) * X) / np.var(Y))
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.axhspan(0, 0.10, color=RED, alpha=0.12)
ax.text(0.02, 0.04, "under 10% gain", color=RED, fontsize=9)
ax.plot(rhos, rhos**2, linestyle="--", color="grey", label="theory: ρ²")
ax.plot(rhos, emp, "o", color=BLUE, label="simulated")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v_, _: f"{v_:.0%}"))
ax.set_xlabel("Correlation between pre-period covariate and outcome (ρ)")
ax.set_ylabel("Variance reduction")
ax.set_title("Variance reduction is ρ²: weak covariates buy almost nothing")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.85))
ax.grid(alpha=0.3)
save(fig, "07_trap4_stale_covariate.png")
