"""
Engineering skill check: plot q+, q-, F, G for the power (PowStep) and
exponential (ExpStep) response families at varying gamma_res.

Grounding (aihwkit 1.1.0 device models, see configs/devices.py):
  Normalized weight w in [-1, 1], bounds b_max=+1, b_min=-1.
  omega(w) = (b_max - w) / (b_max - b_min) = (1 - w) / 2   (distance to upper bound)

  PowStepDevice (power family):
      q+(w) = omega^gamma_plus            (up-direction step size)
      q-(w) = (1 - omega)^gamma_minus     (down-direction step size)

  ExpStepDevice (exponential family):
      z(w)  = 2 * a_es * w / (b_max - b_min) + b_es = a_es * w + b_es
      q+(w) = max(1 - A * exp(+gamma_plus  * z(w)), 0)
      q-(w) = max(1 - A * exp(-gamma_minus * z(w)), 0)

Response-function decomposition (Wu et al. 2025, eq. 6):
      F(w) = (q+(w) + q-(w)) / 2      symmetric component
      G(w) = (q-(w) - q+(w)) / 2      asymmetric component
  Symmetric point W_sp is the weight where G(W_sp) = 0.

gamma_res is the up/down asymmetry knob:
      gamma_plus  = gamma0 * (1 + gamma_res)
      gamma_minus = gamma0 * (1 - gamma_res)
  gamma_res = 0 gives a device whose two directions share the same exponent.
  Raising gamma_res pushes the symmetric point off w = 0, which is exactly the
  drift E-RIDER has to track and the shape our drift model will make time-varying.
"""

import numpy as np
import matplotlib.pyplot as plt

GAMMA_RES = [0.0, 0.1, 0.2, 0.3]          # same sweep values as the S1 c_Lin figure
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
W = np.linspace(-0.99, 0.99, 400)


def power_responses(w, gamma_res, gamma0=2.0):
    omega = (1.0 - w) / 2.0
    gp = gamma0 * (1.0 + gamma_res)
    gm = gamma0 * (1.0 - gamma_res)
    q_plus = omega ** gp
    q_minus = (1.0 - omega) ** gm
    return q_plus, q_minus


def exp_responses(w, gamma_res, gamma0=1.0, a_es=1.0, b_es=0.0, A=0.25):
    z = a_es * w + b_es
    gp = gamma0 * (1.0 + gamma_res)
    gm = gamma0 * (1.0 - gamma_res)
    q_plus = np.clip(1.0 - A * np.exp(gp * z), 0.0, None)
    q_minus = np.clip(1.0 - A * np.exp(-gm * z), 0.0, None)
    return q_plus, q_minus


def symmetric_point(w, G):
    """First sign change of G along w -> linear-interpolated root."""
    sign = np.sign(G)
    idx = np.where(np.diff(sign) != 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    w0, w1, g0, g1 = w[i], w[i + 1], G[i], G[i + 1]
    return w0 - g0 * (w1 - w0) / (g1 - g0)


families = [("Power (PowStep)", power_responses), ("Exponential (ExpStep)", exp_responses)]
cols = ["q+ (up step)", "q- (down step)", "F = (q+ + q-)/2", "G = (q- - q+)/2"]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for r, (fam_name, fn) in enumerate(families):
    for gr, color in zip(GAMMA_RES, COLORS):
        qp, qm = fn(W, gr)
        F = (qp + qm) / 2.0
        G = (qm - qp) / 2.0
        wsp = symmetric_point(W, G)
        label = f"gamma_res={gr}"
        axes[r, 0].plot(W, qp, color=color, label=label)
        axes[r, 1].plot(W, qm, color=color, label=label)
        axes[r, 2].plot(W, F, color=color, label=label)
        axes[r, 3].plot(W, G, color=color, label=label)
        if wsp is not None:
            axes[r, 3].plot(wsp, 0.0, "o", color=color, markersize=7)
    for c in range(4):
        axes[r, c].axhline(0, color="0.7", lw=0.8)
        axes[r, c].axvline(0, color="0.7", lw=0.8)
        axes[r, c].set_xlabel("weight w (normalized)")
        if r == 0:
            axes[r, c].set_title(cols[c])
        axes[r, c].set_ylabel(fam_name.split()[0] if c == 0 else "")
    axes[r, 3].legend(fontsize=8, loc="best")

fig.suptitle(
    "Response functions q+, q-, F, G for power and exponential devices, swept over gamma_res\n"
    "dots on the G panels mark the symmetric point (G = 0); it slides off w=0 as asymmetry grows",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("skill_check_responses.png", dpi=150)
print("saved skill_check_responses.png")

# print the symmetric-point shift table
print("\nSymmetric point W_sp (root of G):")
for fam_name, fn in families:
    print(f"  {fam_name}")
    for gr in GAMMA_RES:
        qp, qm = fn(W, gr)
        G = (qm - qp) / 2.0
        wsp = symmetric_point(W, G)
        print(f"    gamma_res={gr}:  W_sp = {wsp:.4f}" if wsp is not None else f"    gamma_res={gr}:  no root in range")
