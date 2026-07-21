# Drift Model Spec v1

Owners: Hamzah, Jun. Board deadline 7/28. Team sign-off 7/30. No experiments run before sign-off.

Status: draft for review. Every parameter range below carries a citation. Open decisions that need a team call are collected in Section 6.

## 0. What this spec is and is not

This defines time-varying response functions q+(w, t) and q-(w, t): the functional forms, which parameters drift, the law each one drifts by, and the cited range for every parameter. It is the ground truth for what "drift" means in every downstream experiment.

It does not cover the implementation in drift_model/ (that is the 8/2 task, Hamzah and Adi) or the identifiability analysis (Week 4). It fixes the spec those tasks build against.

Positioning against our own claim: Wu et al. 2025 assume the response functions are fixed. Xiao et al. 2026 (E-RIDER) let the symmetric point be unknown and track it online, but still assume the response shape itself never changes. This spec is the object that breaks that assumption. Everything here exists to make q+ and q- functions of time, so we can show SP-only tracking fails when the shape moves.

## 1. Static response model (recap from the skill check)

Normalized weight w in [-1, 1], bounds b_max = +1, b_min = -1. For a positive pulse the conductance changes by dw_min * q+(w), for a negative pulse by -dw_min * q-(w), with q+ and q- the response functions (Wu et al. 2025, eq. 6; aihwkit PowStep and ExpStep device models).

Power family (PowStep). With omega = (1 - w)/2 the distance to the upper bound:
    q+(w) = omega^(gamma_plus),   q-(w) = (1 - omega)^(gamma_minus)

Exponential family (ExpStep). With z = a_es * w + b_es:
    q+(w) = max(1 - A * exp(+gamma_plus * z), 0),   q-(w) = max(1 - A * exp(-gamma_minus * z), 0)

Decomposition (Wu et al. 2025, eq. 6):
    F(w) = (q+(w) + q-(w)) / 2      symmetric component
    G(w) = (q-(w) - q+(w)) / 2      asymmetric component
Symmetric point W_sp is the root of G. The asymmetry knob is gamma_res, entering as gamma_plus = gamma0(1 + gamma_res), gamma_minus = gamma0(1 - gamma_res). At gamma_res = 0 the two directions share an exponent. The skill-check figure shows the power-family SP sliding from 0 to -0.21 as gamma_res goes 0 to 0.3, while the exponential-family SP stays pinned at 0.

## 2. Making the response time-varying

Principle: keep the functional form from Section 1, let its parameters drift with a clock variable. So q+(w) becomes q+(w; theta(t)) where theta is the parameter vector (gamma_plus, gamma_minus, w_max, an amplitude scale a, per-cell noise).

Two clocks, because the physics runs on two different timescales:
- Wall-clock time t: retention and structural relaxation. This is the PCM t^-nu drift. It runs whether or not the device is being updated.
- Cumulative pulse count n: cycling and wear. This runs only when the device is updated. Training is millions of updates, so n grows fast.

Both map to the training step k (Section 3). We model three drift processes, the three the board named.

### 2.1 PCM conductance drift (power law in wall-clock time)

    g(t) = g(t_c) * (t / t_c)^(-nu)

where g is the stored conductance, t_c the time of the last programming pulse, nu the drift exponent (Le Gallo and Sebastian 2020; aihwkit PCM statistical model). Drift shrinks conductance magnitude over time, which compresses the usable dynamic range and reshapes the response near the bounds. In our model it multiplies the response amplitude, a(t) = a0 (t/t_c)^(-nu), pulling F down and, because the two directions saturate at different rates, moving G.

Parameter nu is state-dependent and varies cell to cell. The aihwkit model, calibrated on an array of 1 million IBM PCM devices, samples nu from N(mu_nu, sigma_nu) with:
    mu_nu    = min(max(-0.0155 ln(g_T) + 0.0244, 0.049), 0.1)     range [0.049, 0.1]
    sigma_nu = min(max(-0.0125 ln(g_T) - 0.0059, 0.008), 0.045)   range [0.008, 0.045]
(Nandakumar et al. 2019; Joshi et al. 2020). Broader amorphous-phase literature reports nu in 0.1 to 0.15 (Le Gallo and Sebastian 2020). Physical origin is collective structural relaxation of the amorphous phase (Le Gallo et al. 2018).

### 2.2 ReRAM cycling degradation (monotone in pulse count n)

As update pulses accumulate, the response nonlinearity and dynamic range degrade (Chen et al., Applied Physics Reviews 7, 011301, 2020). We model this as the asymmetry exponent growing and the usable range shrinking with cycle count:
    gamma_res(n) = gamma_res0 + kappa * (n / N_ref)
    w_max(n)     = w_max0 * (1 - eta * (n / N_end))

Both push the response toward a more asymmetric, more compressed shape as training proceeds, which is exactly the SP-and-shape drift E-RIDER cannot follow.

Ranges. Analog RRAM endurance reaches about 1e11 cycles, enough to train networks from MNIST to ImageNet, and weak-pulse incremental switching (the regime used in training) extends endurance more than five orders of magnitude over full-window switching (ASU, Characterizing Endurance Degradation of Incremental Switching in Analog RRAM, IEEE 2019; some bilayer stacks reach 1e12 in binary mode). Intermediate analog levels degrade earlier than the binary endurance number suggests (Endurance and Retention Degradation of Intermediate Levels in Filamentary Analog RRAM, 2019). For a concrete device we anchor on the RRAM-RfO2 preset with 4 to 5 states, strong device-to-device mismatch, and cycle-to-cycle writing noise (Gong et al., IEDM 2022), the same preset Xiao et al. use. So N_end sits in 1e9 to 1e11, kappa and eta are the knobs we sweep, both cited to "degrades during cycling" rather than a single hard number, which is honest for v1.

### 2.3 Per-cell variance (stochastic, per device, may grow with n)

Every cell has its own nu, its own gamma_res, and its own cycle-to-cycle update noise. We model per-cell parameters as draws, per_cell = mean + N(0, sigma), and per-update discretization noise as Var[b_k] = Theta(alpha * dw_min) (Li et al. 2025; Xiao et al. 2026, Assumption 3.4).

Ranges bracket a well-behaved device and a rough one. ECRAM, the clean end, shows cycle-to-cycle variation of 2.3 percent (sigma/mu) spatial-temporal, potentiation and depression cycle-to-cycle under 6.5 and 9.3 percent, and device-to-device about 12 percent (open-loop ECRAM array, Nature Communications 2023; Nature Communications 2025). PCM programming noise, the write-error floor, is sigma_prog = max(-1.1731 g_T^2 + 1.965 g_T + 0.2635, 0) (aihwkit; Nandakumar et al. 2019). RRAM under the Gong 2022 preset is worse than both. We sweep per-cell sigma from the ECRAM floor to the RRAM ceiling.

## 3. Mapping the clocks to the training step

Let k be the training step. Wall-clock time t = t_c + k * t_step, with t_step the seconds per minibatch. Cumulative pulses n = k * B * BL, with batch size B = 64 and average pulse length BL = 5 (Xiao et al. 2026, Fig. 4). We choose t_step and the normalizers N_ref, N_end so the total drift over a 30 to 80 epoch run is physically realistic, not a cartoon. This mapping is a decision that needs sign-off (Section 6).

## 4. Parameter table

| Symbol | Meaning | Family | Clock | Range | Source |
|---|---|---|---|---|---|
| nu | PCM drift exponent | PCM | wall-clock t | mean 0.049 to 0.1, per-cell std 0.008 to 0.045; amorphous 0.1 to 0.15 | Nandakumar 2019; Joshi 2020; Le Gallo and Sebastian 2020 |
| t_c | last-programming time (drift reference) | PCM | wall-clock t | device and schedule dependent | Le Gallo et al. 2018 |
| sigma_prog | PCM programming noise | PCM | at write | max(-1.1731 g^2 + 1.965 g + 0.2635, 0) | Nandakumar 2019; aihwkit |
| gamma_res0 | initial response asymmetry | power/exp | static | 0.0 to 0.3 (same sweep as S1 c_Lin) | Wu et al. 2025; skill check |
| kappa | asymmetry growth per cycle | RRAM | pulses n | swept; anchored to "degrades during cycling" | Chen et al. APR 2020; ASU 2019 |
| eta, N_end | dynamic-range loss and endurance | RRAM | pulses n | N_end 1e9 to 1e11; eta swept | ASU 2019; Gong et al. 2022 |
| sigma_d2d | device-to-device parameter spread | all | static per cell | 12 percent (ECRAM) to RRAM preset | Nat. Comm. 2023; Gong et al. 2022 |
| sigma_c2c | cycle-to-cycle update noise | all | pulses n | 2.3 to 9.3 percent (ECRAM) up | Nat. Comm. 2023/2025 |
| b_k | per-update discretization noise | all | pulses n | Var = Theta(alpha * dw_min) | Li et al. 2025; Xiao et al. 2026 |

## 5. Identifiability preview (feeds the Week 4 task)

Not all of these are recoverable from a realistic on-chip measurement budget. A global amplitude drift (the nu scale) is cheap to estimate from a few pulse measurements. A per-cell gamma_res shape change is expensive. The Week 4 identifiability analysis sweeps the measurement budget k and reports estimation error per parameter. Negative findings are findings: naming what cannot be recovered defines what any full solution has to assume.

## 6. Open decisions

1. Primary drift regime. This is the one to settle first, and it is not obvious. PCM t^-nu drift is classically a retention and inference problem: weights drift after programming, over seconds to years, while the device sits idle. Our setting is on-chip training, where weights are rewritten every step, so pure retention drift is partly refreshed by the training itself. The physically dominant training-time drift is cycling degradation (2.2) and variance growth (2.3), not retention. My recommendation: make cycling degradation the headline regime, keep PCM t^-nu as a secondary retention regime for runs with idle gaps. The board named all three, so this is a framing choice, not a scope cut. Hamzah should weigh in before anything runs.

2. Shape change versus SP shift as the headline. E-RIDER tracks SP location. The cleanest way to beat it is a regime where the SP barely moves but the response shape changes, so tracking the SP buys nothing. The exponential family (SP pinned, shape drifts) is a candidate headline device for exactly this reason. Decide whether the main result targets SP shift, shape change, or both.

3. Clock mapping constants. t_step, N_ref, N_end. These set how much drift a run actually sees. Pin them to a realistic value so a reviewer at a hardware venue cannot dismiss the magnitude.

## References

- Wu, Xiao, et al. Analog In-memory Training on General Non-ideal Resistive Elements: The Impact of Response Functions. arXiv 2502.06309, 2025.
- Xiao, Li, Wu, Gokmen, Chen. Dynamic Symmetric Point Tracking. arXiv 2602.21321, 2026.
- Nandakumar et al. Phase-change memory models for deep learning training and inference. IEEE ICECS, 2019.
- Joshi et al. Accurate deep neural network inference using computational phase-change memory. Nature Communications 11, 2473, 2020.
- Le Gallo, Sebastian. An overview of phase-change memory device physics. J. Phys. D 53, 213002, 2020.
- Le Gallo et al. Collective Structural Relaxation in Phase-Change Memory Devices. Adv. Electron. Mater. 4, 1700627, 2018.
- Chen et al. Reliability of analog resistive switching memory for neuromorphic computing. Applied Physics Reviews 7, 011301, 2020.
- ASU. Characterizing Endurance Degradation of Incremental Switching in Analog RRAM for Neuromorphic Systems. IEEE, 2019.
- Endurance and Retention Degradation of Intermediate Levels in Filamentary Analog RRAM, 2019.
- Gong et al. Deep learning acceleration in 14nm CMOS compatible ReRAM array. IEDM, 2022.
- Open-loop analog programmable electrochemical memory array. Nature Communications 14, 2023.
- Li, Wu, Liu, Gokmen, Chen. In-memory training on analog devices with limited conductance states via multi-tile residual learning. arXiv 2510.02516, 2025.
- aihwkit 1.1.0 PCM inference documentation.
