
# QCTPF / QODER Missile Guidance Project — Full Working Summary

Author: Soumyadip Chakrabarti
Date: 2025-12-26

This document captures, in detail, the evolution of the project from Phase 1 to Phase 6.6.
It is intended to be pasted into a new chat to continue work without losing context.

---------------------------------------------------------------------
PROJECT GOAL (Initial)
---------------------------------------------------------------------

Build a hybrid Quantum–Classical guidance and tracking system where:
- A Classical Coordinated-Turn Particle Filter (CTPF) tracks a maneuvering target.
- A Quantum Neural / QNEAT-evolved circuit (QODER) learns residual dynamics.
- The quantum model augments the PF to improve prediction under high-G maneuvers.

Baseline: Classical CTPF only.
Target: Reduce miss distance and PF error under aggressive maneuvers.

---------------------------------------------------------------------
PHASE 1 — Classical Baseline (CTPF)
---------------------------------------------------------------------

Implemented:
- Target dynamics: position + velocity, coordinated turn.
- Missile dynamics: PN-like guidance.
- Particle filter:
    State: [x, y, z, vx, vy, vz]
    Predict: coordinated turn model.
    Update: measurement likelihood.
    Resample: systematic.

Files:
- ctpf.py
- target_dynamics.py
- missile_dynamics.py
- STEP4_High_maneuver.py

Outcome:
- CTPF gives reasonable performance on smooth maneuvers.
- Under high-G or jinking, PF error grows but remains stable.

This became the performance baseline.

---------------------------------------------------------------------
PHASE 2 — Quantum Model Infrastructure (QODER / QNEAT)
---------------------------------------------------------------------

Added:
- Genome encoding quantum circuits.
- build_circuit_from_genome(genome)
- Qiskit Aer simulator backend.
- Measurement decoding to continuous outputs.

Goal:
Evolve circuits that map classical features → corrections.

Outcome:
- Infrastructure ready.
- No coupling yet to PF.

---------------------------------------------------------------------
PHASE 3 — Feature Engineering
---------------------------------------------------------------------

Designed classical features:

From geometry:
- closing_velocity
- azimuth
- azimuth_rate (LOS turn-rate proxy)
- thermal_proxy = 1 / r^2

Later extended to:
- omega_classical (turn-rate)

Functions:
- build_training_features() in training.
- _build_features_for_particle() in QCTPF.

Normalized features:
    f = f / ||f||

Important: Training and inference feature pipelines must match.

---------------------------------------------------------------------
PHASE 4 — QODER Training Loop
---------------------------------------------------------------------

File:
- STEP5_train_qoder_high_g.py

Fitness:
- Multiple episodes.
- Scenario generator with increasing maneuver difficulty.
- For each step:
    - Build features.
    - Bind circuit parameters.
    - Run quantum shots.
    - Decode avg_values.
    - Compute error vs target proxy.
- Genome fitness = 1 / (1 + mean_error).

Early target:
- Predict acceleration or LOS behavior.

Outcome:
- Circuits converge to stable but generic outputs.
- Typically produce mid-range values ~0.5 → mapped to ~10–15 m/s^2.

---------------------------------------------------------------------
PHASE 5 — First Quantum Injection (Acceleration Residual)
---------------------------------------------------------------------

In QCTPF.predict():

Steps:
1) Call super().predict(dt) for classical PF.
2) For each particle:
    - Build features.
    - Run quantum circuit.
    - Decode first 3 qubits → a_q.
3) Apply:
    v += alpha_q * a_q * dt

Code snippet:
    a_q = (avg_values[0:3] - 0.5) * 20.0
    dv = alpha_q * a_q * dt
    self.particles[i, 3:6] += dv

Outcome:
- QCTPF consistently worse than CTPF.
- Large alpha → divergence.
- Small alpha → negligible effect.

Conclusion:
Unconstrained acceleration residual destabilizes PF.

---------------------------------------------------------------------
PHASE 6.1 — LOS Turn-Rate Learning
---------------------------------------------------------------------

Idea:
Instead of full acceleration, learn LOS turn-rate omega.

Training target:
- az_rate from geometry.

Quantum output:
    pred_omega = (avg_values[0] - 0.5) * 0.6

Used as residual or proxy.

Outcome:
- Model predicts near-constant values.
- No strong correlation to actual turn dynamics.

---------------------------------------------------------------------
PHASE 6.4 — Direct Omega Prediction
---------------------------------------------------------------------

Changed loss to:
    omega_error = |pred_omega - true_omega|

Goal:
Make quantum learn omega directly.

Outcome:
- Training converges.
- But inference still injects noisy constant residuals.
- QCTPF worse than CTPF.

---------------------------------------------------------------------
PHASE 6.5 — Gated Residual Injection
---------------------------------------------------------------------

Only apply quantum when:
    |omega_classical| > threshold

Code:
    if abs(omega_classical) > 0.05:
        self.particles[i,6] += alpha_q * omega_residual_q

Also added omega into particle state in some attempts.

Issues:
- Dimension mismatches (index 6 errors).
- Inconsistent state definition caused bugs.
- Even when fixed, performance degraded.

Outcome:
- Gating reduces frequency but not harm.
- Still worse than classical.

---------------------------------------------------------------------
PHASE 6.6 — Innovation Gating
---------------------------------------------------------------------

Idea:
Only apply quantum when PF appears inconsistent.

Innovation metric:
    rel_pos = particles[:,0:3] - ownship
    ranges = ||rel_pos||
    innovation = std(ranges)

If innovation > threshold → quantum active.

Code:
    if innovation < INNOV_THRESHOLD:
        return

Then run quantum batch and inject acceleration residuals.

Logging:
    print(f"[QCTPF] innov={innovation:.1f}, |a_q|={np.linalg.norm(a_q):.3f}")

Observed:
- Innovation starts ~100 and grows to >3000.
- Quantum fires every step.
- |a_q| ≈ 10–15 m/s^2 constantly.
- Positive feedback → divergence.

Results at 600 steps:
- CTPF miss ≈ 20 km.
- QCTPF miss ≈ 24 km.
- PF error ≈ 10 km.

Outcome:
❌ Innovation gating failed.
Quantum residuals amplify divergence.

---------------------------------------------------------------------
KEY OBSERVATIONS
---------------------------------------------------------------------

1) Quantum model outputs are nearly constant after training.
2) No directional coupling between:
   - error direction,
   - and injected correction.
3) Scalar innovation cannot guide vector corrections.
4) PF does NOT know truth — it cannot reset itself.
5) Any wrong correction accumulates → divergence.

Hence:
Unconstrained residual learning is fundamentally unstable here.

---------------------------------------------------------------------
FINAL CONCLUSION
---------------------------------------------------------------------

- Hybrid QCTPF system was successfully implemented.
- Quantum models were trained and integrated.
- However, across multiple strategies (Phases 5–6.6),
  quantum corrections consistently degraded performance vs classical CTPF.

This demonstrates:
✔ Feasibility of integration.
✔ Limits of naive quantum residual learning.
✔ Importance of physics-constrained hybrid designs.

Scientifically valid negative result.

---------------------------------------------------------------------
RECOMMENDED NEXT DIRECTION
---------------------------------------------------------------------

Instead of more phases:
- Freeze CTPF as baseline.
- Perform ablation:
    alpha_q = 0, tiny, moderate.
- Quantify degradation.
- Write analysis.

Optionally explore:
- Directional correction (along LOS only).
- Or use quantum only to estimate noise / covariance, not dynamics.

---------------------------------------------------------------------
END OF SUMMARY
---------------------------------------------------------------------
