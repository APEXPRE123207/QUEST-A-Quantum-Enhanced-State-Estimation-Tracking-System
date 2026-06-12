import numpy as np
import matplotlib.pyplot as plt

from Simulation.target_dynamics import Target
from Simulation.missile_dynamics import Missile
from Simulation.sensor_model import Sensor
from Quantum_Core.ctpf import CTPF
from config import SimOptions
from Quantum_Core.ctpf import QCTPF   # instead of CTPF
import pickle
from future_hypothesis import FutureHypothesis
BETA_TERMINAL = 20.0
GAMMA_RANGERATE = 2.0    # start conservative
EPS = 1e-6  
lambda_u = 0.01
RISK_TAIL_FRACTION = 0.2   # 20% worst futures (start here)
WARMUP_STEPS = 10
zero_future_count = 0

# def extract_belief_tube(pf, H):
#     trajs = []
#     weights = []

#     for i in range(pf.num_particles):
#         if pf.weights[i] <= 0:
#             continue

#         # particle state: [x, y, z, vx, vy, vz]
#         x0 = pf.particles[i].copy()
#         w  = pf.weights[i]

#         X = []
#         x = x0.copy()

#         for _ in range(H):
#             x[:3] += x[3:6] * pf._dt   # constant-velocity rollout
#             X.append(x[:3].copy())

#         trajs.append(np.array(X))
#         weights.append(w)

#     if len(trajs) == 0:
#         return np.array([]), np.array([])

#     weights = np.array(weights)
#     weights /= np.sum(weights)

#     return np.array(trajs), weights



def encode_state(missile):
    return np.hstack([missile.position, missile.velocity])

def decode_state(x):
    m = Missile(
        initial_position=x[:3].copy(),
        initial_velocity=x[3:6].copy()
    )
    return m

# def rollout_ego(x0, U, dt):
#     """
#     x0 = [pos(3), vel(3)]
#     U  = [u0, u1, ..., u_{T-1}]  (each u is acceleration 3-vector)
#     """
#     p, v = x0[:3].copy(), x0[3:].copy()
#     traj = []
#     for u in U:
#         v = v + u * dt
#         p = p + v * dt
#         traj.append(p.copy())
#     return np.array(traj)  # shape (T,3)
def rollout(x0, U, dt, dynamics_fn):
    x = x0.copy()
    traj = []
    for u in U:
        x = dynamics_fn(x, u, dt)
        traj.append(x[:3].copy())  # position only for cost
    return np.array(traj)


# def belief_cost(U, x0, future_trajs, weights, alpha, dt, lam):
#     ego_traj = rollout_ego(x0, U, dt)  # (T,3)
#     H = future_trajs.shape[1]
#     J = 0.0

#     for i in range(len(weights)):
#         w = weights[i]
#         for t in range(H):
#             J += w * alpha[t] * np.linalg.norm(
#                     ego_traj[t] - future_trajs[i, t]
#                  )**2

#     # Control regularization
#     for u in U:
#         J += lam * np.linalg.norm(u)**2

#     return J
# def belief_cost(U, x0, future_trajs, weights, alpha, lambda_u, dt, dynamics_fn):
#     ego_traj = rollout(x0, U, dt, dynamics_fn)
#     cost = 0.0
#     for i, X in enumerate(future_trajs):
#         w = weights[i]
#         for t in range(len(U)):
#             err = ego_traj[t] - X[t]
#             cost += w * alpha[t] * np.dot(err, err)
#     cost += lambda_u * np.sum(U**2)
#     return cost

# def belief_cost(U, x0, future_trajs, weights, alpha, lambda_u, dt, dynamics_fn):
#     ego_traj = rollout(x0, U, dt, dynamics_fn)
#     cost = 0.0

#     for i, X in enumerate(future_trajs):
#         w = weights[i]

#         # Running cost (tube tracking)
#         for t in range(len(U)):
#             err = ego_traj[t] - X[t]
#             cost += w * alpha[t] * np.dot(err, err)

#         # Terminal proximity cost
#         terminal_err = ego_traj[-1] - X[-1]
#         cost += w * BETA_TERMINAL * np.dot(terminal_err, terminal_err)

#         # Terminal range-rate (NEW)
#         # Approximate terminal ego velocity from rollout
#         if len(ego_traj) >= 2:
#             v_m_H = (ego_traj[-1] - ego_traj[-2]) / dt
#         else:
#             v_m_H = np.zeros(3)

#         # Approximate terminal target velocity from future trajectory
#         if len(X) >= 2:
#             v_t_H = (X[-1] - X[-2]) / dt
#         else:
#             v_t_H = np.zeros(3)

#         r = ego_traj[-1] - X[-1]
#         r_norm = np.linalg.norm(r) + EPS
#         range_rate = np.dot(r, (v_m_H - v_t_H)) / r_norm

#         cost += w * GAMMA_RANGERATE * (range_rate ** 2)


#     # Control effort
#     cost += lambda_u * np.sum(U**2)
#     return cost


# def belief_cost(U, x0, future_trajs, weights, alpha, lambda_u, dt, dynamics_fn):
#     ego_traj = rollout(x0, U, dt, dynamics_fn)

#     per_future_costs = []

#     for i, X in enumerate(future_trajs):
#         w = weights[i]
#         cost_i = 0.0

#         # Running (tube-tracking) cost
#         for t in range(len(U)):
#             err = ego_traj[t] - X[t]
#             # cost_i += alpha[t] * np.dot(err, err)
#             cost_i += w * alpha[t] * np.dot(err, err)


#         # Terminal proximity
#         terminal_err = ego_traj[-1] - X[-1]
#         cost_i += BETA_TERMINAL * np.dot(terminal_err, terminal_err)

#         # Terminal range-rate
#         if len(ego_traj) >= 2:
#             v_m_H = (ego_traj[-1] - ego_traj[-2]) / dt
#         else:
#             v_m_H = np.zeros(3)

#         if len(X) >= 2:
#             v_t_H = (X[-1] - X[-2]) / dt
#         else:
#             v_t_H = np.zeros(3)

#         r = ego_traj[-1] - X[-1]
#         r_norm = np.linalg.norm(r) + 1e-6
#         range_rate = np.dot(r, (v_m_H - v_t_H)) / r_norm
#         cost_i += GAMMA_RANGERATE * (range_rate ** 2)

#         # store weighted per-future cost
#         # per_future_costs.append(w * cost_i)
#         per_future_costs.append(cost_i)

#     # ---- CVaR-lite (risk-biased aggregation) ----
#     # per_future_costs = np.array(per_future_costs)
#     # K = max(1, int(RISK_TAIL_FRACTION * len(per_future_costs)))
#     # worst_costs = np.sort(per_future_costs)[-K:]

#     # # Mean of worst-K + control effort
#     # return np.mean(worst_costs) + lambda_u * np.sum(U**2)

#     per_future_costs = np.array(per_future_costs)

#     # sort by raw cost
#     K = max(1, int(RISK_TAIL_FRACTION * len(per_future_costs)))
#     worst_idx = np.argsort(per_future_costs)[-K:]

#     # weighted mean over worst futures
#     cvar_cost = 0.0
#     weight_sum = 0.0
#     for idx in worst_idx:
#         cvar_cost += weights[idx] * per_future_costs[idx]
#         weight_sum += weights[idx]

#     cvar_cost /= (weight_sum + 1e-6)

#     return cvar_cost + lambda_u * np.sum(U**2)

def true_dynamics(x, u, dt):
    m = decode_state(x)
    m.velocity += u * dt
    m.position += m.velocity * dt
    return encode_state(m)


def time_weights(H):
    return (np.arange(1, H+1) / H)**2

# def cem_optimize(x0, future_trajs, weights, dt,
#                  T=5, H=8, iters=5, samples=200, elite_frac=0.1):

#     dim = 3 * T
#     mu = np.zeros(dim)
#     sigma = np.eye(dim) * 5.0

#     for _ in range(iters):
#         Us = np.random.multivariate_normal(mu, sigma, samples)
#         costs = []
#         alpha = time_weights(H)
#         lambda_u = 0.01
#         for j in range(samples):
#             U = Us[j].reshape(T,3)
            
#             J = belief_cost(
#                 U, x0, future_trajs, weights,
#                 alpha, lambda_u, dt, true_dynamics
#             )

#             costs.append(J)

#         costs = np.array(costs)
#         elite_idx = np.argsort(costs)[:int(elite_frac*samples)]
#         elite = Us[elite_idx]

#         mu = elite.mean(axis=0)
#         sigma = np.cov(elite.T)
#         sigma += 0.05 * np.eye(dim)   # exploration floor
#     return mu.reshape(T,3)

def run_qctpf_guidance_viewer():
    sim_opts = SimOptions()
    dt = sim_opts.dt
    num_steps = 300

    target = Target(
        initial_position=[12000.0, 0.0, 10000.0],
        initial_velocity=[-250.0, 40.0, 0.0]
    )

    missile = Missile(
        initial_position=[0.0, 0.0, 9000.0],
        initial_velocity=[300.0, 0.0, 0.0]
    )

    sensor = Sensor(
        radar_noise_std={'range': 50.0, 'velocity': 5.0, 'azimuth': 0.005},
        irst_noise_std=0.1
    )

    # Load champion genome
    with open("best_qneat_dz_genome.pkl", "rb") as f:
        champion = pickle.load(f)

    num_particles = 500
    pf = QCTPF(num_particles=num_particles,
           trained_genome=champion,
           sensor_model=sensor)


    init_state = np.concatenate([target.position, target.velocity])
    pf.initialize_swarm(init_state, pos_sigma=100.0, vel_sigma=30.0, omega_sigma=0.03)

    target_hist = []
    missile_hist = []
    est_hist = []
    pf_pos_errors = []
    guid_state = None          # [x,y,z,vx,vy,vz]
    smooth_alpha = 0.1         # 0<alpha<=1; smaller = more smoothing
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    target_line, = ax.plot([], [], [], 'r-', label='Target (truth)')
    missile_line, = ax.plot([], [], [], 'b-', label='Missile')
    est_line,    = ax.plot([], [], [], 'g--', label='QCTPF Estimate')
    # --- Start / End markers ---
    t_start = ax.scatter([], [], [], c='red',   s=60, marker='o')
    t_end   = ax.scatter([], [], [], c='red',   s=80, marker='x')

    m_start = ax.scatter([], [], [], c='blue',  s=60, marker='o')
    m_end   = ax.scatter([], [], [], c='blue',  s=80, marker='x')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("QCTPF Guidance Viewer")
    ax.legend()
    ax.set_xlim(-2000, 13000)
    ax.set_ylim(-4000, 4000)
    ax.set_zlim(8000, 12000)
    collapsed = False
    # frozen_state = None
    x_est = np.hstack([target.position, target.velocity])  # safe initial guess
    frozen_state = x_est.copy()

    for step in range(num_steps):
        target.update(dt, ('turn', {'g_force': 0.5}))
        obs = sensor.observe(target, missile.position)

        # pf.predict(dt, missile.position, alpha_q=0.1)
        # pf.generate_measurement_futures(H=10, M=5)
        # if step % 20 == 0:
        #     print("Mean |Δz_pred|:", np.mean(np.abs(pf.last_dz_pred)))
        # pf.predict(dt, missile.position)
        # obs_position = target.position.copy()
        # z_obs = sensor.observe(target, missile.position)
        # z_obs = sensor.observe(target, missile.position)

        r = obs['range']
        az = obs['azimuth']

        # Assume same altitude as target estimate or missile (planar radar)
                # Use estimated altitude, not missile altitude
        


        # alive = sum(len(f) for f in pf.futures)
        # print(f"step {step}: surviving futures = {alive}")
        # if alive == 0 and not collapsed:
        #     collapsed = True
        #     frozen_state = pf.estimate_state().copy()
        # if not collapsed:
        #     pf.generate_measurement_futures(H=10, M=5)
        #     pf.validate_futures(obs_position, threshold=500.0)
        # if not collapsed:
        #     pf.generate_measurement_futures(H=10, M=5)
        #     # pf.validate_futures(obs_position, threshold=700.0)
        #     # pf.validate_futures(z_pos, threshold=500.0)  # threshold in σ-units
        #     gate = max(500.0, 0.1 * r)
        #     if step < WARMUP_STEPS:
        #         pf.validate_futures(z_pos, threshold=3 * gate)
        #     else:
        #         pf.validate_futures(z_pos, threshold=gate)


        # 1. Predict
        pf.predict(dt, missile.position)

        # 2. Update PF with measurement FIRST
        if not collapsed:
            pf.update(obs, missile.position)
            if pf.effective_sample_size() < 0.5 * pf.num_particles:
                pf.resample()
            # x_est = pf.estimate_state()
            x_pf = pf.estimate_state()

            # Trust PF for lateral motion (observable)
            x_est = x_pf.copy()
            x_est_prev = x_est.copy()
            # Freeze Z using kinematic propagation (not PF)
            x_est[2] = x_est_prev[2] + x_est_prev[5] * dt   # z = z + vz·dt
            x_est[5] = 0.0                      # vz constant


        if not collapsed:
            z_est = x_est[2]
            z_pos = np.array([
                missile.position[0] + r * np.cos(az),
                missile.position[1] + r * np.sin(az),
                z_est
            ])

        # 3. Now generate futures from UPDATED belief
        if not collapsed:
            gate = np.clip(0.1 * r, 800.0, 3000.0)
            pf.generate_measurement_futures(H=8, M=5)
            pf.validate_futures(z_pos, threshold=gate)


       # Count surviving futures
        alive = sum(len(f) for f in pf.futures)
        print(f"step {step}: surviving futures = {alive}")

        # Current range based on latest estimate
        

        # --- COLLAPSE DETECTION (with persistence handled outside if desired) ---
        if alive == 0:
            zero_future_count += 1
        else:
            zero_future_count = 0
        if ( zero_future_count >= 3
            and not collapsed
            and step > WARMUP_STEPS
            and range_to_target < 6000.0
        ):
            collapsed = True
            frozen_state = x_est.copy()

        # --- STATE SELECTION ---
        if collapsed:
            x_est = frozen_state.copy()
        # else: x_est is already correct
        range_to_target = np.linalg.norm(x_est[:3] - missile.position)

        # pf.validate_futures(obs_position, threshold=300.0)

        # --- Belief-weighted future intercept (Stage A) ---
        H = 8  # must match generate_measurement_futures(H=10, ...)
        # future_points = []
        # future_weights = []

        # for i in range(pf.num_particles):
        #     if len(pf.futures[i]) >= H:
        #         fh = pf.futures[i][H-1]   # this is a FutureHypothesis object
        #         # print(dir(fh))
        #         # exit()

        #         # Extract the predicted position from the hypothesis
        #         # Most likely stored as a sequence of z's
        #         # z_future = fh.z_seq[-1]   # shape (3,)
        #         z_future = fh.z_hat[-1]    # shape (3,)
        #         future_points.append(z_future)
        #         future_weights.append(pf.weights[i])

        # if len(future_points) > 0:
        #     future_points = np.array(future_points)
        #     future_weights = np.array(future_weights)
        #     future_weights /= np.sum(future_weights)

        #     # x_aim = np.sum(future_points * future_weights[:, None], axis=0)
        # else:
        #     # All futures killed → fallback to last valid PF mean
        #     if np.sum(pf.weights) > 0:
        #         x_aim = pf.estimate_state()[:3]
        #     else:
        #         # Total degeneracy: reinitialize weights uniformly
        #         pf.weights[:] = 1.0 / pf.num_particles
        #         # x_aim = np.mean(pf.particles[:, :3], axis=0)


        # pf.update(obs, missile.position)


        # if pf.effective_sample_size() < 0.5 * pf.num_particles:
        #     pf.resample()

        # if not collapsed:
        #     pf.update(obs, missile.position)

        #     if pf.effective_sample_size() < 0.5 * pf.num_particles:
        #         pf.resample()


        # est_state = pf.estimate_state()
        est_state = x_est

        est_pos = est_state[:3]
        est_vel = est_state[3:6]

        # pf_err = np.linalg.norm(est_pos - target.position)
        if not collapsed:
            pf_err = np.linalg.norm(est_pos - target.position)
        else:
            pf_err = np.nan

        pf_pos_errors.append(pf_err)

        # --- Smooth the estimate before feeding it to guidance ---
        # if guid_state is None:
        #     # Initialize with first estimate
        #     guid_state = np.concatenate([est_pos, est_vel])
        # else:
        #     guid_state = (
        #         (1.0 - smooth_alpha) * guid_state +
        #         smooth_alpha * np.concatenate([est_pos, est_vel])
        #     )

        # guid_pos = guid_state[:3]
        # guid_vel = guid_state[3:6]

        # Missile guided by SMOOTHED PF estimate
        # missile.update(dt, guid_pos, guid_vel)
        # --- Belief-weighted guidance ---
        # 1. Get belief over future target motion
        # future_trajs, weights = extract_belief_tube(pf, H)

        # # --- HANDLE BELIEF COLLAPSE ---
        # if future_trajs is None or len(future_trajs) == 0:
        #     # fall back to PF mean state guidance
        #     x_est = pf.estimate_state()
        #     aim_point = x_est[:3]
        # else:
        #     T = future_trajs.shape[1]
        # # 2. Ego state
        # x0 = np.hstack([missile.position, missile.velocity])
        # # 3. Solve belief-space MPC
        # U_star = cem_optimize(
        #     x0,
        #     future_trajs,
        #     weights,
        #     dt,
        #     T=future_trajs.shape[1],
        #     H=future_trajs.shape[1]
        # )

        # # 4. Apply only first control (receding horizon)
        # x_next = true_dynamics(
        #     np.hstack([missile.position, missile.velocity]),
        #     U_star[0],
        #     dt
        # )
        # missile.position = x_next[:3]
        # missile.velocity = x_next[3:6]
        # future_trajs, weights = extract_belief_tube(pf, H)

        # # Ego state
        # x0 = np.hstack([missile.position, missile.velocity])

        # # --- HANDLE BELIEF COLLAPSE ---
        # if future_trajs is None or len(future_trajs) == 0:
        #     # terminal proportional guidance
        #     # aim_point = pf.estimate_state()[:3]
        #     aim_point = x_est[:3]
        #     direction = aim_point - missile.position
        #     direction_unit = direction / (np.linalg.norm(direction) + 1e-6)
        #     closing_speed = np.dot(direction_unit, missile.velocity)

        #     # u0 = 2.0 * direction_unit - 0.3 * closing_speed * direction_unit
        #     r = aim_point - missile.position
        #     r_hat = r / (np.linalg.norm(r) + 1e-6)

        #     v_rel = x_est[3:6] - missile.velocity
        #     closing_speed = np.dot(v_rel, r_hat)

        #     u0 = (
        #         2.0 * r_hat
        #         + 0.5 * (v_rel - closing_speed * r_hat)
        #     )
        # else:
        #     U_star = cem_optimize(
        #         x0,
        #         future_trajs,
        #         weights,
        #         dt,
        #         T=future_trajs.shape[1],
        #         H=future_trajs.shape[1]
        #     )
        #     u0 = U_star[0]

        x0 = np.hstack([missile.position, missile.velocity])

        # Authoritative estimate (must already be set earlier in the loop)
# x_est = pf.estimate_state()  OR frozen_state if collapsed

# x0 = np.hstack([missile.position, missile.velocity])

        # if collapsed:
        #     # ===============================
        #     # TERMINAL MODE (NO BELIEF, NO MPC)
        #     # ===============================
        #     aim_point = x_est[:3]

        #     r = aim_point - missile.position
        #     r_hat = r / (np.linalg.norm(r) + 1e-6)

        #     v_rel = x_est[3:6] - missile.velocity
        #     closing_speed = np.dot(v_rel, r_hat)

        #     u0 = (
        #         3.0 * r_hat
        #         + 1.0 * (v_rel - closing_speed * r_hat)
        #     )

        #     amax = 60.0  # m/s^2
        #     norm_u = np.linalg.norm(u0)
        #     if norm_u > amax:
        #         u0 = u0 / norm_u * amax

        # else:
        #     # ===============================
        #     # BELIEF MODE (MPC OVER FUTURES)
        #     # ===============================
        #     # future_trajs, weights = extract_belief_tube(pf, H)
        #     # if step % 20 == 0 and not collapsed:
        #     #     fig2 = plt.figure(2)
        #     #     ax2 = fig2.add_subplot(111, projection='3d')
        #     #     ax2.clear()

        #     #     for k in range(min(30, future_trajs.shape[0])):
        #     #         ax2.plot(
        #     #             future_trajs[k,:,0],
        #     #             future_trajs[k,:,1],
        #     #             future_trajs[k,:,2],
        #     #             alpha=0.3
        #     #         )

        #     #     ax2.scatter(*target.position, c='r', label='Target')
        #     #     ax2.scatter(*missile.position, c='b', label='Missile')
        #     #     ax2.set_title(f"Belief Futures @ step {step}")
        #     #     plt.pause(0.001)
        #     # # ---- CRITICAL GUARD ----
        #     # if future_trajs.ndim < 2 or future_trajs.shape[0] == 0:
        #     #     # Fall back safely (do NOT run MPC on empty belief)
        #     #     aim_point = x_est[:3]

        #     #     r = aim_point - missile.position
        #     #     r_hat = r / (np.linalg.norm(r) + 1e-6)

        #     #     v_rel = x_est[3:6] - missile.velocity
        #     #     closing_speed = np.dot(v_rel, r_hat)

        #     #     u0 = (
        #     #         3.0 * r_hat
        #     #         + 1.0 * (v_rel - closing_speed * r_hat)
        #     #     )
        #     # else:
        #     #     T_ctrl = 4
        #     #     H_pred = H

        #     #     U_star = cem_optimize(
        #     #         x0,
        #     #         future_trajs,
        #     #         weights,
        #     #         dt,
        #     #         T=T_ctrl,
        #     #         H=H_pred
        #     #     )

        #     #     u0 = U_star[0]

        #     # ===============================
        #     # GREEDY BELIEF-BASED GUIDANCE
        #     # ===============================

        #     # Predict short-horizon target position (greedy lookahead)
        #     # tau = 2.0   # seconds (critical, do not make large)
        #     # x_t = x_est[:3]
        #     # v_t = x_est[3:6]

        #     # aim_point = x_t + tau * v_t

        #     # # Relative geometry
        #     # r = aim_point - missile.position
        #     # r_norm = np.linalg.norm(r) + 1e-6
        #     # r_hat = r / r_norm

        #     # # Relative velocity
        #     # v_rel = v_t - missile.velocity
        #     # closing_speed = np.dot(v_rel, r_hat)

        #     # # Greedy acceleration command
        #     # k_p = 4.0
        #     # k_d = 1.5

        #     # u0 = (
        #     #     k_p * r_hat +
        #     #     k_d * (v_rel - closing_speed * r_hat)
        #     # )

        #     # # Acceleration saturation
        #     # amax = 60.0  # m/s^2
        #     # norm_u = np.linalg.norm(u0)
        #     # if norm_u > amax:
        #     #     u0 = u0 / norm_u * amax
        #     # === GREEDY GUIDANCE WITH Z-DAMPING ===

        #     aim_pos = x_est[:3]
        #     aim_vel = x_est[3:6]

        #     r = aim_pos - missile.position
        #     r_norm = np.linalg.norm(r) + 1e-6
        #     r_hat = r / r_norm
        #     # --- Terminal aggressiveness boost ---
        #     if r_norm < 4000.0:
        #         u_cmd *= 1.8

        #     v_rel = aim_vel - missile.velocity
        #     closing_speed = np.dot(v_rel, r_hat)

        #     # Proportional + damping
        #     u_cmd = (
        #         3.0 * r_hat
        #         + 1.5 * (v_rel - closing_speed * r_hat)
        #     )

        #     # ---- Z-axis damping (CRITICAL FIX) ----
        #     Z_DAMP = 0.35      # <-- THIS fixes the vertical blow-up
        #     u_cmd[2] *= Z_DAMP

        #     # ---- Acceleration saturation ----
        #     A_MAX = 60.0  # m/s^2
        #     u_norm = np.linalg.norm(u_cmd)
        #     if u_norm > A_MAX:
        #         u_cmd = u_cmd / u_norm * A_MAX

        #     u0 = u_cmd


        # # =========================
        # # GUIDANCE LAW (REPLACEMENT)
        # # =========================

        # aim_pos = x_est[:3]
        # aim_vel = x_est[3:6]

        # r = aim_pos - missile.position
        # r_norm = np.linalg.norm(r) + 1e-6
        # r_hat = r / r_norm

        # v_rel = aim_vel - missile.velocity
        # closing_speed = np.dot(v_rel, r_hat)

        # # Robust pursuit + damping (maneuver-agnostic)
        # u_cmd = (
        #     8.0 * r_hat
        #     + 5.0 * (v_rel - closing_speed * r_hat)
        # )

        # # --- Vertical damping (CRITICAL for your sensor setup) ---
        # u_cmd[2] *= 0.0

        # # Acceleration saturation
        # A_MAX = 120.0  # m/s^2
        # u_norm = np.linalg.norm(u_cmd)
        # if u_norm > A_MAX:
        #     u_cmd = u_cmd / u_norm * A_MAX

        # u0 = u_cmd


        # =========================
        # STABLE LEAD-PURSUIT LAW
        # =========================

        # Estimated target state (QCTPF)
        p_t = x_est[:3]
        v_t = x_est[3:6]

        # Missile state
        p_m = missile.position
        v_m = missile.velocity

        # Relative geometry
        r = p_t - p_m
        r_norm = np.linalg.norm(r) + 1e-6
        r_hat = r / r_norm

        # Relative speed magnitude
        v_m_mag = np.linalg.norm(v_m) + 1e-6

        # Lead time (bounded)
        t_go = np.clip(r_norm / v_m_mag, 1.0, 6.0)

        # Intercept point (simple lead)
        p_aim = p_t + t_go * v_t

        # Desired acceleration direction
        a_dir = p_aim - p_m
        a_dir = a_dir / (np.linalg.norm(a_dir) + 1e-6)

        # Acceleration command
        u0 = 50.0 * a_dir

        # NO vertical control (important)
        u0[2] = 0.0

        # Saturation
        A_MAX = 60.0
        norm_u = np.linalg.norm(u0)
        if norm_u > A_MAX:
            u0 = u0 / norm_u * A_MAX




        # Apply control
        x_next = true_dynamics(x0, u0, dt)
        missile.position = x_next[:3]
        missile.velocity = x_next[3:6]


        target_hist.append(target.position.copy())
        missile_hist.append(missile.position.copy())
        est_hist.append(est_pos.copy())

        if step % 5 == 0 or step == num_steps - 1:
            th = np.array(target_hist)
            mh = np.array(missile_hist)
            eh = np.array(est_hist)

            target_line.set_data(th[:, 0], th[:, 1])
            target_line.set_3d_properties(th[:, 2])
            missile_line.set_data(mh[:, 0], mh[:, 1])
            missile_line.set_3d_properties(mh[:, 2])
            est_line.set_data(eh[:, 0], eh[:, 1])
            est_line.set_3d_properties(eh[:, 2])
            # --- Target start/end markers ---
            t_start._offsets3d = (
                [th[0, 0]], [th[0, 1]], [th[0, 2]]
            )
            t_end._offsets3d = (
                [th[-1, 0]], [th[-1, 1]], [th[-1, 2]]
            )

            # --- Missile start/end markers ---
            m_start._offsets3d = (
                [mh[0, 0]], [mh[0, 1]], [mh[0, 2]]
            )
            m_end._offsets3d = (
                [mh[-1, 0]], [mh[-1, 1]], [mh[-1, 2]]
            )

            miss_now = np.linalg.norm(target.position - missile.position)
            ax.set_title(
                f"QCTPF Guidance | step {step}/{num_steps}  "
                f"Miss: {miss_now:7.1f} m  PF err: {pf_err:7.1f} m"
            )

            plt.draw()
            plt.pause(0.001)

    target_hist = np.array(target_hist)
    missile_hist = np.array(missile_hist)
    est_hist = np.array(est_hist)

    final_miss = np.linalg.norm(target_hist[-1] - missile_hist[-1])
    mean_pf_err = float(np.mean(pf_pos_errors))
    final_pf_err = float(pf_pos_errors[-1])

    print(f"\n[QCTPF] Final miss distance: {final_miss:.2f} meters")
    print(f"[QCTPF] Mean PF position error: {mean_pf_err:.2f} meters")
    print(f"[QCTPF] Final PF position error: {final_pf_err:.2f} meters")

    plt.ioff()
    plt.tight_layout()
    # plt.show(block=True)
    plt.show()


if __name__ == "__main__":
    run_qctpf_guidance_viewer()




