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

def rollout_target_segment(x0, a_seq, dt):
    """
    x0   : [x,y,z,vx,vy,vz]
    a_seq: list/array of accelerations, shape (H,3)
    dt   : timestep

    Returns:
        segment: list of dicts with keys {'pos','vel'}
    """
    pos = x0[:3].copy()
    vel = x0[3:6].copy()

    segment = []
    for a in a_seq:
        vel = vel + a * dt
        pos = pos + vel * dt
        segment.append({
            "pos": pos.copy(),
            "vel": vel.copy()
        })

    return segment

def quantum_predict_intent(pf, x_est, H):
    """
    Uses the QNEAT-QCTPF machinery to predict
    short-horizon acceleration intent.

    Output:
        a_seq: np.ndarray of shape (H,3)
    """

    # We reuse the quantum delta predictor already in QCTPF
    # dz ≈ position deltas → convert to acceleration intent conservatively

    dz = pf._predict_measurement_delta(
        particle_state=np.hstack([x_est, 0.0]),  # omega not used here
        idx=0,
        ownship_position=pf._ownship_position
    )

    # dz is length >= 3H; map to acceleration intent safely
    a_seq = []
    for k in range(H):
        dk = dz[3*k:3*(k+1)]
        if dk.shape[0] < 3:
            dk = np.pad(dk, (0, 3 - dk.shape[0]))
        elif dk.shape[0] > 3:
            dk = dk[:3]

        a_seq.append(0.3 * dk)


    return np.array(a_seq)

def extract_belief_tube(pf, H):
    trajs = []
    weights = []
    for i in range(pf.num_particles):
        if len(pf.futures[i]) > 0 and pf.weights[i] > 0:
            fh = pf.futures[i][-1]
            trajs.append(fh.z_hat[:H])   # (H,3)
            weights.append(pf.weights[i])
    weights = np.array(weights)
    weights /= np.sum(weights)
    return np.array(trajs), weights

def rollout_ego(x0, U, dt):
    """
    x0 = [pos(3), vel(3)]
    U  = [u0, u1, ..., u_{T-1}]  (each u is acceleration 3-vector)
    """
    p, v = x0[:3].copy(), x0[3:].copy()
    traj = []
    for u in U:
        v = v + u * dt
        p = p + v * dt
        traj.append(p.copy())
    return np.array(traj)  # shape (T,3)

def belief_cost(U, x0, future_trajs, weights, alpha, dt, lam):
    ego_traj = rollout_ego(x0, U, dt)  # (T,3)
    H = future_trajs.shape[1]
    J = 0.0

    for i in range(len(weights)):
        w = weights[i]
        for t in range(H):
            J += w * alpha[t] * np.linalg.norm(
                    ego_traj[t] - future_trajs[i, t]
                 )**2

    # Control regularization
    for u in U:
        J += lam * np.linalg.norm(u)**2

    return J

def time_weights(H):
    return (np.arange(1, H+1) / H)**2

def cem_optimize(x0, future_trajs, weights, dt,
                 T=5, H=5, iters=5, samples=200, elite_frac=0.1):

    dim = 3 * T
    mu = np.zeros(dim)
    sigma = np.eye(dim) * 5.0

    alpha = (np.arange(1, H+1) / H)**2
    lam = 0.01

    for _ in range(iters):
        Us = np.random.multivariate_normal(mu, sigma, samples)
        costs = []

        for j in range(samples):
            U = Us[j].reshape(T,3)
            J = belief_cost(U, x0, future_trajs, weights, alpha, dt, lam)
            costs.append(J)

        costs = np.array(costs)
        elite_idx = np.argsort(costs)[:int(elite_frac*samples)]
        elite = Us[elite_idx]

        mu = elite.mean(axis=0)
        sigma = np.cov(elite.T) + 1e-6*np.eye(dim)

    return mu.reshape(T,3)

def run_qctpf_guidance_viewer():


    sim_opts = SimOptions()
    dt = sim_opts.dt
    num_steps = 300

    target = Target(
        initial_position=[5000.0, 500.0, 10000.0],
        initial_velocity=[-250.0, 40.0, 0.0]
    )

    missile = Missile(
        initial_position=[0.0, 0.0, 9000.0],
        initial_velocity=[300.0, 0.0, 0.0]
    )

    sensor = Sensor(
        radar_noise_std={
            'range': 50.0,
            'velocity': 5.0,
            'azimuth': 0.005,
            'elevation': 0.005
        },
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
    ax.set_xlim(-12000, 13000)
    ax.set_ylim(-10000, 10000)
    ax.set_zlim(-15000, 15000)

    predicted_segment = None
    segment_index = 0
    segment_active = False
    segment_steps_left = 0
    segment_target_pos = None   # locked reference point
    SEGMENT_LENGTH = 10         # K (you said K=10)
    SEGMENT_HORIZON = 6 
    for step in range(num_steps):
        # if step < 40:
        #     maneuver = ('straight', {})
        # elif step < 80:
        #     maneuver = ('turn', {'g_force': 4.0})
        # elif step < 120:
        #     maneuver = ('jink', {'frequency': 0.4, 'amplitude': 30.0})
        # elif step < 160:
        #     maneuver = ('climb_dive', {'vertical_g': 2.0})
        # else:
        #     maneuver = ('turn', {'g_force': -5.0})
        if step < 80:
            maneuver = ('straight', {})
        elif step < 160:
            maneuver = ('turn', {'g_force': 3.0})
        else:
            maneuver = ('climb_dive', {'vertical_g': 1.5})

        target.update(dt, maneuver)

        # target.update(dt, ('turn', {'g_force': 4.0}))
        obs = sensor.observe(target, missile.position)
        z_obs = obs["position"] 
        # pf.predict(dt, missile.position, alpha_q=0.1)
        pf.predict(dt, missile.position)
        pf.update(z_obs, missile.position)
        x_est = pf.estimate_state()
        
        # ======================================================
        # SEGMENT REPLANNING (QCTPF)
        # ======================================================

        if (not segment_active) or (segment_steps_left <= 0):

            # ---- Inject truth occasionally (you asked for this) ----
            if step % (2 * SEGMENT_LENGTH) == 0:
                # hard reset using true target position
                seed_state = np.hstack([target.position, target.velocity])
            else:
                # normal replanning from PF estimate
                seed_state = x_est.copy()

            # ---- QCTPF predicts short future segment ----
            dz_seq = []
            for h in range(SEGMENT_HORIZON):
                dz = pf._predict_measurement_delta(
                    particle_state=np.hstack([seed_state, 0.0]),
                    idx=h,
                    ownship_position=pf._ownship_position
                )
                dz_seq.append(dz[:3])   # position delta intent

            # ---- Build predicted segment endpoint ----
            dz_seq = np.array(dz_seq)
            segment_target_pos = seed_state[:3] + np.sum(dz_seq, axis=0)

            # ---- Lock segment ----
            segment_steps_left = SEGMENT_LENGTH
            segment_active = True

        # ======================================================
        # SEGMENT TRACKING GUIDANCE (TURN-RATE ONLY)
        # ======================================================

        # Missile heading
        v_m = missile.velocity
        V = np.linalg.norm(v_m) + 1e-6
        v_hat = v_m / V

        # Vector to locked segment target
        e_seg = segment_target_pos - missile.position

        # Project into turn plane
        e_seg_perp = e_seg - np.dot(e_seg, v_hat) * v_hat
        e_norm = np.linalg.norm(e_seg_perp)

        if e_norm > 1e-6:
            omega_track = np.cross(v_hat, e_seg_perp) / e_norm
        else:
            omega_track = np.zeros(3)

        # Gain (this is your main tuning knob)
        k_track = 0.35
        omega_cmd = k_track * omega_track

        # ======================================================
        # WEAK TERMINAL ANGULAR PN (OPTIONAL)
        # ======================================================

        r_LOS = x_est[:3] - missile.position
        r_norm = np.linalg.norm(r_LOS) + 1e-6
        los_hat = r_LOS / r_norm
        v_rel = x_est[3:6] - missile.velocity

        los_rate = np.cross(r_LOS, v_rel) / (r_norm * r_norm + 1e-6)

        if r_norm < 3000.0:
            omega_cmd += 0.4 * los_rate

        # ======================================================
        # TURN-RATE LIMIT + DYNAMICS
        # ======================================================

        omega_cmd = omega_cmd - np.dot(omega_cmd, v_hat) * v_hat

        omega_max = 0.3
        n = np.linalg.norm(omega_cmd)
        if n > omega_max:
            omega_cmd = omega_cmd / n * omega_max

        v_hat_new = v_hat + np.cross(omega_cmd, v_hat) * dt
        v_hat_new /= np.linalg.norm(v_hat_new) + 1e-6

        missile.velocity = V * v_hat_new
        missile.position += missile.velocity * dt

        segment_steps_left -= 1



        # pf.generate_measurement_futures(H=3, M=5)
        # if step % 20 == 0:
        #     print("Mean |Δz_pred|:", np.mean(np.abs(pf.last_dz_pred)))
        # obs_position = target.position.copy()
        # pf.validate_futures(obs_position, threshold=300.0)
        # r = obs["range"]
        # az = obs["azimuth"]
        # el = obs["elevation"]

        # z_obs = np.array([
        #     r * np.cos(el) * np.cos(az),
        #     r * np.cos(el) * np.sin(az),
        #     r * np.sin(el)
        # ])

        # pf.validate_futures(z_obs, threshold=300.0)
        # spread = np.sqrt(np.trace(np.cov(pf.particles[:, :3].T)))
        # pf.validate_futures(z_obs, threshold=300.0 + spread)

        z_meas = np.array([obs["range"], obs["azimuth"], obs["elevation"]])
        # pf.validate_futures(z_meas, threshold=0.2)
        z_ref = pf.estimate_state()[:3]
        # pf.validate_futures(z_ref, threshold=300.0)


        # alive = sum(len(f) for f in pf.futures)
        # print(f"step {step}: surviving futures = {alive}")
        # --- Belief-weighted future intercept (Stage A) ---
        H = 3  # must match generate_measurement_futures(H=3, ...)
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




        if pf.effective_sample_size() < 0.5 * pf.num_particles:
            pf.resample()
        # --- STATE ESTIMATION (always defined) ---


        # if np.sum(pf.weights) > 1e-6:
        #     x_est = pf.estimate_state()
        # else:
        #     x_est = pf.particles.mean(axis=0)
        # --- Quantum intent prediction (every K steps) ---
        # if step % 10 == 0:
        #     # a_seq = quantum_predict_intent(pf, x_est, H=6)
        #     # predicted_segment = rollout_target_segment(x_est, a_seq, dt)
        #     segment_index = 0
        # # --- Choose aim point ---
        # if predicted_segment is not None and segment_index < len(predicted_segment):
        #     # aim_pos = predicted_segment[segment_index]["pos"]
        #     segment_index += 1
        # else:
        #     # Safe fallback
        #     aim_pos = x_est[:3]

        # ======================================================
        # TURN-RATE–ONLY GUIDANCE (OPTION B)
        # ======================================================

        # # --- Missile kinematics ---
        # v_m = missile.velocity
        # V = np.linalg.norm(v_m) + 1e-6
        # v_hat = v_m / V

        # # --- LOS geometry ---
        # r_LOS = x_est[:3] - missile.position
        # r_norm = np.linalg.norm(r_LOS) + 1e-6
        # los_hat = r_LOS / r_norm

        # v_rel = x_est[3:6] - missile.velocity

        # # --- Time-to-go estimate ---
        # closing_speed = -np.dot(v_rel, los_hat)
        # closing_speed = max(closing_speed, 10.0)
        # t_go = r_norm / closing_speed

        # # ======================================================
        # # (1) QUANTUM INTENT TURN-RATE
        # # ======================================================

        # # Predicted aim point from QCTPF (already computed)
        # e_Q = aim_pos - missile.position

        # # Project into turn plane
        # e_Q_perp = e_Q - np.dot(e_Q, v_hat) * v_hat
        # e_Q_norm = np.linalg.norm(e_Q_perp) + 1e-6

        # k_Q = 0.6   # quantum intent authority
        
        # omega_Q = k_Q * np.cross(v_hat, e_Q_perp) / e_Q_norm
        # if e_Q_norm < 50.0:
        #     omega_Q[:] = 0.0

        # # ======================================================
        # # (2) GEOMETRIC STABILIZATION TURN-RATE
        # # ======================================================

        # k_G = 0.25
        # # omega_geom = k_G * np.cross(v_hat, los_hat)


        # # ======================================================
        # # (3) TERMINAL ANGULAR PN (GATED)
        # # ======================================================

        # los_rate = np.cross(r_LOS, v_rel) / (r_norm * r_norm + 1e-6)

        # k_PN = 3.0
        # omega_PN = k_PN * los_rate

        # # Terminal gate
        # # t_on = 8.0
        # # if t_go < t_on:
        # #     gate = 1.0 - t_go / t_on
        # # else:
        # #     gate = 0.0

        # # omega_PN = gate * omega_PN


        # # ======================================================
        # # (4) COMBINE TURN-RATE COMMAND
        # # ======================================================

        # omega_cmd = omega_Q + omega_geom + omega_PN

        # # Ensure pure turn-rate (no component along velocity)
        # omega_cmd = omega_cmd - np.dot(omega_cmd, v_hat) * v_hat

        # # Turn-rate limit
        # omega_max = 0.45  # rad/s
        # omega_norm = np.linalg.norm(omega_cmd)
        # if omega_norm > omega_max:
        #     omega_cmd = omega_cmd / omega_norm * omega_max


        # # ======================================================
        # # (5) MISSILE HEADING UPDATE (TURN-RATE DYNAMICS)
        # # ======================================================

        # v_hat_new = v_hat + np.cross(omega_cmd, v_hat) * dt
        # v_hat_new /= np.linalg.norm(v_hat_new) + 1e-6

        # missile.velocity = V * v_hat_new
        # missile.position += missile.velocity * dt


                

        target_hist.append(target.position.copy())
        missile_hist.append(missile.position.copy())
        # est_hist.append(est_pos.copy())
        est_hist.append(x_est[:3].copy())
        pf_err = np.linalg.norm(x_est[:3] - target.position)
        pf_pos_errors.append(pf_err)


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
    plt.show()


if __name__ == "__main__":
    run_qctpf_guidance_viewer()










# # est_state = pf.estimate_state()
#         # est_pos = est_state[:3]
#         # est_vel = est_state[3:6]
#         est_pos = x_est[:3]
#         est_vel = x_est[3:6]
#         pos_var = np.var(pf.particles[:, :3], axis=0) + 1e-3
#         pf_err = np.linalg.norm(est_pos - target.position)
#         pf_pos_errors.append(pf_err)

#         # est = pf.estimate_state()
#         # est_pos = est[:3]
#         # est_vel = est[3:6]

#         # r = est_pos - missile.position
#         # v_rel = est_vel - missile.velocity

#         # r_norm = np.linalg.norm(r) + 1e-6
#         # r_hat = r / r_norm

#         # closing = -np.dot(v_rel, r_hat)
#         # t_go = r_norm / max(closing, 50.0)

#         # aim = est_pos + est_vel * t_go

#         # a_cmd = 3.0 * (aim - missile.position) / (t_go * t_go * np.sqrt(pos_var))
#         # r = aim_pos - missile.position
#         # Vector to predicted path (for pursuit shaping)
#         r_path = aim_pos - missile.position

#         # Vector to current estimated target (for PN)
#         r_pn = x_est[:3] - missile.position

#         # r_norm = np.linalg.norm(r) + 1e-6
#         # r_hat = r / r_norm

#         # v_rel = x_est[3:6] - missile.velocity
#         r_norm = np.linalg.norm(r_pn) + 1e-6
#         los_hat = r_pn / r_norm

#         v_rel = x_est[3:6] - missile.velocity
#         los_rate = np.cross(r_pn, v_rel) / (r_norm * r_norm + 1e-6)
#         N = 4.5 
        

#         # closing = -np.dot(v_rel, r_hat)
#         # closing = -np.dot(v_rel, los_hat)

#         # t_go = r_norm / max(closing, 50.0)
#         closing = max(-np.dot(v_rel, los_hat), 10.0)
#         t_go = r_norm / closing
#         # PN gating factor (0 → pursuit, 1 → full PN)
#         t_gate = 12.0  # seconds, midcourse → terminal transition
#         # pn_gain = np.clip(1.0 - t_go / t_gate, 0.0, 1.0)
#         pn_min = 0.5   # always keep some PN for stability
#         pn_gain = np.clip(pn_min + (1.0 - pn_min) * (1.0 - t_go / t_gate), pn_min, 1.0)

#         r_path_hat = r_path / (np.linalg.norm(r_path) + 1e-6)
#         # a_pursuit = 1.5 * r_path_hat
#         v_m = missile.velocity
#         v_norm = np.linalg.norm(v_m) + 1e-6
#         # a_pursuit = 2.0 * v_norm / max(t_go, 1.0) * r_path_hat
#         v_hat = v_m / v_norm
#         a_pursuit = 2.0 * v_norm / max(t_go, 1.0) * (
#             r_path_hat - np.dot(r_path_hat, v_hat) * v_hat
#         )



#         pos_var = np.var(pf.particles[:, :3], axis=0) + 1e-3

#         # a_cmd = 3.0 * r / (t_go * t_go * np.sqrt(pos_var))
#         # Missile velocity unit vector
#         # a_pn = N * np.linalg.norm(missile.velocity) * np.cross(los_rate, v_hat)
#         a_pn = N * v_norm * np.cross(v_hat, los_rate)
#         a_pn[2] *= 0.35
#         # Line-of-sight unit vector
#         # los_hat = r_hat

#         # Lateral component of LOS error (this causes turning)
    
#         # Lateral acceleration magnitude
#         # a_lat = 4.0 * v_norm / max(t_go, 1.0)

#         # a_cmd = a_lat * a_lat_dir
#         # Relative velocity
#         v_rel = x_est[3:6] - missile.velocity

#         # Line-of-sight rate vector
#         # los_rate = np.cross(r, v_rel) / (r_norm * r_norm + 1e-6)

#         # PN-style lateral acceleration
#         # N = 4.5  # navigation constant
#         # a_pn = N * v_norm * np.cross(los_rate, v_hat)
#         # # Vertical PN damping (critical for 3D stability)
#         # a_pn[2] *= 0.35

#         # Pure lateral pursuit term (stabilizer)
#         # a_pursuit = 2.0 * v_norm / max(t_go, 1.0) * a_lat_dir

#         # Combined command
#         a_cmd = pn_gain * a_pn + a_pursuit
#         amax = 40.0
#         if np.linalg.norm(a_cmd) > amax:
#             a_cmd = a_cmd / np.linalg.norm(a_cmd) * amax
#         # --- Longitudinal closure control ---
#         closing = np.dot(x_est[:3] - missile.position, v_hat)

#         # Simple proportional closure law
#         a_long = 0.5 * closing / max(t_go, 1.0)

#         # Limit longitudinal accel
#         a_long = np.clip(a_long, -10.0, 10.0)

#         # Reconstruct full command
#         a_par = np.dot(a_cmd, v_hat) * v_hat          # parallel component
#         a_lat = a_cmd - a_par                          # lateral component
#         a_cmd = a_lat + a_long * v_hat

#         # missile.velocity += a_cmd * dt
#         # missile.position += missile.velocity * dt
#         # --- Physically correct missile dynamics (heading-rate based) ---

#         v = missile.velocity
#         v_norm = np.linalg.norm(v) + 1e-6
#         v_hat = v / v_norm

#         # Decompose acceleration

#         # Heading rate (omega vector)
#         omega = np.cross(v_hat, a_lat) / v_norm        # rad/s vector

#         # Limit turn rate (optional but recommended)
#         omega_max = 0.35   # rad/s  (~20 deg/s)
#         omega_norm = np.linalg.norm(omega)
#         if omega_norm > omega_max:
#             omega = omega / omega_norm * omega_max

#         # Rotate velocity direction
#         v_hat_new = v_hat + np.cross(omega, v_hat) * dt
#         v_hat_new /= np.linalg.norm(v_hat_new) + 1e-6

#         # Keep speed constant (or add thrust model later)
#         # missile.velocity = v_norm * v_hat_new
#         v_norm_new = max(v_norm + a_long * dt, 50.0)
#         missile.velocity = v_norm_new * v_hat_new


#         # Update position
#         missile.position += missile.velocity * dt


#         # --- Smooth the estimate before feeding it to guidance ---
#         # if guid_state is None:
#         #     # Initialize with first estimate
#         #     guid_state = np.concatenate([est_pos, est_vel])
#         # else:
#         #     guid_state = (
#         #         (1.0 - smooth_alpha) * guid_state +
#         #         smooth_alpha * np.concatenate([est_pos, est_vel])
#         #     )

#         # guid_pos = guid_state[:3]
#         # guid_vel = guid_state[3:6]

#         # Missile guided by SMOOTHED PF estimate
#         # missile.update(dt, guid_pos, guid_vel)
#         # --- Belief-weighted guidance ---
#         # 1. Get belief over future target motion
#         # future_trajs, weights = extract_belief_tube(pf, H)
#         # # 2. Ego state
#         # x0 = np.hstack([missile.position, missile.velocity])
#         # # 3. Solve belief-space MPC
#         # # After extracting belief tube
#         # if future_trajs is None or len(future_trajs) == 0:
#         #     # Fallback to state-based guidance
#         #     use_belief_guidance = False
#         # else:
#         #     use_belief_guidance = True

#         # # U_star = cem_optimize(
#         # #     x0,
#         # #     future_trajs,
#         # #     weights,
#         # #     dt,
#         # #     T=future_trajs.shape[1],
#         # #     H=future_trajs.shape[1]
#         # # )

#         # # # 4. Apply only first control (receding horizon)
#         # # u0 = U_star[0]
#         # if use_belief_guidance:
#         #     T = future_trajs.shape[1]

#         #     U_star = cem_optimize(
#         #         x0,
#         #         future_trajs,
#         #         weights,
#         #         dt,
#         #         T=T,
#         #         H=T
#         #     )
#         #     u0 = U_star[0]
#         # else:
#         #     # Simple fallback guidance using current estimate
#         #     # --- Robust fallback guidance (lead pursuit) ---
#         #     r = p_tgt - missile.position
#         #     r_norm = np.linalg.norm(r) + 1e-6
#         #     r_hat = r / r_norm

#         #     # Relative velocity (missile only)
#         #     v_m = missile.velocity
#         #     closing_speed = np.dot(v_m, r_hat)

#         #     # Lateral correction (pure geometry)
#         #     v_lat = v_m - closing_speed * r_hat

#         #     # Guidance command (tunable, stable)
#         #     u_cmd = (
#         #         2.0 * r_hat           # pursuit term
#         #         - 1.2 * v_lat         # damping
#         #     )

#         #     # Acceleration limit
#         #     amax = 40.0
#         #     norm_u = np.linalg.norm(u_cmd)
#         #     if norm_u > amax:
#         #         u_cmd = u_cmd / norm_u * amax


#         # missile.velocity += u0 * dt
#         # missile.position += missile.velocity * dt