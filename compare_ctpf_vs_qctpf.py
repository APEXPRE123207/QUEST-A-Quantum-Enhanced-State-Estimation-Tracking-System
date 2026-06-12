# import numpy as np
# import matplotlib.pyplot as plt

# from trial_run_ctpf import run_ctpf_guidance
# from trial_run import run_qctpf_guidance_viewer


# def run_comparison():

#     print("Running Classical CTPF...")
#     target_ctpf, missile_ctpf = run_ctpf_guidance(return_traj=True)

#     print("Running QCTPF...")
#     target_qctpf, missile_qctpf = run_qctpf_guidance_viewer(return_traj=True)

#     # Convert to numpy arrays
#     target_ctpf = np.array(target_ctpf)
#     missile_ctpf = np.array(missile_ctpf)
#     missile_qctpf = np.array(missile_qctpf)

#     # -----------------------------------
#     # 3D Plot
#     # -----------------------------------
#         # -----------------------------------
#     # 3D Plot
#     # -----------------------------------
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # --- TARGET ---
#     ax.plot(target_ctpf[:,0],
#             target_ctpf[:,1],
#             target_ctpf[:,2],
#             'r', linewidth=2, label="Target (Truth)")

#     # Start & End markers (Target)
#     ax.scatter(target_ctpf[0,0], target_ctpf[0,1], target_ctpf[0,2],
#                color='darkred', marker='o', s=80, label="Target Start")

#     ax.scatter(target_ctpf[-1,0], target_ctpf[-1,1], target_ctpf[-1,2],
#                color='red', marker='X', s=100, label="Target End")

#     # --- Classical Missile ---
#     ax.plot(missile_ctpf[:,0],
#             missile_ctpf[:,1],
#             missile_ctpf[:,2],
#             'b', linewidth=2, label="Classical CTPF")

#     ax.scatter(missile_ctpf[0,0], missile_ctpf[0,1], missile_ctpf[0,2],
#                color='navy', marker='o', s=80, label="CTPF Start")

#     ax.scatter(missile_ctpf[-1,0], missile_ctpf[-1,1], missile_ctpf[-1,2],
#                color='blue', marker='X', s=100, label="CTPF End")

#     # --- QCTPF Missile ---
#     ax.plot(missile_qctpf[:,0],
#             missile_qctpf[:,1],
#             missile_qctpf[:,2],
#             'g--', linewidth=2, label="QCTPF")

#     ax.scatter(missile_qctpf[0,0], missile_qctpf[0,1], missile_qctpf[0,2],
#                color='darkgreen', marker='o', s=80, label="QCTPF Start")

#     ax.scatter(missile_qctpf[-1,0], missile_qctpf[-1,1], missile_qctpf[-1,2],
#                color='green', marker='X', s=100, label="QCTPF End")

#     ax.set_xlabel("X (m)")
#     ax.set_ylabel("Y (m)")
#     ax.set_zlabel("Z (m)")
#     ax.set_title("Trajectory Comparison: Classical CTPF vs QCTPF")

#     ax.legend(loc='best')
#     plt.tight_layout()
#     plt.show()



# if __name__ == "__main__":
#     run_comparison()


import numpy as np
import matplotlib.pyplot as plt

from trial_run_ctpf import run_ctpf_guidance
from trial_run import run_qctpf_guidance_viewer


def run_comparison():

    print("Running Classical CTPF...")
    target_ctpf, missile_ctpf = run_ctpf_guidance(return_traj=True)

    print("Running QCTPF...")
    target_qctpf, missile_qctpf = run_qctpf_guidance_viewer(return_traj=True)

    # Convert to numpy arrays
    target_ctpf = np.array(target_ctpf)
    missile_ctpf = np.array(missile_ctpf)
    missile_qctpf = np.array(missile_qctpf)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # --- TARGET ---
    ax.plot(target_ctpf[:,0],
            target_ctpf[:,1],
            target_ctpf[:,2],
            'r', linewidth=2, label="Target (Truth)", zorder=1)

    ax.scatter(target_ctpf[0,0], target_ctpf[0,1], target_ctpf[0,2],
               color='darkred', marker='o', s=60, label="Target Start", zorder=5)

    ax.scatter(target_ctpf[-1,0], target_ctpf[-1,1], target_ctpf[-1,2],
               color='red', marker='X', s=80, label="Target End", zorder=5)

    # --- Classical Missile ---
    ax.plot(missile_ctpf[:,0],
            missile_ctpf[:,1],
            missile_ctpf[:,2],
            'b', linewidth=2, label="CTPF", zorder=2)

    # Draw QCTPF later so it doesn't hide CTPF start
    ax.scatter(missile_ctpf[0,0], missile_ctpf[0,1], missile_ctpf[0,2],
               color='navy', marker='o', s=70, label="CTPF Start", zorder=10)

    ax.scatter(missile_ctpf[-1,0], missile_ctpf[-1,1], missile_ctpf[-1,2],
               color='blue', marker='X', s=80, label="CTPF End", zorder=10)

    # --- QCTPF Missile ---
    ax.plot(missile_qctpf[:,0],
            missile_qctpf[:,1],
            missile_qctpf[:,2],
            'g--', linewidth=2, label="QCTPF", zorder=3)

    ax.scatter(missile_qctpf[0,0], missile_qctpf[0,1], missile_qctpf[0,2],
               color='darkgreen', marker='o', s=60, label="QCTPF Start", zorder=6)

    ax.scatter(missile_qctpf[-1,0], missile_qctpf[-1,1], missile_qctpf[-1,2],
               color='green', marker='X', s=80, label="QCTPF End", zorder=6)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Trajectory Comparison: Classical CTPF vs QCTPF")

    # Smaller legend, top-right
    ax.legend(loc='upper right', fontsize=8, frameon=True)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    run_comparison()
