import numpy as np

INTENT_NAMES = ['straight', 'turning', 'slowing', 'random']

def detect_intent(past_traj):
    vel = np.diff(past_traj, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    angles = np.arctan2(vel[:, 1], vel[:, 0])
    angle_change = np.abs(np.diff(np.unwrap(angles))).mean()
    speed_change  = np.abs(np.diff(speed)).mean()
    speed_trend   = speed[-1] - speed[0]

    if speed_trend < -0.05 and speed_change > 0.02:
        intent = 'slowing'
    elif angle_change > 0.08:
        intent = 'turning'
    elif angle_change < 0.03 and speed_change < 0.03:
        intent = 'straight'
    else:
        intent = 'random'

    early_angle = np.arctan2(vel[:5, 1],  vel[:5, 0]).mean()
    late_angle  = np.arctan2(vel[-5:, 1], vel[-5:, 0]).mean()
    shift = float(abs(late_angle - early_angle) > 0.5 or
                  abs(speed[-5:].mean() - speed[:5].mean()) > 0.3)
    return intent, bool(shift)

def compute_risk(pred_trajs, ego_vehicle_pos, threshold_hi=3.0, threshold_med=6.0):
    all_dists = []
    for traj in pred_trajs:
        dists = np.linalg.norm(traj - ego_vehicle_pos, axis=1)
        all_dists.append(dists.min())
    min_dist = min(all_dists)
    if min_dist < threshold_hi:
        return 'High', min_dist
    elif min_dist < threshold_med:
        return 'Medium', min_dist
    else:
        return 'Low', min_dist