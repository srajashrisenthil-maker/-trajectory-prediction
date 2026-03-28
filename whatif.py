import numpy as np

def insert_virtual_obstacle(past_traj, obstacle_pos, influence_radius=3.0, repulsion=2.0):
    modified = past_traj.copy()
    for t in range(len(modified)):
        diff = modified[t] - obstacle_pos
        dist = np.linalg.norm(diff)
        if dist < influence_radius and dist > 1e-6:
            repel = (diff / dist) * repulsion * (1 - dist / influence_radius)
            modified[t:] += repel
            break
    return modified

def run_whatif(model_fn, past_traj, positions, obstacle_pos, **kwargs):
    orig_trajs = model_fn(past_traj, positions)
    mod_past   = insert_virtual_obstacle(past_traj, obstacle_pos, **kwargs)
    mod_trajs  = model_fn(mod_past, positions)
    return orig_trajs, mod_trajs, mod_past