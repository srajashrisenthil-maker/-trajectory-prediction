import numpy as np

PAST_STEPS = 20
FUTURE_STEPS = 30

def add_features(traj):
    xy = traj
    vel = np.diff(xy, axis=0, prepend=xy[[0]])
    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    direction = np.arctan2(vel[:, 1:2], vel[:, 0:1])
    return np.concatenate([xy, vel, speed, direction], axis=1)

def normalize(data, mean=None, std=None):
    if mean is None:
        mean = data.mean(axis=(0, 1, 2))
        std  = data.std(axis=(0, 1, 2)) + 1e-8
    return (data - mean) / std, mean, std

def make_windows(scenes):
    N, A, T, _ = scenes.shape
    total = PAST_STEPS + FUTURE_STEPS
    past_list, future_list = [], []
    for s in range(N):
        for t in range(T - total + 1):
            p, f = [], []
            for a in range(A):
                seg = scenes[s, a, t:t+total]
                seg_feat = add_features(seg)
                p.append(seg_feat[:PAST_STEPS])
                f.append(seg[PAST_STEPS:, :2])
            past_list.append(p)
            future_list.append(f)
    return np.array(past_list), np.array(future_list)

def load_and_prepare(data_dir='data'):
    scenes = np.load(f'{data_dir}/scenes.npy')
    labels = np.load(f'{data_dir}/labels.npy')
    past, future = make_windows(scenes)
    past, mean, std = normalize(past)
    split = int(len(past) * 0.8)
    return {
        'train_past': past[:split],
        'train_future': future[:split],
        'val_past': past[split:],
        'val_future': future[split:],
        'mean': mean, 'std': std,
        'labels': labels
    }

if __name__ == '__main__':
    data = load_and_prepare()
    print("Train past shape:", data['train_past'].shape)
    print("Train future shape:", data['train_future'].shape)