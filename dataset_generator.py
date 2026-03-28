import numpy as np
import os

np.random.seed(42)

def generate_straight(n_steps, start, speed):
    angle = np.random.uniform(0, 2*np.pi)
    vx, vy = speed * np.cos(angle), speed * np.sin(angle)
    x = start[0] + vx * np.arange(n_steps)
    y = start[1] + vy * np.arange(n_steps)
    return np.stack([x, y], axis=1)

def generate_turning(n_steps, start, speed):
    angle = np.random.uniform(0, 2*np.pi)
    turn_rate = np.random.choice([-0.1, 0.1])
    x, y = [start[0]], [start[1]]
    for i in range(1, n_steps):
        angle += turn_rate
        x.append(x[-1] + speed * np.cos(angle))
        y.append(y[-1] + speed * np.sin(angle))
    return np.array(list(zip(x, y)))

def generate_slowing(n_steps, start, speed):
    angle = np.random.uniform(0, 2*np.pi)
    x, y = [start[0]], [start[1]]
    for i in range(1, n_steps):
        s = max(0.01, speed * (1 - i / n_steps))
        x.append(x[-1] + s * np.cos(angle))
        y.append(y[-1] + s * np.sin(angle))
    return np.array(list(zip(x, y)))

def generate_random_walk(n_steps, start, speed):
    x, y = [start[0]], [start[1]]
    angle = np.random.uniform(0, 2*np.pi)
    for _ in range(1, n_steps):
        angle += np.random.normal(0, 0.3)
        x.append(x[-1] + speed * np.cos(angle))
        y.append(y[-1] + speed * np.sin(angle))
    return np.array(list(zip(x, y)))

PATTERNS = [generate_straight, generate_turning, generate_slowing, generate_random_walk]
PATTERN_NAMES = ['straight', 'turning', 'slowing', 'random']

def generate_dataset(n_scenes=500, n_agents=5, n_steps=50, save_dir='data'):
    os.makedirs(save_dir, exist_ok=True)
    scenes, labels = [], []
    for _ in range(n_scenes):
        scene = []
        scene_labels = []
        for _ in range(n_agents):
            pattern_idx = np.random.randint(4)
            start = np.random.uniform(-20, 20, 2)
            speed = np.random.uniform(0.3, 1.5)
            traj = PATTERNS[pattern_idx](n_steps, start, speed)
            scene.append(traj)
            scene_labels.append(pattern_idx)
        scenes.append(scene)
        labels.append(scene_labels)
    scenes = np.array(scenes)
    labels = np.array(labels)
    np.save(os.path.join(save_dir, 'scenes.npy'), scenes)
    np.save(os.path.join(save_dir, 'labels.npy'), labels)
    print(f"Saved {n_scenes} scenes. Shape: {scenes.shape}")
    return scenes, labels

if __name__ == '__main__':
    generate_dataset()