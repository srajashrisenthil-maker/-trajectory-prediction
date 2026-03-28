import matplotlib.pyplot as plt
import numpy as np

COLORS = ['#FF6B6B', '#4ECDC4', '#FFE66D']

def plot_prediction(past, gt_future, pred_trajs, probs, intent, risk,
                    attn_weights=None, neighbor_pasts=None, title='Prediction'):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    ax.plot(past[:, 0], past[:, 1], 'w--', lw=1.5, label='Past', alpha=0.7)
    ax.scatter(past[-1, 0], past[-1, 1], c='white', s=80, zorder=5)

    if neighbor_pasts is not None:
        for i, nbr in enumerate(neighbor_pasts):
            alpha = float(attn_weights[i]) * 2 if attn_weights is not None else 0.3
            ax.plot(nbr[:, 0], nbr[:, 1], color='#aaaaaa', lw=1, alpha=min(alpha, 0.8))
            if attn_weights is not None:
                ax.scatter(nbr[-1, 0], nbr[-1, 1],
                           s=60 + 100 * float(attn_weights[i]),
                           c='#aaaaaa', alpha=0.8, zorder=4)

    ax.plot(gt_future[:, 0], gt_future[:, 1], 'lime', lw=2, label='Ground truth', alpha=0.9)
    ax.scatter(gt_future[-1, 0], gt_future[-1, 1], c='lime', s=100, marker='*', zorder=6)

    for i, (traj, prob) in enumerate(zip(pred_trajs, probs)):
        ax.plot(traj[:, 0], traj[:, 1], color=COLORS[i], lw=2,
                alpha=0.5 + 0.5 * float(prob), label=f'Mode {i+1} ({100*float(prob):.0f}%)')
        ax.scatter(traj[-1, 0], traj[-1, 1], color=COLORS[i], s=80, zorder=5)

    risk_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    circle = plt.Circle(past[-1], radius=3, color=risk_colors[risk],
                        fill=True, alpha=0.1, lw=2, linestyle='--')
    ax.add_patch(circle)

    ax.set_title(f'{title}\nIntent: {intent}  |  Risk: {risk}',
                 color='white', fontsize=12)
    ax.legend(facecolor='#2a2a4e', labelcolor='white', fontsize=9)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    return fig

def plot_whatif(past, orig_pred, mod_pred, obstacle_pos, title='What-If Simulation'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#1a1a2e')
    for ax, preds, label in zip(axes, [orig_pred, mod_pred], ['Before', 'After']):
        ax.set_facecolor('#1a1a2e')
        ax.plot(past[:, 0], past[:, 1], 'w--', lw=1.5, alpha=0.7)
        ax.scatter(past[-1, 0], past[-1, 1], c='white', s=80, zorder=5)
        for i, traj in enumerate(preds):
            ax.plot(traj[:, 0], traj[:, 1], color=COLORS[i], lw=2, alpha=0.8)
        ax.scatter(*obstacle_pos, c='red', s=200, marker='X', zorder=7, label='Obstacle')
        ax.set_title(f'{label}', color='white', fontsize=11)
        ax.legend(facecolor='#2a2a4e', labelcolor='white')
        ax.tick_params(colors='white')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']:
            ax.spines[sp].set_color('white')
    fig.suptitle(title, color='white', fontsize=13)
    plt.tight_layout()
    return fig