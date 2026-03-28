import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data.dataset_generator import generate_dataset
from data.preprocessing import load_and_prepare
from models.model import TrajectoryModel
from models.train import train
from modules.risk_intent import detect_intent, compute_risk
from modules.whatif import insert_virtual_obstacle
from viz.visualize import plot_prediction, plot_whatif
from eval.metrics import evaluate_batch

print("Generating dataset...")
generate_dataset(n_scenes=300)

print("\nTraining model...")
model = train(epochs=15, batch_size=32)

model.eval()
data = load_and_prepare()
past_t   = torch.tensor(data['val_past'][:1],   dtype=torch.float32)
future_t = torch.tensor(data['val_future'][:1], dtype=torch.float32)
pos_t    = past_t[:, :, -1, :2]

with torch.no_grad():
    trajs, probs, intent_logits, shift_logits, attn = model(past_t, pos_t)

trajs_np  = trajs[0].numpy()
probs_np  = probs[0].numpy()
past_np   = data['val_past'][0, 0, :, :2]
future_np = data['val_future'][0, 0]
nbr_past  = data['val_past'][0, 1:, :, :2]
attn_np   = attn[0].numpy()

intent, shift = detect_intent(past_np)
risk, dist    = compute_risk(trajs_np, ego_vehicle_pos=np.array([0.0, 0.0]))

print(f"\nIntent: {intent} | Shift detected: {shift} | Risk: {risk} (dist={dist:.2f})")

obstacle = past_np[-1] + np.array([1.5, 1.5])

def simple_pred(past, pos):
    from data.preprocessing import add_features
    past_feat = add_features(past)
    t = torch.tensor(past_feat[None, None], dtype=torch.float32)
    t = t.expand(-1, 5, -1, -1)
    p = torch.tensor(pos[None, None], dtype=torch.float32).expand(-1, 5, -1)
    with torch.no_grad():
        out, _, _, _, _ = model(t, p)
    return out[0].numpy()
orig_preds = simple_pred(past_np, past_np[-1])
mod_past   = insert_virtual_obstacle(past_np, obstacle)
mod_preds  = simple_pred(mod_past, mod_past[-1])

os.makedirs('outputs', exist_ok=True)

fig1 = plot_prediction(past_np, future_np, trajs_np, probs_np,
                       intent, risk, attn_np, nbr_past)
fig1.savefig('outputs/prediction.png', dpi=120, bbox_inches='tight')

fig2 = plot_whatif(past_np, orig_preds, mod_preds, obstacle)
fig2.savefig('outputs/whatif.png', dpi=120, bbox_inches='tight')
plt.show()

all_preds, all_gt = [], []
for i in range(min(100, len(data['val_past']))):
    pt = torch.tensor(data['val_past'][i:i+1], dtype=torch.float32)
    pp = pt[:, :, -1, :2]
    with torch.no_grad():
        t_out, _, _, _, _ = model(pt, pp)
    all_preds.append(t_out[0].numpy())
    all_gt.append(data['val_future'][i, 0])

metrics = evaluate_batch(all_preds, all_gt)
print("\n── Evaluation ──────────────────────────")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")