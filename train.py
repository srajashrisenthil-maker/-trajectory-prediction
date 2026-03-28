import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.model import TrajectoryModel
import os

def best_of_n_loss(pred_trajs, gt, probs):
    gt_exp = gt.unsqueeze(1).expand_as(pred_trajs)
    ade = ((pred_trajs - gt_exp) ** 2).sum(-1).sqrt().mean(-1)
    min_ade, best_mode = ade.min(dim=1)
    prob_loss = -torch.log(probs[torch.arange(len(probs)), best_mode] + 1e-8)
    return min_ade.mean() + 0.1 * prob_loss.mean()

def train(epochs=20, batch_size=32, lr=1e-3, data_dir='data', save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    from data.preprocessing import load_and_prepare
    data = load_and_prepare(data_dir)
    past_t   = torch.tensor(data['train_past'],   dtype=torch.float32)
    future_t = torch.tensor(data['train_future'], dtype=torch.float32)
    past_v   = torch.tensor(data['val_past'],     dtype=torch.float32)
    future_v = torch.tensor(data['val_future'],   dtype=torch.float32)

    train_pos = past_t[:, :, -1, :2]
    val_pos   = past_v[:, :, -1, :2]

    train_loader = DataLoader(TensorDataset(past_t, future_t, train_pos), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(past_v, future_v, val_pos),   batch_size=batch_size)

    model = TrajectoryModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    intent_criterion = nn.CrossEntropyLoss()

    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for past_b, fut_b, pos_b in train_loader:
            trajs, probs, intent, shift, _ = model(past_b, pos_b)
            ego_fut = fut_b[:, 0]
            vel_diff = (past_b[:, 0, -1, 2:4] - past_b[:, 0, 0, 2:4]).norm(dim=1)
            intent_lbl = (vel_diff > 0.5).long()
            intent_loss = intent_criterion(intent, intent_lbl)
            loss = best_of_n_loss(trajs, ego_fut, probs) + 0.1 * intent_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for past_b, fut_b, pos_b in val_loader:
                trajs, probs, intent, shift, _ = model(past_b, pos_b)
                val_loss += best_of_n_loss(trajs, fut_b[:, 0], probs).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | train: {total_loss/len(train_loader):.4f} | val: {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f'{save_dir}/best_model.pt')
    return model