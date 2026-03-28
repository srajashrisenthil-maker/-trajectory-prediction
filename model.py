import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, h = self.gru(x)
        return h.squeeze(0)

class SocialAttention(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, ego_feat, neighbor_feats, positions):
        B, n_nbr, H = neighbor_feats.shape
        ego_exp = ego_feat.unsqueeze(1).expand_as(neighbor_feats)
        combined = torch.cat([ego_exp, neighbor_feats], dim=-1)
        raw_scores = self.attn(combined).squeeze(-1)
        ego_pos = positions[:, 0:1, :]
        nbr_pos = positions[:, 1:,  :]
        dists = torch.norm(ego_pos - nbr_pos, dim=-1)
        raw_scores = raw_scores - 0.1 * dists
        weights = F.softmax(raw_scores, dim=-1).unsqueeze(-1)
        social_ctx = (weights * neighbor_feats).sum(dim=1)
        return social_ctx, weights.squeeze(-1)

class MultiModalDecoder(nn.Module):
    def __init__(self, hidden_dim=64, future_steps=30, n_modes=3):
        super().__init__()
        self.n_modes = n_modes
        self.future_steps = future_steps
        self.traj_head   = nn.Linear(hidden_dim * 2, n_modes * future_steps * 2)
        self.prob_head   = nn.Linear(hidden_dim * 2, n_modes)
        self.intent_head = nn.Linear(hidden_dim * 2, 4)
        self.shift_head  = nn.Linear(hidden_dim * 2, 2)

    def forward(self, fused):
        B = fused.size(0)
        trajs  = self.traj_head(fused).view(B, self.n_modes, self.future_steps, 2)
        probs  = F.softmax(self.prob_head(fused), dim=-1)
        intent = self.intent_head(fused)
        shift  = self.shift_head(fused)
        return trajs, probs, intent, shift

class TrajectoryModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, future_steps=30, n_modes=3):
        super().__init__()
        self.encoder = TrajectoryEncoder(input_dim, hidden_dim)
        self.social  = SocialAttention(hidden_dim)
        self.decoder = MultiModalDecoder(hidden_dim, future_steps, n_modes)

    def forward(self, past, positions):
        B, A, T, F_dim = past.shape
        all_feats = torch.stack([self.encoder(past[:, a]) for a in range(A)], dim=1)
        ego_feat  = all_feats[:, 0]
        nbr_feats = all_feats[:, 1:]
        social_ctx, attn_weights = self.social(ego_feat, nbr_feats, positions)
        fused = torch.cat([ego_feat, social_ctx], dim=-1)
        trajs, probs, intent, shift = self.decoder(fused)
        return trajs, probs, intent, shift, attn_weights