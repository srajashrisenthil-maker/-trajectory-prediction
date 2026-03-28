import numpy as np

def ade(pred, gt):
    return np.linalg.norm(pred - gt, axis=1).mean()

def fde(pred, gt):
    return np.linalg.norm(pred[-1] - gt[-1])

def min_ade(preds, gt):
    return min(ade(p, gt) for p in preds)

def min_fde(preds, gt):
    return min(fde(p, gt) for p in preds)

def evaluate_batch(all_preds, all_gt):
    ades, fdes, mades, mfdes = [], [], [], []
    for preds, gt in zip(all_preds, all_gt):
        best = preds[0]
        ades.append(ade(best, gt))
        fdes.append(fde(best, gt))
        mades.append(min_ade(preds, gt))
        mfdes.append(min_fde(preds, gt))
    return {
        'ADE':    np.mean(ades),
        'FDE':    np.mean(fdes),
        'minADE': np.mean(mades),
        'minFDE': np.mean(mfdes),
    }