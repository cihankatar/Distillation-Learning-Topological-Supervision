import numpy as np

def iou_and_dice(pred, gt):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    
    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()
    iou  = intersection / union if union > 0 else 0.0
    
    dice = 2 * intersection / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0.0
    return iou, dice
