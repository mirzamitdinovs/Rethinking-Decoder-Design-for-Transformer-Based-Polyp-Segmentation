import numpy as np


def dice_coefficient(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    intersection = (pred * gt).sum()
    return (2.0 * intersection + 1e-8) / (pred.sum() + gt.sum() + 1e-8)


def iou_score(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    return (intersection + 1e-8) / (union + 1e-8)


def mae_score(pred, gt):
    return np.abs(pred - gt).mean()


def s_measure(pred, gt, alpha=0.5):
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    y = gt.mean()
    if y == 0:
        score = 1.0 - pred.mean()
        return max(score, 0.0)
    elif y == 1:
        score = pred.mean()
        return max(score, 0.0)

    so = _s_object(pred, gt)
    sr = _s_region(pred, gt)

    return alpha * so + (1 - alpha) * sr


def _s_object(pred, gt):
    fg = pred * gt
    bg = (1 - pred) * (1 - gt)

    u = gt.mean()
    sigma_x_fg = fg.mean()
    sigma_x_bg = bg.mean()

    score_fg = 2.0 * u * sigma_x_fg / (u * u + sigma_x_fg + 1e-8)
    score_bg = 2.0 * (1 - u) * sigma_x_bg / ((1 - u) ** 2 + sigma_x_bg + 1e-8)

    return u * score_fg + (1 - u) * score_bg


def _s_region(pred, gt):
    h, w = gt.shape
    x_center = int(round(w / 2))
    y_center = int(round(h / 2))

    gt1 = gt[:y_center, :x_center]
    gt2 = gt[:y_center, x_center:]
    gt3 = gt[y_center:, :x_center]
    gt4 = gt[y_center:, x_center:]

    pred1 = pred[:y_center, :x_center]
    pred2 = pred[:y_center, x_center:]
    pred3 = pred[y_center:, :x_center]
    pred4 = pred[y_center:, x_center:]

    w1 = _ssim(pred1, gt1)
    w2 = _ssim(pred2, gt2)
    w3 = _ssim(pred3, gt3)
    w4 = _ssim(pred4, gt4)

    total = gt1.size + gt2.size + gt3.size + gt4.size
    score = (w1 * gt1.size + w2 * gt2.size + w3 * gt3.size + w4 * gt4.size) / total

    return score


def _ssim(pred, gt):
    h, w = pred.shape
    n = h * w

    if n == 0:
        return 0.0

    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) ** 2).sum() / (n - 1 + 1e-8)
    sigma_y2 = ((gt - y) ** 2).sum() / (n - 1 + 1e-8)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (n - 1 + 1e-8)

    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        score = alpha / (beta + 1e-8)
    elif alpha == 0 and beta == 0:
        score = 1.0
    else:
        score = 0.0

    return max(score, 0.0)


def e_measure(pred, gt):
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    if gt.sum() == 0:
        enhanced_matrix = 1.0 - pred
    elif gt.mean() == 1:
        enhanced_matrix = pred
    else:
        mean_pred = pred.mean()
        mean_gt = gt.mean()

        align_pred = pred - mean_pred
        align_gt = gt - mean_gt

        align_matrix = 2 * align_gt * align_pred / (
            align_gt * align_gt + align_pred * align_pred + 1e-8
        )

        enhanced_matrix = (align_matrix + 1) ** 2 / 4

    score = enhanced_matrix.mean()
    return score


def weighted_f_measure(pred, gt, beta2=1.0):
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    if gt.sum() == 0:
        if pred.sum() == 0:
            return 1.0
        return 0.0

    from scipy.ndimage import distance_transform_edt
    dst = distance_transform_edt(1 - gt)
    idxt = np.where(gt > 0.5)

    if len(idxt[0]) == 0:
        return 0.0

    e = np.abs(pred - gt)
    et = e.copy()
    et[gt == 0] = et[gt == 0] * (1.0 - np.exp(-dst[gt == 0] / 5.0))

    tp_w = (1 - et[gt > 0.5]).sum()
    fp_w = et[gt <= 0.5].sum()
    fn_w = et[gt > 0.5].sum()

    precision = tp_w / (tp_w + fp_w + 1e-8)
    recall = tp_w / (tp_w + fn_w + 1e-8)

    wfm = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
    return wfm


def evaluate_prediction(pred, gt):
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    return {
        'dice': dice_coefficient(pred, gt),
        'iou': iou_score(pred, gt),
        'mae': mae_score(pred, gt),
        'sm': s_measure(pred, gt),
        'em': e_measure(pred, gt),
        'wfm': weighted_f_measure(pred, gt),
    }


def evaluate_dataset(predictions, ground_truths):
    all_metrics = []
    for pred, gt in zip(predictions, ground_truths):
        metrics = evaluate_prediction(pred, gt)
        all_metrics.append(metrics)

    mean_metrics = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        mean_metrics[f'mean_{key}'] = float(np.mean(values))

    return mean_metrics
