"""
test_images.py -- modern (PyTorch 2.x / scipy>=1.3 compatible) replacement for MyTest.py.
Runs pretrained SINet on a test dataset, saves prediction maps + side-by-side panels,
and computes MAE and a simple max-F / IoU so our own numbers can go into the report.

    python test_images.py --model_path SINet_40.pth --dataset COD10K --limit 300
Writes:
    ./Result/<dataset>/<name>.png            prediction maps
    ../results/panel_<k>.png                  image | GT | prediction | overlay (first --panels images)
    ../results/our_metrics.json               metrics shown automatically on the project webpage
"""
import argparse, json, os, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from video_inference import build_model, predict

ap = argparse.ArgumentParser()
ap.add_argument('--model_path', default='./Snapshot/2020-CVPR-SINet/SINet_40.pth')
ap.add_argument('--data_root', default='./Dataset/TestDataset')
ap.add_argument('--dataset', default='COD10K')
ap.add_argument('--testsize', type=int, default=352)
ap.add_argument('--limit', type=int, default=0, help='0 = all images')
ap.add_argument('--panels', type=int, default=4)
ap.add_argument('--cpu', action='store_true')
opt = ap.parse_args()

device = torch.device('cpu' if opt.cpu or not torch.cuda.is_available() else 'cuda')
model = build_model(opt.model_path, device)
img_dir = os.path.join(opt.data_root, opt.dataset, 'Image')
gt_dir = os.path.join(opt.data_root, opt.dataset, 'GT')
save_dir = os.path.join('Result', opt.dataset); os.makedirs(save_dir, exist_ok=True)
os.makedirs('../results', exist_ok=True)

names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png')))
if opt.limit:
    names = names[:opt.limit]

maes, ious, fms, per_img, t0 = [], [], [], [], time.time()
for k, n in enumerate(names):
    img = cv2.imread(os.path.join(img_dir, n))
    gt_path = os.path.join(gt_dir, os.path.splitext(n)[0] + '.png')
    gt = cv2.imread(gt_path, 0)
    if img is None or gt is None:
        continue
    if gt.shape != img.shape[:2]:
        img = cv2.resize(img, (gt.shape[1], gt.shape[0]))
    pred = predict(model, img, opt.testsize, device)
    g = (gt > 127).astype(np.float32)
    cv2.imwrite(os.path.join(save_dir, os.path.splitext(n)[0] + '.png'), (pred * 255).astype(np.uint8))

    mae = float(np.abs(pred - g).mean())
    b = pred >= 0.5
    inter, union = float((b & (g > 0)).sum()), float((b | (g > 0)).sum())
    iou = inter / union if union > 0 else 1.0
    tp = inter; prec = tp / (b.sum() + 1e-8); rec = tp / (g.sum() + 1e-8)
    fm = 1.3 * prec * rec / (0.3 * prec + rec + 1e-8)          # beta^2 = 0.3 (standard in SOD/COD)
    maes.append(mae); ious.append(iou); fms.append(fm)
    per_img.append({'name': n, 'mae': round(mae, 4), 'iou': round(iou, 4)})

    if k < opt.panels:
        gt3 = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
        pr3 = cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        ov = img.copy(); ov[b] = (0.5 * ov[b] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        panel = np.hstack([img, gt3, pr3, ov])
        scale = 1400.0 / panel.shape[1]
        cv2.imwrite('../results/panel_%d.png' % (k + 1), cv2.resize(panel, None, fx=scale, fy=scale))

dt = time.time() - t0
per_img.sort(key=lambda d: d['mae'], reverse=True)
out = {
    'dataset': opt.dataset, 'num_images': len(maes), 'device': str(device),
    'MAE': round(float(np.mean(maes)), 4), 'IoU@0.5': round(float(np.mean(ious)), 4),
    'F_beta@0.5': round(float(np.mean(fms)), 4), 'sec_per_image': round(dt / max(len(maes), 1), 4),
    'worst_5': per_img[:5],
}
json.dump(out, open('../results/our_metrics.json', 'w'), indent=2)
print(json.dumps(out, indent=2))

# Save the 2 worst cases as failure-case panels for the report
for j, d in enumerate(out['worst_5'][:2]):
    n = d['name']
    img = cv2.imread(os.path.join(img_dir, n)); gt = cv2.imread(os.path.join(gt_dir, os.path.splitext(n)[0] + '.png'), 0)
    pr = cv2.imread(os.path.join(save_dir, os.path.splitext(n)[0] + '.png'), 0)
    if img is None or gt is None or pr is None:
        continue
    if gt.shape != img.shape[:2]:
        img = cv2.resize(img, (gt.shape[1], gt.shape[0]))
    panel = np.hstack([img, cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR), cv2.cvtColor(pr, cv2.COLOR_GRAY2BGR)])
    scale = 1200.0 / panel.shape[1]
    cv2.imwrite('../results/failure_%d.png' % (j + 1), cv2.resize(panel, None, fx=scale, fy=scale))
