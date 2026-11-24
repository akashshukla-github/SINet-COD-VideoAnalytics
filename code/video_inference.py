"""
video_inference.py  --  Group extension for EE5340 Video Analytics (IIT Guwahati)

Extends the image-based SINet (Fan et al., CVPR 2020) to VIDEO:
  frame -> SINet camouflage map C_i(t) -> temporal EMA smoothing -> binary mask
        -> connected component -> bounding box + centroid track -> overlay video + CSV log

Temporal smoothing (exponential moving average over frames):
    S(t) = alpha * C(t) + (1 - alpha) * S(t-1),   S(0) = C(0)
Binarisation:   M(t) = 1[ S(t) >= tau ]
Centroid track: (x_c, y_c)(t) = mean of pixel coordinates where M(t) = 1
Displacement:   d(t) = || (x_c, y_c)(t) - (x_c, y_c)(t-1) ||_2   (pixels / frame)

Usage (run from inside the `code/` folder):
    python video_inference.py --video input.mp4 --model_path ./Snapshot/2020-CVPR-SINet/SINet_40.pth
    python video_inference.py --video input.mp4 --model_path SINet_40.pth --alpha 0.6 --tau 0.5 --cpu
Outputs (in --out_dir):  overlay.mp4, mask.mp4, track.csv, and a few sample frame PNGs.
"""
import argparse
import csv
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from Src.SINet import SINet_ResNet50

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model(model_path, device):
    # Skip the ImageNet initialisation (it downloads ResNet-50); we load the trained SINet weights instead.
    SINet_ResNet50.initialize_weights = lambda self: None
    model = SINet_ResNet50(channel=32)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def preprocess(frame_bgr, size):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


@torch.no_grad()
def predict(model, frame_bgr, size, device):
    h, w = frame_bgr.shape[:2]
    x = preprocess(frame_bgr, size).to(device)
    _, cam = model(x)                                   # use identification-stage map C_i
    cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=True)
    cam = cam.sigmoid().squeeze().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def largest_component(mask):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[idx]
    return (int(x), int(y), int(bw), int(bh)), int(area), (float(cents[idx][0]), float(cents[idx][1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--model_path', default='./Snapshot/2020-CVPR-SINet/SINet_40.pth')
    ap.add_argument('--out_dir', default='./VideoResult')
    ap.add_argument('--testsize', type=int, default=352)
    ap.add_argument('--alpha', type=float, default=0.6, help='EMA weight of current frame (1.0 = no smoothing)')
    ap.add_argument('--tau', type=float, default=0.5, help='binarisation threshold')
    ap.add_argument('--max_frames', type=int, default=0, help='0 = all frames')
    ap.add_argument('--save_every', type=int, default=30, help='save a sample PNG every N frames')
    ap.add_argument('--cpu', action='store_true')
    opt = ap.parse_args()

    device = torch.device('cpu' if opt.cpu or not torch.cuda.is_available() else 'cuda')
    os.makedirs(opt.out_dir, exist_ok=True)
    model = build_model(opt.model_path, device)

    cap = cv2.VideoCapture(opt.video)
    if not cap.isOpened():
        raise SystemExit('Cannot open video: ' + opt.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw_overlay = cv2.VideoWriter(os.path.join(opt.out_dir, 'overlay.mp4'), fourcc, fps, (W, H))
    vw_mask = cv2.VideoWriter(os.path.join(opt.out_dir, 'mask.mp4'), fourcc, fps, (W, H))

    csv_f = open(os.path.join(opt.out_dir, 'track.csv'), 'w', newline='')
    log = csv.writer(csv_f)
    log.writerow(['frame', 'detected', 'x', 'y', 'w', 'h', 'area_px', 'cx', 'cy', 'disp_px', 'mean_conf'])

    S, prev_c, t = None, None, 0
    while True:
        ok, frame = cap.read()
        if not ok or (opt.max_frames and t >= opt.max_frames):
            break
        C = predict(model, frame, opt.testsize, device)
        S = C if S is None else opt.alpha * C + (1.0 - opt.alpha) * S      # temporal EMA
        M = (S >= opt.tau)

        overlay = frame.copy()
        overlay[M] = (0.5 * overlay[M] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        comp = largest_component(M)
        row = [t, 0, '', '', '', '', 0, '', '', '', float(S.mean())]
        if comp is not None:
            (x, y, bw, bh), area, (cx, cy) = comp
            disp = '' if prev_c is None else float(np.hypot(cx - prev_c[0], cy - prev_c[1]))
            prev_c = (cx, cy)
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(overlay, (int(cx), int(cy)), 4, (255, 255, 0), -1)
            row = [t, 1, x, y, bw, bh, area, round(cx, 1), round(cy, 1),
                   '' if disp == '' else round(disp, 2), round(float(S[M].mean()), 4)]
        cv2.putText(overlay, 'frame %d' % t, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        log.writerow(row)

        heat = cv2.applyColorMap((S * 255).astype(np.uint8), cv2.COLORMAP_JET)
        vw_overlay.write(overlay)
        vw_mask.write(heat)
        if t % opt.save_every == 0:
            cv2.imwrite(os.path.join(opt.out_dir, 'frame_%05d.png' % t),
                        np.hstack([frame, heat, overlay]))
        t += 1
        if t % 25 == 0:
            print('[video] processed %d frames' % t)

    cap.release(); vw_overlay.release(); vw_mask.release(); csv_f.close()
    print('[done] %d frames -> %s' % (t, opt.out_dir))


if __name__ == '__main__':
    main()
