import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

def detect_with_roi(model: YOLO, img_path_or_np, 
                    coarse_conf: float = 0.25, coarse_iou: float = 0.45,
                    crop_pad: float = 0.2, min_area: int = 32,
                    high_conf: float = 0.001,  # pass-through to high-res predict
                    device=None, verbose=False):
    """
    Two-stage ROI detection:
    1) Run a coarse pass to get proposals (lower imgsz is ok).
    2) For each coarse bbox, crop the original image with padding and run high-resolution detect.
    Returns concatenated Results from high-res crops (mapped to original image coords).
    """
    # Load image
    if isinstance(img_path_or_np, str):
        img0 = cv2.imread(img_path_or_np)[:, :, ::-1]  # BGR->RGB
    else:
        img0 = img_path_or_np.copy()
    H, W = img0.shape[:2]

    # Coarse detection (fast)
    coarse = model.predict(source=img0, imgsz=640, conf=coarse_conf, iou=coarse_iou, device=device, verbose=verbose)
    # coarse is a list of Results; take first result's boxes
    proposals = []
    if len(coarse) and len(coarse[0].boxes):
        boxes = coarse[0].boxes.xyxy.cpu().numpy()  # (N,4)
        scores = coarse[0].boxes.conf.cpu().numpy() if hasattr(coarse[0].boxes, "conf") else np.ones(len(boxes))
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area:
                # pad
                w = x2 - x1
                h = y2 - y1
                px = w * crop_pad
                py = h * crop_pad
                nx1 = max(0, int(x1 - px))
                ny1 = max(0, int(y1 - py))
                nx2 = min(W, int(x2 + px))
                ny2 = min(H, int(y2 + py))
                proposals.append((nx1, ny1, nx2, ny2))

    # If no proposals found, fallback to full-image detection
    final_results = []
    if not proposals:
        final_results = model.predict(source=img0, imgsz=max(H, W), conf=coarse_conf, device=device, verbose=verbose)
        return final_results

    # Run high-res detect per crop and remap boxes
    for (nx1, ny1, nx2, ny2) in proposals:
        crop = img0[ny1:ny2, nx1:nx2]
        # high-res inference on crop (keep small imgsz matching crop)
        high = model.predict(source=crop, imgsz= max(crop.shape[0], crop.shape[1]), conf=high_conf, device=device, verbose=verbose)
        if not len(high):
            continue
        res = high[0]
        # remap boxes back
        if hasattr(res, "boxes") and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else np.ones(len(xyxy))
            cls = res.boxes.cls.cpu().numpy().astype(int) if hasattr(res.boxes, "cls") else np.zeros(len(xyxy), dtype=int)
            remapped = []
            for (x1, y1, x2, y2), c, cl in zip(xyxy, confs, cls):
                rx1 = x1 + nx1
                ry1 = y1 + ny1
                rx2 = x2 + nx2
                ry2 = y2 + ny2
                remapped.append([rx1, ry1, rx2, ry2, c, cl])
            final_results.extend(remapped)
    return final_results