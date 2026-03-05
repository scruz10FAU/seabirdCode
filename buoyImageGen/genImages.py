import os
import cv2
import json
import math
import random
import numpy as np
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
GREEN_IMG = "./images/LightBeacon_Green_Above_Pool.JPG"
RED_IMG   = "./images/LightBeacon_Red_Above_Pool.JPG"

OUT_DIR = Path("./synthetic_buoy_dataset")
N_IMAGES = 2000                 # how many to generate
OUT_W, OUT_H = 1280, 720        # output resolution

# YOLO: one class named "buoy"
CLASS_ID = 0

random.seed(7)
np.random.seed(7)

# -----------------------------
# Utility: photometric effects
# -----------------------------
def apply_gamma(img, gamma):
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.arange(256) / 255.0) ** inv
    table = np.clip(table * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, table)

def add_noise(img, sigma):
    if sigma <= 0:
        return img
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def jpeg_artifacts(img, quality):
    quality = int(np.clip(quality, 20, 100))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, enc = cv2.imencode(".jpg", img, encode_param)
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec

def color_jitter(img):
    # brightness/contrast
    alpha = random.uniform(0.75, 1.35)  # contrast
    beta  = random.uniform(-25, 25)     # brightness
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # gamma
    out = apply_gamma(out, random.uniform(0.75, 1.35))

    # HSV shift (mild)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + random.randint(-8, 8)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-35, 35), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + random.randint(-20, 20), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

def maybe_blur(img):
    if random.random() < 0.55:
        k = random.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (k, k), random.uniform(0.6, 2.2))
    return img

def maybe_fog(img):
    if random.random() < 0.25:
        # simple haze/fog overlay
        fog = np.full_like(img, 255, dtype=np.uint8)
        alpha = random.uniform(0.05, 0.25)
        return cv2.addWeighted(img, 1 - alpha, fog, alpha, 0)
    return img

def maybe_vignette(img):
    if random.random() < 0.35:
        rows, cols = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, random.uniform(cols/2.2, cols/1.4))
        kernel_y = cv2.getGaussianKernel(rows, random.uniform(rows/2.2, rows/1.4))
        kernel = kernel_y @ kernel_x.T
        mask = kernel / kernel.max()
        out = img.astype(np.float32)
        out[..., 0] *= mask
        out[..., 1] *= mask
        out[..., 2] *= mask
        return np.clip(out, 0, 255).astype(np.uint8)
    return img

# -----------------------------
# Utility: geometry & compositing
# -----------------------------
def warp_affine_rgba(rgba, angle_deg, scale, shear=0.0):
    h, w = rgba.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)

    # Add a tiny shear to make it less "perfect"
    if abs(shear) > 1e-6:
        S = np.array([[1.0, shear, 0.0],
                      [0.0, 1.0,  0.0]], dtype=np.float32)
        M = S @ np.vstack([M, [0, 0, 1]])
        M = M[:2, :]

    # compute new bounds
    corners = np.array([[0, 0, 1],
                        [w, 0, 1],
                        [w, h, 1],
                        [0, h, 1]], dtype=np.float32).T
    warped = (np.vstack([M, [0, 0, 1]]) @ corners).T
    xs, ys = warped[:, 0], warped[:, 1]
    minx, miny = xs.min(), ys.min()
    maxx, maxy = xs.max(), ys.max()

    new_w = int(math.ceil(maxx - minx))
    new_h = int(math.ceil(maxy - miny))

    # translate so all coords positive
    M2 = M.copy()
    M2[0, 2] -= minx
    M2[1, 2] -= miny

    out = cv2.warpAffine(rgba, M2, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0,0))
    return out

def alpha_paste(bg_bgr, fg_rgba, x, y):
    """Paste fg_rgba onto bg_bgr at (x,y). Returns bg copy + bbox of pasted alpha."""
    bg = bg_bgr.copy()
    bh, bw = bg.shape[:2]
    fh, fw = fg_rgba.shape[:2]

    # clip paste region to background
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + fw, bw)
    y1 = min(y + fh, bh)
    if x0 >= x1 or y0 >= y1:
        return bg, None

    fg_crop = fg_rgba[(y0 - y):(y1 - y), (x0 - x):(x1 - x)]
    alpha = fg_crop[..., 3:4].astype(np.float32) / 255.0
    fg_bgr = fg_crop[..., :3].astype(np.float32)
    bg_roi = bg[y0:y1, x0:x1].astype(np.float32)

    comp = fg_bgr * alpha + bg_roi * (1.0 - alpha)
    bg[y0:y1, x0:x1] = np.clip(comp, 0, 255).astype(np.uint8)

    # bbox from alpha mask
    a = fg_crop[..., 3]
    ys, xs = np.where(a > 10)
    if len(xs) == 0 or len(ys) == 0:
        return bg, None
    bx0 = int(x0 + xs.min())
    by0 = int(y0 + ys.min())
    bx1 = int(x0 + xs.max())
    by1 = int(y0 + ys.max())
    return bg, (bx0, by0, bx1, by1)

def yolo_line_from_bbox(bbox, img_w, img_h, cls=0):
    x0, y0, x1, y1 = bbox
    # YOLO expects center_x, center_y, w, h normalized 0-1
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w  = (x1 - x0)
    h  = (y1 - y0)
    return f"{cls} {cx/img_w:.6f} {cy/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}\n"

# -----------------------------
# Build buoy cutouts (RGBA) with GrabCut
# -----------------------------
def make_cutout_rgba(img_bgr, rect):
    """rect = (x,y,w,h) initial GrabCut rectangle around buoy."""
    img = img_bgr.copy()
    mask = np.zeros(img.shape[:2], np.uint8)

    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgModel, fgModel, 7, cv2.GC_INIT_WITH_RECT)

    # mask: 0/2 = bg, 1/3 = fg
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # clean edges a bit
    fg = cv2.medianBlur(fg, 5)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = fg
    return rgba

def auto_rect_for_buoy(img):
    """
    You can tune this if GrabCut misses.
    For your photos, buoy is near center-bottom; we create a reasonable default rectangle.
    """
    h, w = img.shape[:2]
    # rectangle roughly around the buoy + base (above pool edge)
    rx = int(w * 0.38)
    ry = int(h * 0.28)
    rw = int(w * 0.24)
    rh = int(h * 0.42)
    return (rx, ry, rw, rh)

# -----------------------------
# Background library from your own images
# -----------------------------
def random_water_crop(img_bgr, out_w, out_h):
    """
    Crops a random patch (likely water) from the lower portion of the source image,
    then resizes to output size.
    """
    h, w = img_bgr.shape[:2]
    # bias crops to bottom area (more water)
    y_min = int(h * 0.45)
    y_max = int(h * 0.90)
    x_min = 0
    x_max = w

    # ensure crop size
    crop_w = random.randint(int(w * 0.45), int(w * 0.95))
    crop_h = random.randint(int(h * 0.30), int(h * 0.55))

    x0 = random.randint(x_min, max(x_min, x_max - crop_w))
    y0 = random.randint(y_min, max(y_min, y_max - crop_h))

    patch = img_bgr[y0:y0+crop_h, x0:x0+crop_w]
    patch = cv2.resize(patch, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return patch

# -----------------------------
# Main generation
# -----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = OUT_DIR / "images"
    lbl_dir = OUT_DIR / "labels"
    img_dir.mkdir(exist_ok=True)
    lbl_dir.mkdir(exist_ok=True)

    green = cv2.imread(GREEN_IMG, cv2.IMREAD_COLOR)
    red   = cv2.imread(RED_IMG,   cv2.IMREAD_COLOR)
    if green is None or red is None:
        raise FileNotFoundError("Could not load one of the input images. Check paths at top of script.")

    # Create buoy cutouts
    green_rgba = make_cutout_rgba(green, auto_rect_for_buoy(green))
    red_rgba   = make_cutout_rgba(red,   auto_rect_for_buoy(red))

    sources = [
        ("green", green, green_rgba),
        ("red",   red,   red_rgba),
    ]

    for i in range(N_IMAGES):
        name, src_bgr, buoy_rgba = random.choice(sources)

        # background from either source image (keeps texture consistent)
        bg = random_water_crop(random.choice([green, red]), OUT_W, OUT_H)

        # randomize buoy appearance (lighting etc.)
        buoy_bgr = buoy_rgba[..., :3]
        buoy_a   = buoy_rgba[..., 3]
        buoy_bgr = color_jitter(buoy_bgr)
        buoy_bgr = maybe_blur(buoy_bgr)
        buoy_bgr = add_noise(buoy_bgr, sigma=random.uniform(0, 8))
        buoy_bgr = maybe_fog(buoy_bgr)

        buoy_aug = cv2.cvtColor(buoy_bgr, cv2.COLOR_BGR2BGRA)
        buoy_aug[..., 3] = buoy_a

        # geometry: scale/rotate a bit
        angle = random.uniform(-10, 10)
        scale = random.uniform(0.55, 0.95)
        shear = random.uniform(-0.05, 0.05)
        buoy_warp = warp_affine_rgba(buoy_aug, angle, scale, shear=shear)

        # place buoy on bg (keep it in lower half)
        fh, fw = buoy_warp.shape[:2]

        # --- Ensure buoy isn't bigger than canvas ---
        max_scale_x = OUT_W / max(fw, 1)
        max_scale_y = OUT_H / max(fh, 1)
        max_scale = min(max_scale_x, max_scale_y, 1.0)

        if max_scale < 1.0:
            buoy_warp = cv2.resize(
                buoy_warp,
                (int(fw * max_scale), int(fh * max_scale)),
                interpolation=cv2.INTER_LINEAR
            )
            fh, fw = buoy_warp.shape[:2]

        # --- Safe placement ranges ---
        xmin = -int(fw * 0.10)
        xmax = OUT_W - int(fw * 0.90)

        ymin = int(OUT_H * 0.35)
        ymax = OUT_H - int(fh * 0.90)

        # Clamp so randint never receives invalid bounds
        if xmin >= xmax:
            xmin, xmax = 0, max(0, OUT_W - fw)

        if ymin >= ymax:
            ymin, ymax = 0, max(0, OUT_H - fh)

        x = random.randint(xmin, xmax)
        y = random.randint(ymin, ymax)

        comp, bbox = alpha_paste(bg, buoy_warp, x, y)

        # final post effects on full image
        comp = color_jitter(comp)
        comp = maybe_vignette(comp)
        comp = add_noise(comp, sigma=random.uniform(0, 6))
        comp = maybe_blur(comp)
        comp = jpeg_artifacts(comp, quality=random.randint(35, 95))

        # Save
        stem = f"{i:06d}_{name}"
        out_img = img_dir / f"{stem}.jpg"
        out_lbl = lbl_dir / f"{stem}.txt"

        cv2.imwrite(str(out_img), comp)

        # If bbox failed, write empty label (or skip). Here we skip saving label if missing.
        if bbox is not None:
            with open(out_lbl, "w") as f:
                f.write(yolo_line_from_bbox(bbox, OUT_W, OUT_H, cls=CLASS_ID))
        else:
            # no buoy detected in alpha (rare). Save empty label file for consistency.
            out_lbl.write_text("")

        if (i + 1) % 200 == 0:
            print(f"Generated {i+1}/{N_IMAGES}")

    # Write classes file for convenience
    (OUT_DIR / "classes.txt").write_text("buoy\n")
    print(f"\nDone. Dataset in: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()