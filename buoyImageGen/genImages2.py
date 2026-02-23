import os
import cv2
import json
import math
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

# -----------------------------
# CONFIG
# -----------------------------
GREEN_IMG = "./images/LightBeacon_Green_Above_Pool.JPG"
RED_IMG   = "./images/LightBeacon_Red_Above_Pool.JPG"

#OUT_DIR = Path(r"C:\Users\Precision 3571\OneDrive\Desktop\buoyImageGen\synthetic_buoy_dataset_v2")
OUT_DIR = Path("./synthetic_buoy_dataset_vs")
N_IMAGES = 10
OUT_W, OUT_H = 1280, 720

# Labels
CLASS_NAMES = ["buoy"]          # single-class detector (color can be separate classifier)
CLASS_ID = 0

# If you want 2 classes instead, set:
# CLASS_NAMES = ["green_buoy", "red_buoy"]
# and use CLASS_ID = 0/1 based on chosen buoy.

# Placement + rejection
MIN_BBOX_AREA_FRAC = 0.003      # reject if bbox area < this fraction of image area
MAX_ALPHA_CUT_FRAC = 0.60       # reject if too much of buoy alpha is cut off by edges

# Augmentation strength
SCALE_RANGE = (0.55, 1.10)
ROT_RANGE_DEG = (-12, 12)
SHEAR_RANGE = (-0.07, 0.07)

ENABLE_GLARE = True
ENABLE_RIPPLE = True
ENABLE_MOTION_BLUR = True

# Performance
#NUM_WORKERS = max(1, min(cpu_count() - 1, 8))
NUM_WORKERS = 1

# Reproducibility
BASE_SEED = 7


# -----------------------------
# Utils: photometric
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
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def color_jitter(img):
    alpha = random.uniform(0.75, 1.35)  # contrast
    beta  = random.uniform(-25, 25)     # brightness
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    out = apply_gamma(out, random.uniform(0.75, 1.35))

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + random.randint(-8, 8)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-35, 35), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + random.randint(-20, 20), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def maybe_blur(img):
    if random.random() < 0.55:
        k = random.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (k, k), random.uniform(0.6, 2.2))
    return img

def maybe_fog(img):
    if random.random() < 0.25:
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
# Utils: geometry/composite
# -----------------------------
def warp_affine_rgba(rgba, angle_deg, scale, shear=0.0):
    h, w = rgba.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)

    if abs(shear) > 1e-6:
        S = np.array([[1.0, shear, 0.0],
                      [0.0, 1.0,  0.0]], dtype=np.float32)
        M = S @ np.vstack([M, [0, 0, 1]])
        M = M[:2, :]

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

    M2 = M.copy()
    M2[0, 2] -= minx
    M2[1, 2] -= miny

    out = cv2.warpAffine(rgba, M2, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0,0))
    return out

def alpha_paste(bg_bgr, fg_rgba, x, y):
    bg = bg_bgr.copy()
    bh, bw = bg.shape[:2]
    fh, fw = fg_rgba.shape[:2]

    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + fw, bw)
    y1 = min(y + fh, bh)
    if x0 >= x1 or y0 >= y1:
        return bg, None, 1.0  # fully cut

    fg_crop = fg_rgba[(y0 - y):(y1 - y), (x0 - x):(x1 - x)]
    alpha = fg_crop[..., 3:4].astype(np.float32) / 255.0
    fg_bgr = fg_crop[..., :3].astype(np.float32)
    bg_roi = bg[y0:y1, x0:x1].astype(np.float32)

    comp = fg_bgr * alpha + bg_roi * (1.0 - alpha)
    bg[y0:y1, x0:x1] = np.clip(comp, 0, 255).astype(np.uint8)

    # bbox from alpha
    a = fg_crop[..., 3]
    ys, xs = np.where(a > 10)
    if len(xs) == 0 or len(ys) == 0:
        return bg, None, 1.0

    bx0 = int(x0 + xs.min())
    by0 = int(y0 + ys.min())
    bx1 = int(x0 + xs.max())
    by1 = int(y0 + ys.max())

    # fraction of alpha kept
    total_alpha = np.count_nonzero(fg_rgba[...,3] > 10)
    kept_alpha  = np.count_nonzero(fg_crop[...,3] > 10)
    cut_frac = 1.0 - (kept_alpha / max(total_alpha, 1))
    return bg, (bx0, by0, bx1, by1), cut_frac

def yolo_line_from_bbox(bbox, img_w, img_h, cls=0):
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w  = (x1 - x0)
    h  = (y1 - y0)
    return f"{cls} {cx/img_w:.6f} {cy/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}\n"

def bbox_area_frac(bbox, img_w, img_h):
    x0, y0, x1, y1 = bbox
    area = max(0, x1-x0) * max(0, y1-y0)
    return area / float(img_w * img_h)

def clamp_randint(a, b):
    """Return random int in [a,b] but never fail."""
    if a > b:
        a, b = b, a
    return random.randint(a, b)

def ensure_fits(rgba, out_w, out_h):
    """If warped buoy is bigger than canvas, downscale to fit."""
    h, w = rgba.shape[:2]
    sx = out_w / max(w, 1)
    sy = out_h / max(h, 1)
    s = min(sx, sy, 1.0)
    if s < 1.0:
        new_w = max(1, int(w * s))
        new_h = max(1, int(h * s))
        rgba = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return rgba


# -----------------------------
# Realism extras
# -----------------------------
def motion_blur(img, ksize=15, angle_deg=0.0):
    """Linear motion blur kernel rotated by angle."""
    ksize = int(max(3, ksize))
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize//2, :] = 1.0
    kernel /= kernel.sum()

    M = cv2.getRotationMatrix2D((ksize/2, ksize/2), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    kernel /= max(kernel.sum(), 1e-6)

    return cv2.filter2D(img, -1, kernel)

def add_glare_bloom(img_bgr, center_xy, strength=0.35, radius=90):
    """Adds a soft bloom spot near the buoy light."""
    h, w = img_bgr.shape[:2]
    cx, cy = center_xy
    cx = int(np.clip(cx, 0, w-1))
    cy = int(np.clip(cy, 0, h-1))

    overlay = np.zeros((h, w), dtype=np.float32)
    cv2.circle(overlay, (cx, cy), int(radius), 1.0, -1)
    overlay = cv2.GaussianBlur(overlay, (0,0), sigmaX=radius/3)

    # Convert overlay into 3-channel and blend additively
    out = img_bgr.astype(np.float32)
    add = (overlay[..., None] * 255.0 * strength)
    out = np.clip(out + add, 0, 255)
    return out.astype(np.uint8)

def ripple_distort(img_bgr, amp=3.0, wl=120.0):
    """Subtle water-like ripple warp across image."""
    h, w = img_bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    phase = random.uniform(0, 2*np.pi)
    dx = amp * np.sin(2*np.pi * yy / wl + phase)
    dy = amp * np.cos(2*np.pi * xx / (wl*1.3) + phase)

    map_x = np.clip(xx + dx, 0, w-1).astype(np.float32)
    map_y = np.clip(yy + dy, 0, h-1).astype(np.float32)

    return cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# -----------------------------
# GrabCut cutout
# -----------------------------
def auto_rect_for_buoy(img_bgr):
    h, w = img_bgr.shape[:2]
    rx = int(w * 0.36)
    ry = int(h * 0.25)
    rw = int(w * 0.28)
    rh = int(h * 0.48)
    return (rx, ry, rw, rh)

def make_cutout_rgba(img_bgr, rect):
    img = img_bgr.copy()
    mask = np.zeros(img.shape[:2], np.uint8)

    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgModel, fgModel, 7, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    fg = cv2.medianBlur(fg, 5)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = fg
    return rgba

def make_cutout_rgba_color_seed(img_bgr, buoy_color: str):
    """
    buoy_color: "green" or "red"
    Uses HSV thresholds to seed sure-foreground from the illuminated lens,
    then runs GrabCut with a mask init so it doesn't grab the deck.
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if buoy_color == "green":
        # green lens range (tune if needed)
        lower = np.array([35, 80, 80])
        upper = np.array([95, 255, 255])
        seed = cv2.inRange(hsv, lower, upper)
    else:
        # red lens wraps HSV hue, so use two ranges
        lower1 = np.array([0, 90, 70])
        upper1 = np.array([12, 255, 255])
        lower2 = np.array([165, 90, 70])
        upper2 = np.array([179, 255, 255])
        seed = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # Keep only the biggest connected component (the lens)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((seed > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        # fallback: rectangle-based grabcut if seeding fails
        rect = auto_rect_for_buoy(img)
        return make_cutout_rgba(img, rect)

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    seed = (labels == largest).astype(np.uint8) * 255

    # Grow seed so it covers lens + some buoy body
    seed_fg = cv2.dilate(seed, np.ones((25, 25), np.uint8), iterations=1)
    seed_fg = cv2.GaussianBlur(seed_fg, (0, 0), 7)

    # Initialize GrabCut mask:
    # GC_BGD=0, GC_FGD=1, GC_PR_BGD=2, GC_PR_FGD=3
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # sure foreground where seed is strong
    gc_mask[seed_fg > 40] = cv2.GC_PR_FGD
    gc_mask[seed > 0] = cv2.GC_FGD

    # sure background: image borders are almost never buoy
    border = 15
    gc_mask[:border, :] = cv2.GC_BGD
    gc_mask[-border:, :] = cv2.GC_BGD
    gc_mask[:, :border] = cv2.GC_BGD
    gc_mask[:, -border:] = cv2.GC_BGD

    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, gc_mask, None, bgModel, fgModel, 7, cv2.GC_INIT_WITH_MASK)

    alpha = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Cleanup: remove small junk and smooth edges
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=2)

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = alpha
    return rgba

def add_buoy_reflection(bg_bgr, buoy_rgba, bbox, strength=0.35):
    """
    Creates a reflection from the buoy itself:
    - flip vertically
    - squash vertically
    - ripple distort
    - fade out with a vertical gradient
    - blur slightly
    """
    x0, y0, x1, y1 = bbox
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)

    # Extract the buoy region from the composed image by using bbox-sized reflection of buoy_rgba
    # Better: build reflection from buoy_rgba itself by cropping its non-transparent bbox.
    a = buoy_rgba[..., 3]
    ys, xs = np.where(a > 10)
    if len(xs) == 0:
        return bg_bgr

    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max(), ys.max()
    buoy_crop = buoy_rgba[by0:by1+1, bx0:bx1+1].copy()

    # Flip vertically to reflect
    refl = cv2.flip(buoy_crop, 0)

    # Squash reflection (water reflections are vertically compressed)
    squash = random.uniform(0.35, 0.65)
    refl = cv2.resize(refl, (refl.shape[1], max(1, int(refl.shape[0] * squash))), interpolation=cv2.INTER_LINEAR)

    # Apply ripple to reflection only
    if random.random() < 0.9:
        rh, rw = refl.shape[:2]
        yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
        phase = random.uniform(0, 2*np.pi)
        amp = random.uniform(1.0, 4.0)
        wl  = random.uniform(25.0, 60.0)
        dx = amp * np.sin(2*np.pi * yy / wl + phase)
        map_x = np.clip(xx + dx, 0, rw-1).astype(np.float32)
        map_y = yy.astype(np.float32)
        refl = cv2.remap(refl, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Fade out with distance (alpha gradient)
    rh, rw = refl.shape[:2]
    grad = np.linspace(1.0, 0.0, rh).astype(np.float32)[:, None]
    refl[..., 3] = np.clip(refl[..., 3].astype(np.float32) * grad * (255.0 * strength / 255.0), 0, 255).astype(np.uint8)

    # Blur reflection
    refl_bgr = refl[..., :3]
    refl_a   = refl[..., 3]
    refl_bgr = cv2.GaussianBlur(refl_bgr, (0,0), sigmaX=random.uniform(1.5, 3.5))
    refl = cv2.cvtColor(refl_bgr, cv2.COLOR_BGR2BGRA)
    refl[..., 3] = refl_a

    # Place reflection directly below buoy bbox
    # Reflection "starts" near bottom of buoy bbox
    place_x = x0 + int(0.05 * w)
    place_y = y1 - int(0.05 * h)

    comp, _, _ = alpha_paste(bg_bgr, refl, place_x, place_y)
    return comp

# -----------------------------
# Background crops
# -----------------------------
def random_water_crop(img_bgr, out_w, out_h):
    h, w = img_bgr.shape[:2]
    y_min = int(h * 0.60)
    y_max = int(h * 0.95)

    crop_w = random.randint(int(w * 0.45), int(w * 0.98))
    crop_h = random.randint(int(h * 0.30), int(h * 0.60))

    x0 = random.randint(0, max(0, w - crop_w))
    y0 = random.randint(y_min, max(y_min, y_max - crop_h))

    patch = img_bgr[y0:y0+crop_h, x0:x0+crop_w]
    patch = cv2.resize(patch, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return patch


# -----------------------------
# Worker job
# -----------------------------
@dataclass
class Assets:
    green_bgr: np.ndarray
    red_bgr: np.ndarray
    green_rgba: np.ndarray
    red_rgba: np.ndarray

ASSETS = None  # populated by init_worker()

def init_worker(seed, green_path, red_path):
    global ASSETS
    random.seed(seed)
    np.random.seed(seed)

    g = cv2.imread(green_path, cv2.IMREAD_COLOR)
    r = cv2.imread(red_path,   cv2.IMREAD_COLOR)
    if g is None:
        raise FileNotFoundError(f"Could not load GREEN_IMG at: {green_path}")
    if r is None:
        raise FileNotFoundError(f"Could not load RED_IMG at: {red_path}")

    # Use the images you just loaded (g and r)
    green_rgba = make_cutout_rgba_color_seed(g, "green")
    red_rgba   = make_cutout_rgba_color_seed(r, "red")

    ASSETS = Assets(green_bgr=g, red_bgr=r, green_rgba=green_rgba, red_rgba=red_rgba)

def generate_one(i):
    # Balanced sampling: alternate green/red
    is_green = (i % 2 == 0)
    name = "green" if is_green else "red"

    src_bgr  = ASSETS.green_bgr if is_green else ASSETS.red_bgr
    buoy_rgba = ASSETS.green_rgba if is_green else ASSETS.red_rgba

    # background from either image
    bg_src = ASSETS.green_bgr if random.random() < 0.5 else ASSETS.red_bgr
    bg = random_water_crop(bg_src, OUT_W, OUT_H)

    # buoy photometric aug
    buoy_bgr = buoy_rgba[..., :3]
    buoy_a   = buoy_rgba[..., 3]

    buoy_bgr = color_jitter(buoy_bgr)
    buoy_bgr = maybe_blur(buoy_bgr)
    buoy_bgr = add_noise(buoy_bgr, sigma=random.uniform(0, 8))
    buoy_bgr = maybe_fog(buoy_bgr)

    buoy_aug = cv2.cvtColor(buoy_bgr, cv2.COLOR_BGR2BGRA)
    buoy_aug[..., 3] = buoy_a

    # geometry aug
    angle = random.uniform(*ROT_RANGE_DEG)
    scale = random.uniform(*SCALE_RANGE)
    shear = random.uniform(*SHEAR_RANGE)

    buoy_warp = warp_affine_rgba(buoy_aug, angle, scale, shear=shear)
    buoy_warp = ensure_fits(buoy_warp, OUT_W, OUT_H)
    fh, fw = buoy_warp.shape[:2]

    # SAFE placement ranges (never empty)
    xmin = -int(fw * 0.10)
    xmax = OUT_W - int(fw * 0.90)
    ymin = int(OUT_H * 0.35)
    ymax = OUT_H - int(fh * 0.90)

    # Clamp if needed
    if xmin >= xmax:
        xmin, xmax = 0, max(0, OUT_W - fw)
    if ymin >= ymax:
        ymin, ymax = 0, max(0, OUT_H - fh)

    x = clamp_randint(xmin, xmax)
    y = clamp_randint(ymin, ymax)

    comp, bbox, cut_frac = alpha_paste(bg, buoy_warp, x, y)

    if bbox is not None:
        comp = add_buoy_reflection(comp, buoy_warp, bbox, strength=random.uniform(0.20, 0.45))
    if bbox is None:
        return None  # reject

    # Rejection checks
    if bbox_area_frac(bbox, OUT_W, OUT_H) < MIN_BBOX_AREA_FRAC:
        return None
    if cut_frac > MAX_ALPHA_CUT_FRAC:
        return None

    # Optional ripple (subtle)
    if ENABLE_RIPPLE and random.random() < 0.55:
        comp = ripple_distort(comp, amp=random.uniform(1.0, 3.5), wl=random.uniform(90, 160))

    # Global post effects
    comp = color_jitter(comp)
    comp = maybe_vignette(comp)
    comp = add_noise(comp, sigma=random.uniform(0, 6))

    # Motion blur
    if ENABLE_MOTION_BLUR and random.random() < 0.35:
        k = random.choice([9, 13, 17, 21])
        ang = random.uniform(0, 180)
        comp = motion_blur(comp, ksize=k, angle_deg=ang)

    comp = maybe_blur(comp)
    comp = jpeg_artifacts(comp, quality=random.randint(35, 95))

    # Glare bloom near bbox top-center (approx buoy light position)
    if ENABLE_GLARE and random.random() < 0.35:
        x0, y0, x1, y1 = bbox
        cx = int((x0 + x1) / 2)
        cy = int(y0 + 0.18 * (y1 - y0))
        comp = add_glare_bloom(comp, (cx, cy),
                               strength=random.uniform(0.15, 0.45),
                               radius=random.uniform(50, 110))

    # Output paths
    stem = f"{i:06d}_{name}"
    return {
        "stem": stem,
        "image": comp,
        "bbox": bbox,
        "class_id": CLASS_ID,  # change to 0/1 if using 2-class setup
        "meta": {
            "i": i,
            "name": name,
            "cut_frac": float(cut_frac),
            "bbox_area_frac": float(bbox_area_frac(bbox, OUT_W, OUT_H)),
            "angle_deg": float(angle),
            "scale": float(scale),
            "shear": float(shear)
        }
    }


# -----------------------------
# Main
# -----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = OUT_DIR / "images"
    lbl_dir = OUT_DIR / "labels"
    img_dir.mkdir(exist_ok=True)
    lbl_dir.mkdir(exist_ok=True)

    (OUT_DIR / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n")

    meta_path = OUT_DIR / "meta.jsonl"
    if meta_path.exists():
        meta_path.unlink()

    # Generate with multiprocessing; retry rejected samples
    target = N_IMAGES
    produced = 0
    attempted = 0

    with Pool(processes=NUM_WORKERS,
              initializer=init_worker,
              initargs=(BASE_SEED, GREEN_IMG, RED_IMG)) as pool:

        # We generate in chunks; rejected ones return None; we keep going until produced == target.
        chunk = max(1, target//5)
        while produced < target:
            # schedule indices for this chunk
            idxs = list(range(attempted, attempted + chunk))
            attempted += chunk

            for result in pool.imap_unordered(generate_one, idxs):
                if result is None:
                    continue

                stem = result["stem"]
                img  = result["image"]
                bbox = result["bbox"]
                cls  = result["class_id"]

                out_img = img_dir / f"{stem}.jpg"
                out_lbl = lbl_dir / f"{stem}.txt"

                cv2.imwrite(str(out_img), img)
                with open(out_lbl, "w", encoding="utf-8") as f:
                    f.write(yolo_line_from_bbox(bbox, OUT_W, OUT_H, cls=cls))

                with open(meta_path, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps(result["meta"]) + "\n")

                produced += 1
                if produced % 250 == 0:
                    print(f"Produced {produced}/{target} (attempted {attempted})")

                if produced >= target:
                    break

    print(f"\nDone.\n  Output: {OUT_DIR}\n  Images: {img_dir}\n  Labels: {lbl_dir}\n  Meta:   {meta_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCTRL+C detected — stopping workers...")