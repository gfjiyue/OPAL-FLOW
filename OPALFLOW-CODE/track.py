from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class TrackConfig:

    model_path: str
    img_dir: str
    project_dir: str

    track_name: str = "obb/track"
    conf: float = 0.30
    iou: float = 0.50
    device: str = "0"
    tracker: str = "botsort.yaml"
    save_vis: bool = True
    save_txt: bool = True
    show_labels: bool = True
    show_conf: bool = False
    persist: bool = True
    retina_masks: bool = True
    stream: bool = False

    rotated_output_dir: Optional[str] = None
    rotated_scale: float = 1.0

                  
    force_portrait: bool = True

               
    auto_head_up: bool = True
    head_up_vote_frames: int = 8
    head_up_min_mask_area: int = 80

                           
    invert_head_rule: bool = False

              
    rename_folders: bool = True
    rename_zero_pad: int = 3

              
    save_debug_json: bool = True


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _iter_image_files(img_dir: Path):
    for p in sorted(img_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def imread_unicode(path: str | Path):
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext not in IMG_EXTS:
        ext = ".jpg"

    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Failed to encode image for saving: {path}")
    buf.tofile(str(path))


def scale_quad(pts: np.ndarray, scale: float = 1.0) -> np.ndarray:
    center = np.mean(pts, axis=0, keepdims=True)
    pts_scaled = center + (pts - center) * float(scale)
    return pts_scaled.astype(np.float32)


def obb_to_upright_crop(
    img: np.ndarray,
    pts_src: np.ndarray,
    scale: float = 1.0,
    force_portrait: bool = True,
) -> Tuple[Optional[np.ndarray], dict]:
    debug = {
        "ok": False,
        "orig_width": None,
        "orig_height": None,
        "rot90_applied": False,
        "reason": "",
    }

    if pts_src.shape != (4, 2):
        debug["reason"] = "pts_src shape invalid"
        return None, debug

    pts_scaled = scale_quad(pts_src, scale=scale)
    rect = order_points(pts_scaled)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    crop_w = int(round(max(width_a, width_b)))
    crop_h = int(round(max(height_a, height_b)))

    if crop_w < 2 or crop_h < 2:
        debug["reason"] = f"invalid crop size ({crop_w}, {crop_h})"
        return None, debug

    dst = np.array(
        [
            [0, 0],
            [crop_w - 1, 0],
            [crop_w - 1, crop_h - 1],
            [0, crop_h - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    crop = cv2.warpPerspective(
        img,
        M,
        (crop_w, crop_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    rot90_applied = False
    if force_portrait and crop_w > crop_h:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        rot90_applied = True

    debug["ok"] = True
    debug["orig_width"] = int(crop_w)
    debug["orig_height"] = int(crop_h)
    debug["rot90_applied"] = bool(rot90_applied)
    debug["reason"] = "ok"
    return crop, debug


def build_foreground_mask(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

               
    _, mask1 = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                  
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask2 = np.zeros_like(gray, dtype=np.uint8)
    cond_green = (h >= 20) & (h <= 105) & (v >= 25)
    cond_light = (s <= 140) & (v >= 70)
    mask2[(cond_green | cond_light)] = 255

    mask = cv2.bitwise_and(mask1, mask2)

                    
    if int(np.sum(mask > 0)) < 50:
        mask = mask1

                   
    if int(np.sum(mask > 0)) < 30:
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[gray > 10] = 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

           
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == best_idx] = 255
    return out


def crop_mask_bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return mask
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return mask[y1:y2 + 1, x1:x2 + 1]


def estimate_need_flip_180(
    img_bgr: np.ndarray,
    min_mask_area: int = 80,
) -> Optional[bool]:
    mask = build_foreground_mask(img_bgr)
    area = int(np.sum(mask > 0))
    if area < min_mask_area:
        return None

    mask = crop_mask_bbox(mask)
    h, w = mask.shape[:2]
    if h < 10 or w < 5:
        return None

    span = max(3, int(round(h * 0.30)))
    top_band = mask[:span, :]
    bottom_band = mask[h - span:, :]

    top_mass = float(np.sum(top_band > 0))
    bottom_mass = float(np.sum(bottom_band > 0))

              
    denom = max(top_mass, bottom_mass, 1.0)
    if abs(bottom_mass - top_mass) / denom < 0.05:
        return None

                                  
    return bottom_mass > top_mass


def auto_fix_head_up_in_instance_folders(
    rotated_root: str | Path,
    vote_frames: int = 8,
    min_mask_area: int = 80,
    invert_head_rule: bool = False,
    save_debug_json: bool = True,
) -> None:
    rotated_root = Path(rotated_root)
    if not rotated_root.exists():
        raise FileNotFoundError(f"Rotated crop directory not found: {rotated_root}")

    folder_summaries = []

    instance_dirs = [d for d in rotated_root.iterdir() if d.is_dir()]
    instance_dirs.sort(key=lambda x: x.name)

    for instance_dir in instance_dirs:
        img_files = [p for p in sorted(instance_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if not img_files:
            continue

        votes = []
        checked = 0

        for p in img_files[: min(vote_frames, len(img_files))]:
            img = imread_unicode(p)
            if img is None:
                continue

            v = estimate_need_flip_180(img, min_mask_area=min_mask_area)
            checked += 1
            if v is not None:
                votes.append(bool(v))

        if len(votes) == 0:
            flip_all = False
        else:
            flip_all = (sum(votes) > (len(votes) / 2.0))

        if invert_head_rule:
            flip_all = not flip_all

        if flip_all:
            for p in img_files:
                img = imread_unicode(p)
                if img is None:
                    continue
                img = cv2.rotate(img, cv2.ROTATE_180)
                imwrite_unicode(p, img)

        summary = {
            "folder": instance_dir.name,
            "n_images": len(img_files),
            "checked_frames": checked,
            "valid_votes": len(votes),
            "true_votes_need_flip": int(sum(votes)) if len(votes) > 0 else 0,
            "flip_all_180": bool(flip_all),
        }
        folder_summaries.append(summary)

        print(
            f"[INFO] Head-up vote: {instance_dir.name} | "
            f"checked={checked}, valid_votes={len(votes)}, "
            f"need_flip_votes={int(sum(votes)) if len(votes) > 0 else 0}, "
            f"flip_all={flip_all}"
        )

    if save_debug_json:
        debug_path = rotated_root / "_head_up_debug.json"
        debug_path.write_text(
            json.dumps(folder_summaries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[INFO] Head-up debug saved: {debug_path}")


def rotated_crop_from_txt(
    img_dir: str | Path,
    txt_dir: str | Path,
    output_base: str | Path,
    scale: float = 1.0,
    force_portrait: bool = True,
    save_debug_json: bool = True,
) -> None:
    img_dir = Path(img_dir)
    txt_dir = Path(txt_dir)
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    debug_records = []

    for img_path in _iter_image_files(img_dir):
        prefix = img_path.stem
        txt_path = txt_dir / f"{prefix}.txt"

        if not txt_path.exists():
            print(f"[WARN] Missing label file: {txt_path}")
            continue

        img = imread_unicode(img_path)
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        h, w = img.shape[:2]
        lines = txt_path.read_text(encoding="utf-8").splitlines()

        for line_idx, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if len(parts) < 9:
                print(f"[WARN] Bad label line ({txt_path.name}:{line_idx}): {line}")
                continue

            try:
                coords = list(map(float, parts[1:9]))
            except ValueError:
                print(f"[WARN] Non-numeric coords ({txt_path.name}:{line_idx}): {line}")
                continue

            instance_id = parts[9] if len(parts) >= 10 else f"line_{line_idx:04d}"
            instance_dir = output_base / str(instance_id)
            instance_dir.mkdir(parents=True, exist_ok=True)

            points = []
            for i in range(0, 8, 2):
                x = coords[i] * w
                y = coords[i + 1] * h
                x = min(max(x, 0), w - 1)
                y = min(max(y, 0), h - 1)
                points.append([x, y])
            pts_src = np.array(points, dtype=np.float32)

            cropped, dbg = obb_to_upright_crop(
                img=img,
                pts_src=pts_src,
                scale=scale,
                force_portrait=force_portrait,
            )

            if cropped is None or cropped.size == 0:
                print(f"[WARN] Empty/invalid crop for {img_path.name}, id={instance_id}")
                debug_records.append(
                    {
                        "image": img_path.name,
                        "instance_id": str(instance_id),
                        "ok": False,
                        **dbg,
                    }
                )
                continue

            save_name = f"{prefix}_ID{instance_id}.jpg"
            save_path = instance_dir / save_name
            imwrite_unicode(save_path, cropped)

            debug_records.append(
                {
                    "image": img_path.name,
                    "instance_id": str(instance_id),
                    "ok": True,
                    "save_path": str(save_path),
                    **dbg,
                }
            )

    if save_debug_json:
        debug_path = output_base / "_crop_debug.json"
        debug_path.write_text(
            json.dumps(debug_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"[INFO] Upright rotated crop finished: {output_base}")


def sort_and_rename_instance_folders(
    rotated_root: str | Path,
    zero_pad: int = 3,
    dry_run: bool = False,
) -> None:
    rotated_root = Path(rotated_root)
    if not rotated_root.is_dir():
        raise FileNotFoundError(f"Rotated crop directory not found: {rotated_root}")

    folders = [d for d in rotated_root.iterdir() if d.is_dir()]
    stats = []
    for folder in folders:
        n_imgs = sum(
            1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS
        )
        stats.append((folder.name, n_imgs))

    stats.sort(key=lambda x: x[1], reverse=True)

    print("\n[INFO] Folder ranking by image count:")
    for rank, (name, cnt) in enumerate(stats, start=1):
        print(f"{rank:03d}. {name} -> {cnt} images")

    if dry_run:
        print("[INFO] dry_run=True, rename skipped.")
        return

    for rank, (name, _cnt) in enumerate(stats, start=1):
        old_path = rotated_root / name
        tmp_name = f"__tmp_{rank:0{zero_pad}d}_{name}"
        os.rename(old_path, rotated_root / tmp_name)

    for rank, (name, _cnt) in enumerate(stats, start=1):
        tmp_name = f"__tmp_{rank:0{zero_pad}d}_{name}"
        final_name = f"{rank:0{zero_pad}d}_{name}"
        os.rename(rotated_root / tmp_name, rotated_root / final_name)

    print("[INFO] Folder rename finished.")


def run_yolo_tracking(cfg: TrackConfig) -> Tuple[Path, Path]:
    img_dir = Path(cfg.img_dir)
    project_dir = Path(cfg.project_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not Path(cfg.model_path).exists():
        raise FileNotFoundError(f"Model file not found: {cfg.model_path}")

    model = YOLO(cfg.model_path)
    results = model.track(
        source=str(img_dir),
        show=False,
        save=cfg.save_vis,
        conf=cfg.conf,
        show_labels=cfg.show_labels,
        show_conf=cfg.show_conf,
        persist=cfg.persist,
        save_txt=cfg.save_txt,
        iou=cfg.iou,
        tracker=cfg.tracker,
        retina_masks=cfg.retina_masks,
        device=cfg.device,
        stream=cfg.stream,
        project=str(project_dir),
        name=cfg.track_name,
    )

                                                   
    if cfg.stream:
        results = list(results)

    if not results:
        raise RuntimeError("YOLO returned empty results.")

    save_dir = Path(str(results[0].save_dir))
    txt_dir = save_dir / "labels"
    if not txt_dir.exists():
        raise RuntimeError(f"Tracking finished but label directory not found: {txt_dir}")

    print(f"[INFO] YOLO save_dir: {save_dir}")
    print(f"[INFO] YOLO txt_dir : {txt_dir}")
    return save_dir, txt_dir


def run_tracking_pipeline(cfg: TrackConfig) -> Path:
    _save_dir, txt_dir = run_yolo_tracking(cfg)

    output_base = Path(cfg.rotated_output_dir) if cfg.rotated_output_dir else Path(cfg.img_dir) / "rotated_crop"

    rotated_crop_from_txt(
        img_dir=cfg.img_dir,
        txt_dir=txt_dir,
        output_base=output_base,
        scale=cfg.rotated_scale,
        force_portrait=cfg.force_portrait,
        save_debug_json=cfg.save_debug_json,
    )

    if cfg.auto_head_up:
        auto_fix_head_up_in_instance_folders(
            rotated_root=output_base,
            vote_frames=cfg.head_up_vote_frames,
            min_mask_area=cfg.head_up_min_mask_area,
            invert_head_rule=cfg.invert_head_rule,
            save_debug_json=cfg.save_debug_json,
        )

    if cfg.rename_folders:
        sort_and_rename_instance_folders(
            rotated_root=output_base,
            zero_pad=cfg.rename_zero_pad,
            dry_run=False,
        )

    return output_base
