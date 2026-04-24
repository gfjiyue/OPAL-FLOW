from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

from aggregate import AggregateConfig, run_mode_aggregation
from infer import InferConfig, run_inference_on_root
from super_resolution import SRConfig, run_super_resolution_batch
from track import TrackConfig, run_tracking_pipeline


WORK_ROOT = Path(r"OPAL_FLOW_WORK")

HF_DATASET_REPO_ID = "njauyang/rice"
HF_DATASET_SUBDIR = "rice"

DET_MODEL_PATH = Path(r"trackbest.pt")

UPSCAYL_DOWNLOAD_URL = "https://github.com/upscayl/upscayl/releases"
SR_BACKEND = "upscayl"
SR_EXE = Path(r"Upscayl\resources\bin\upscayl-bin.exe")
SR_MODELS_DIR = Path(r"Program Files\Upscayl\resources\models")
SR_MODEL_NAME = "upscayl-standard-4x"
SR_SCALE = 4
SR_EXT = "jpg"
SR_CMD_TEMPLATE = None

CKPT_PATH = Path(r"panicletimebest.pt")
VMAE_MODEL_DIR = "OpenGVLab/VideoMAEv2-Base"

MIN_INFER_RATIO = 0.75
STRICT_RATIO = False
VAL_STRIDE = 2
CLEANUP_AFTER_RUN = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RUN_ROOT = WORK_ROOT / ".opal_flow_runtime"


def resolve_raw_image_dir() -> Path:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        dataset_root = Path(
            snapshot_download(
                repo_id=HF_DATASET_REPO_ID,
                repo_type="dataset",
            )
        )

    raw_dir = dataset_root / HF_DATASET_SUBDIR if HF_DATASET_SUBDIR else dataset_root

    if not raw_dir.exists():
        raise FileNotFoundError(f"Hugging Face dataset folder not found: {raw_dir}")

    return raw_dir


def prepare_temporal_checkpoint() -> Path:
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu")

    if not isinstance(ckpt, dict):
        return CKPT_PATH

    ckpt = dict(ckpt)
    ckpt_cfg = dict(ckpt.get("cfg", {}))
    ckpt_cfg["model"] = VMAE_MODEL_DIR
    ckpt["cfg"] = ckpt_cfg

    out_path = RUN_ROOT / "panicletimemae_hf_vmae.pt"
    torch.save(ckpt, out_path)
    return out_path


def count_raw_images(img_dir: Path) -> int:
    return sum(
        1 for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def filter_sequences(
    rotated_root: Path,
    reference_total_frames: int,
    ratio: float,
    strict: bool,
) -> Path:
    threshold = reference_total_frames * ratio

    for seq_dir in sorted([d for d in rotated_root.iterdir() if d.is_dir()], key=lambda p: p.name):
        n_images = sum(
            1 for p in seq_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )

        keep = n_images > threshold if strict else n_images >= threshold

        if not keep:
            shutil.rmtree(seq_dir, ignore_errors=True)

    return rotated_root


def run_stage(label: str, func):
    with tqdm(total=1, desc=label, unit="stage", file=sys.stdout, leave=True) as bar:
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                result = func()
        except Exception:
            log_text = buffer.getvalue()
            if log_text:
                print(log_text)
            raise
        bar.update(1)
    return result


def run_tracking_stage(raw_img_dir: Path, rotated_root: Path) -> Path:
    track_cfg = TrackConfig(
        model_path=str(DET_MODEL_PATH),
        img_dir=str(raw_img_dir),
        project_dir=str(RUN_ROOT / "runs"),
        save_vis=False,
        save_txt=True,
        rotated_output_dir=str(rotated_root),
        rotated_scale=1.05,
        force_portrait=True,
        auto_head_up=True,
        head_up_vote_frames=8,
        head_up_min_mask_area=80,
        invert_head_rule=False,
        rename_folders=True,
        rename_zero_pad=3,
        save_debug_json=False,
    )

    current_root = run_tracking_pipeline(track_cfg)

    return filter_sequences(
        rotated_root=current_root,
        reference_total_frames=count_raw_images(raw_img_dir),
        ratio=MIN_INFER_RATIO,
        strict=STRICT_RATIO,
    )


def run_super_resolution_stage(input_root: Path, sr_root: Path) -> Path:
    sr_cfg = SRConfig(
        backend=SR_BACKEND,
        sr_exe=str(SR_EXE),
        models_dir=str(SR_MODELS_DIR),
        model_name=SR_MODEL_NAME,
        scale=SR_SCALE,
        ext=SR_EXT,
        extra_args=[],
        cmd_template=SR_CMD_TEMPLATE,
    )

    return run_super_resolution_batch(
        input_root=input_root,
        output_root=sr_root,
        cfg=sr_cfg,
    )


def run_inference_stage(input_root: Path, infer_out: Path, temporal_ckpt_path: Path) -> dict:
    infer_cfg = InferConfig(
        ckpt_path=str(temporal_ckpt_path),
        model_dir=str(VMAE_MODEL_DIR),
        output_dir=str(infer_out),
        img_size=224,
        clip_len=16,
        val_stride=VAL_STRIDE,
        device=None,
        local_files_only=False,
        use_fasg=None,
        enforce_order=True,
        save_prob_csv=False,
        reference_total_frames=None,
        min_frame_ratio=0.0,
        strict_ratio=False,
    )

    pred_out_dir = run_inference_on_root(input_root, infer_cfg)

    agg_cfg = AggregateConfig(
        pred_json_path=str(pred_out_dir / "per_sequence_predictions.json"),
        output_dir=str(pred_out_dir),
        time_format="%Y-%m-%d %H:%M:%S",
    )

    summary_path = run_mode_aggregation(agg_cfg)
    return json.loads(Path(summary_path).read_text(encoding="utf-8"))


def validate_paths() -> None:
    if not DET_MODEL_PATH.exists():
        raise FileNotFoundError(f"Detection model not found: {DET_MODEL_PATH}")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Temporal model checkpoint not found: {CKPT_PATH}")

    if not SR_EXE.exists():
        raise FileNotFoundError(
            f"Upscayl executable not found: {SR_EXE}\n"
            f"Please install Upscayl from: {UPSCAYL_DOWNLOAD_URL}"
        )

    if not SR_MODELS_DIR.exists():
        raise FileNotFoundError(
            f"Upscayl models directory not found: {SR_MODELS_DIR}\n"
            f"Please install Upscayl from: {UPSCAYL_DOWNLOAD_URL}"
        )


def cleanup(work_root_existed: bool) -> None:
    if CLEANUP_AFTER_RUN:
        shutil.rmtree(RUN_ROOT, ignore_errors=True)

        if not work_root_existed:
            with contextlib.suppress(OSError):
                WORK_ROOT.rmdir()


def main() -> None:
    work_root_existed = WORK_ROOT.exists()

    validate_paths()

    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT, ignore_errors=True)

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    rotated_root = RUN_ROOT / "01_rotated_crop"
    sr_root = RUN_ROOT / "02_superres"
    infer_out = RUN_ROOT / "03_predictions"

    try:
        raw_img_dir = resolve_raw_image_dir()
        temporal_ckpt_path = prepare_temporal_checkpoint()

        tracked_root = run_stage(
            "Tracking",
            lambda: run_tracking_stage(raw_img_dir, rotated_root),
        )

        superres_root = run_stage(
            "Super-resolution",
            lambda: run_super_resolution_stage(tracked_root, sr_root),
        )

        summary = run_stage(
            "Inferring anthesis time",
            lambda: run_inference_stage(superres_root, infer_out, temporal_ckpt_path),
        )

        print(f"Tstart: {summary.get('Tstart_mode')}")
        print(f"Tpeak: {summary.get('Tpeak_mode')}")

    finally:
        cleanup(work_root_existed)


if __name__ == "__main__":
    main()