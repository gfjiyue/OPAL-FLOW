from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SRConfig:

                                   
    backend: str = "upscayl"

         
    sr_exe: str = ""

         
    models_dir: str = ""

               
    model_name: str = "upscayl-standard-4x"

    scale: int = 4
    ext: str = "jpg"

                           
    extra_args: list[str] = field(default_factory=list)

                      
    cmd_template: Optional[str] = None
    shell: bool = True


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def _copy_images_only(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in sorted(input_dir.iterdir()):
        if is_image_file(p):
            shutil.copy2(p, output_dir / p.name)
            count += 1
    print(f"[SR] backend=none, copied {count} images -> {output_dir}")


def _validate_upscayl_paths(cfg: SRConfig) -> None:
    if not cfg.sr_exe:
        raise ValueError("Upscayl executable path is empty. Please set SRConfig.sr_exe")
    if not cfg.models_dir:
        raise ValueError("Upscayl models_dir is empty. Please set SRConfig.models_dir")

    exe_path = Path(cfg.sr_exe)
    model_dir = Path(cfg.models_dir)

    if not exe_path.exists():
        raise FileNotFoundError(f"Upscayl executable not found: {exe_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Upscayl models directory not found: {model_dir}")


def _build_upscayl_cmd(input_dir: Path, output_dir: Path, cfg: SRConfig) -> list[str]:
    _validate_upscayl_paths(cfg)

    cmd = [
        str(cfg.sr_exe),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-m", str(cfg.models_dir),
        "-n", cfg.model_name,
        "-s", str(cfg.scale),
        "-f", cfg.ext,
    ]

    if cfg.extra_args:
        cmd.extend(cfg.extra_args)

    return cmd


def run_super_resolution_on_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    cfg: SRConfig,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input SR folder not found: {input_dir}")

    if cfg.backend == "none":
        _copy_images_only(input_dir, output_dir)
        return

    if cfg.backend == "upscayl":
        cmd = _build_upscayl_cmd(input_dir, output_dir, cfg)
        print("[SR] Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return

    if cfg.backend == "external_cmd":
        if not cfg.cmd_template:
            raise ValueError("backend='external_cmd' requires cmd_template")

        cmd = cfg.cmd_template.format(
            exe=cfg.sr_exe,
            input=str(input_dir),
            output=str(output_dir),
            model=cfg.model_name,
            scale=cfg.scale,
            ext=cfg.ext,
            models_dir=cfg.models_dir,
        )
        print("[SR] Running:", cmd)
        subprocess.run(cmd, check=True, shell=cfg.shell)
        return

    raise ValueError(f"Unsupported SR backend: {cfg.backend}")


def find_sequence_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    seq_dirs = [d for d in sorted(subdirs) if any(is_image_file(p) for p in d.iterdir())]

    if seq_dirs:
        return seq_dirs

    if any(is_image_file(p) for p in root.iterdir()):
        return [root]

    raise RuntimeError(f"No image folders found under: {root}")


def run_super_resolution_batch(
    input_root: str | Path,
    output_root: str | Path,
    cfg: SRConfig,
) -> Path:
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    seq_dirs = find_sequence_dirs(input_root)

    for seq_dir in seq_dirs:
        if seq_dir == input_root:
            out_dir = output_root
        else:
            out_dir = output_root / seq_dir.name

        print(f"\n[SR] Processing folder: {seq_dir.name}")
        run_super_resolution_on_folder(seq_dir, out_dir, cfg)

    print(f"\n[SR] All folders finished: {output_root}")
    return output_root
