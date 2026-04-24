from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class InferConfig:
    ckpt_path: str
    model_dir: str
    output_dir: str

    img_size: int = 224
    clip_len: int = 16
    val_stride: int = 1
    device: Optional[str] = None
    local_files_only: bool = True
    use_fasg: Optional[bool] = None
    enforce_order: bool = True
    save_prob_csv: bool = True
    reference_total_frames: Optional[int] = None
    min_frame_ratio: float = 0.0
    strict_ratio: bool = False


@dataclass
class ResolvedModelConfig:
    model_name: str
    img_size: int
    clip_len: int
    use_fasg: bool
    fasg_short: int
    fasg_long: int
    fasg_gamma: float


def is_img(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def extract_ts_from_name(name: str) -> Optional[datetime]:
    match = re.search(r"\d{14}", name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(0), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def sort_key(path: Path):
    ts = extract_ts_from_name(path.name)
    return (ts is None, ts if ts else path.name)


def list_frames(seq_dir: Path) -> list[Path]:
    frames = [p for p in seq_dir.iterdir() if is_img(p)]
    frames.sort(key=sort_key)
    return frames


def list_sequence_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    seq_dirs = [d for d in sorted(subdirs) if len(list_frames(d)) > 0]
    if seq_dirs:
        return seq_dirs

    if len(list_frames(root)) > 0:
        return [root]

    raise RuntimeError(f"No sequence folders found under: {root}")


class PyramidDilatedTCN(nn.Module):
    def __init__(self, hidden: int, dilations=(1, 2, 4, 8), drop: float = 0.05):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=d, dilation=d, groups=hidden, bias=False),
                    nn.GELU(),
                    nn.Conv1d(hidden, hidden, kernel_size=1, bias=False),
                )
                for d in dilations
            ]
        )
        self.fuse = nn.Conv1d(hidden * len(dilations), hidden, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        outs = [branch(xt) for branch in self.branches]
        y = self.fuse(torch.cat(outs, dim=1))
        y = self.dropout(y)
        return (y + xt).transpose(1, 2)


class DepthwiseSmoother(nn.Module):
    def __init__(self, hidden: int, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=pad, groups=hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        xs = self.conv(xt)
        return 0.5 * (x + xs.transpose(1, 2))


class SGFeatureFilter(nn.Module):
    def __init__(self, coeff: list[float]):
        super().__init__()
        w = torch.tensor(coeff, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("w_base", w, persistent=False)
        self.K = int(w.shape[-1])
        assert self.K % 2 == 1, "SG window must be odd"
        self.pad = self.K // 2

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        _, _, c = z.shape
        x = z.transpose(1, 2)
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        w = self.w_base.expand(c, 1, self.K)
        y = F.conv1d(x, w, bias=None, stride=1, padding=0, groups=c)
        return y.transpose(1, 2)


class FASG(nn.Module):
    def __init__(self, short_win=3, long_win=7, gate_gamma=1.0):
        super().__init__()

        if short_win == 3:
            coeff3 = [-1.0, 3.0, -1.0]
        else:
            raise ValueError("short_win only supports 3")

        if long_win == 7:
            coeff7 = [-2 / 21, 3 / 21, 6 / 21, 7 / 21, 6 / 21, 3 / 21, -2 / 21]
        else:
            raise ValueError("long_win only supports 7")

        self.sg_short = SGFeatureFilter(coeff3)
        self.sg_long = SGFeatureFilter(coeff7)
        self.gate_gamma = float(gate_gamma)

    def forward(self, z_tcn: torch.Tensor, head_start: nn.Module, head_peak: nn.Module) -> torch.Tensor:
        p_s0 = torch.sigmoid(head_start(z_tcn).squeeze(-1))
        p_p0 = torch.sigmoid(head_peak(z_tcn).squeeze(-1))

        conf = torch.maximum(p_s0, p_p0).clamp(0.0, 1.0)
        if self.gate_gamma != 1.0:
            conf = conf ** self.gate_gamma

        alpha_short = conf
        alpha_long = 1.0 - alpha_short

        z_short = self.sg_short(z_tcn)
        z_long = self.sg_long(z_tcn)

        z_out = alpha_short.unsqueeze(-1) * z_short + alpha_long.unsqueeze(-1) * z_long
        return z_out


class VideoMAE2StartPeak(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze_backbone: bool = False,
        use_pdtcn: bool = True,
        use_feat_smooth: bool = True,
        use_fasg: bool = True,
        fasg_short: int = 3,
        fasg_long: int = 7,
        fasg_gamma: float = 1.0,
        local_files_only: bool = True,
    ):
        super().__init__()
        self.v2 = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if freeze_backbone:
            for p in self.v2.parameters():
                p.requires_grad = False

        cfg = getattr(self.v2, "config", None)
        self.hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "embed_dim", 768)))
        self.patch = int(getattr(cfg, "patch_size", 16))
        self.tube = int(getattr(cfg, "tubelet_size", 2))
        self.img_sz = getattr(cfg, "image_size", 224)
        self.num_frames = int(getattr(cfg, "num_frames", 16))

        if isinstance(self.img_sz, (list, tuple)):
            h, w = self.img_sz
        else:
            h = w = self.img_sz

        self.grid_h = h // self.patch
        self.grid_w = w // self.patch
        self.spatial_tokens = self.grid_h * self.grid_w

        self.ln = nn.LayerNorm(self.hidden)
        self.temporal = PyramidDilatedTCN(self.hidden, dilations=(1, 2, 4, 8), drop=0.05) if use_pdtcn else nn.Identity()
        self.feat_smooth = DepthwiseSmoother(self.hidden, kernel=5) if use_feat_smooth else None

        self.head_start = nn.Linear(self.hidden, 1)
        self.head_peak = nn.Linear(self.hidden, 1)

        self.fasg = FASG(fasg_short, fasg_long, gate_gamma=fasg_gamma) if use_fasg else None

        self.act = nn.GELU()
        self.drop = nn.Dropout(0.10)

    def _temporal_resample(self, clip: torch.Tensor, target_t: int) -> torch.Tensor:
        _, t = clip.shape[:2]
        if t == target_t:
            return clip
        idxs = torch.linspace(0, t - 1, target_t, device=clip.device).round().long()
        return clip[:, idxs, ...]

    def _v2_tokens_ts(self, clip: torch.Tensor) -> torch.Tensor:
        target_t = self.num_frames
        clip = self._temporal_resample(clip, target_t)
        x = clip.permute(0, 2, 1, 3, 4).contiguous()

        v2 = self.v2
        if hasattr(v2, "patch_embed"):
            tokens = v2.patch_embed(x)
        else:
            tokens = v2.model.patch_embed(x)

        pos = None
        if hasattr(v2, "pos_embed"):
            pos = v2.pos_embed.to(x.device)
        elif hasattr(v2, "model") and hasattr(v2.model, "pos_embed"):
            pos = v2.model.pos_embed.to(x.device)

        if pos is not None:
            if pos.shape[1] == tokens.shape[1]:
                tokens = tokens + pos
            elif pos.shape[1] == tokens.shape[1] + 1 and hasattr(v2, "cls_token"):
                cls = v2.cls_token.expand(tokens.shape[0], -1, -1).to(x.device)
                tokens = torch.cat((cls, tokens), dim=1) + pos
            else:
                if hasattr(v2, "interpolate_pos_encoding"):
                    tokens = v2.interpolate_pos_encoding(tokens)
                else:
                    min_len = min(pos.shape[1], tokens.shape[1])
                    tokens = tokens[:, :min_len, :] + pos[:, :min_len, :]

        blocks = getattr(v2, "blocks", None) or getattr(getattr(v2, "model", None), "blocks", None)
        norm = getattr(v2, "norm", None) or getattr(getattr(v2, "model", None), "norm", None)

        if blocks is None:
            feats = v2.forward_features(x)
            tokens = feats["last_hidden_state"] if isinstance(feats, dict) else feats
        else:
            for blk in blocks:
                tokens = blk(tokens)
            if norm is not None:
                tokens = norm(tokens)

        batch, length, dim = tokens.shape
        if self.spatial_tokens > 0 and (length % self.spatial_tokens == 1):
            tokens = tokens[:, 1:, :]
            length -= 1

        assert length % self.spatial_tokens == 0, "Token length cannot be reshaped to [T, S]"
        time_tokens = length // self.spatial_tokens
        tokens = tokens.view(batch, time_tokens, self.spatial_tokens, dim)

        repeat = max(1, self.tube)
        if repeat > 1:
            tokens = tokens.unsqueeze(2).repeat(1, 1, repeat, 1, 1).view(
                batch, time_tokens * repeat, self.spatial_tokens, dim
            )

        if tokens.shape[1] != target_t:
            b0, t0, s0, d0 = tokens.shape
            ch = tokens.permute(0, 2, 3, 1).reshape(b0, s0 * d0, t0)
            ch_up = F.interpolate(ch, size=target_t, mode="linear", align_corners=False)
            tokens = ch_up.reshape(b0, s0, d0, target_t).permute(0, 3, 1, 2).contiguous()

        return tokens

    def forward(self, clip: torch.Tensor):
        tok_ts = self._v2_tokens_ts(clip)
        x_pool = tok_ts.mean(dim=2)

        x = self.ln(x_pool)
        z_tcn = self.temporal(x)
        if self.feat_smooth is not None:
            z_tcn = self.feat_smooth(z_tcn)

        if self.fasg is not None:
            z = self.fasg(z_tcn, self.head_start, self.head_peak)
        else:
            z = z_tcn

        z = self.act(z)
        z = self.drop(z)

        logit_s = self.head_start(z).squeeze(-1)
        logit_p = self.head_peak(z).squeeze(-1)
        return logit_s, logit_p


def _load_processor(model_dir: str, img_size: int, local_files_only: bool) -> AutoImageProcessor:
    try:
        return AutoImageProcessor.from_pretrained(model_dir, size=img_size, local_files_only=local_files_only)
    except TypeError:
        return AutoImageProcessor.from_pretrained(model_dir, local_files_only=local_files_only)


def _resolve_model_cfg_from_ckpt(user_cfg: InferConfig, ckpt_cfg: dict) -> ResolvedModelConfig:
    ckpt_cfg = ckpt_cfg if isinstance(ckpt_cfg, dict) else {}

    model_name = str(ckpt_cfg.get("model", user_cfg.model_dir))
    img_size = int(ckpt_cfg.get("IMG_SIZE", user_cfg.img_size))
    clip_len = int(ckpt_cfg.get("CLIP_LEN", user_cfg.clip_len))

    if user_cfg.use_fasg is None:
        use_fasg = bool(ckpt_cfg.get("USE_FASG", True))
    else:
        use_fasg = bool(user_cfg.use_fasg)

    fasg_short = int(ckpt_cfg.get("FASG_SHORT_WIN", 3))
    fasg_long = int(ckpt_cfg.get("FASG_LONG_WIN", 7))
    fasg_gamma = float(ckpt_cfg.get("FASG_GATE_GAMMA", 1.0))

    return ResolvedModelConfig(
        model_name=model_name,
        img_size=img_size,
        clip_len=clip_len,
        use_fasg=use_fasg,
        fasg_short=fasg_short,
        fasg_long=fasg_long,
        fasg_gamma=fasg_gamma,
    )


def load_model_and_processor(cfg: InferConfig):
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(cfg.ckpt_path, map_location=device)

    ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    resolved = _resolve_model_cfg_from_ckpt(cfg, ckpt_cfg)

    processor = _load_processor(resolved.model_name, resolved.img_size, cfg.local_files_only)

    model = VideoMAE2StartPeak(
        model_name=resolved.model_name,
        freeze_backbone=False,
        use_fasg=resolved.use_fasg,
        fasg_short=resolved.fasg_short,
        fasg_long=resolved.fasg_long,
        fasg_gamma=resolved.fasg_gamma,
        local_files_only=cfg.local_files_only,
    ).to(device)

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("  missing keys:")
            for k in missing:
                print("   -", k)
        if unexpected:
            print("  unexpected keys:")
            for k in unexpected:
                print("   -", k)

    model.eval()
    return model, processor, device, resolved, ckpt_cfg


def load_frames_tensor(seq_dir: Path, processor: AutoImageProcessor) -> Tuple[list[Path], torch.Tensor]:
    frames = list_frames(seq_dir)
    if not frames:
        raise ValueError(f"No frames found in: {seq_dir}")

    images = [Image.open(p).convert("RGB") for p in frames]
    processed = processor(images=images, return_tensors="pt")
    pixel_values = processed["pixel_values"]
    if pixel_values.dim() == 5:
        pixel_values = pixel_values.squeeze(0)
    return frames, pixel_values


def decode_order_constrained(ps: np.ndarray, pp: np.ndarray) -> tuple[int, int]:
    ps = np.asarray(ps, dtype=np.float64)
    pp = np.asarray(pp, dtype=np.float64)

    if ps.size == 0:
        return 0, 0

    ps = np.clip(ps, 1e-12, 1.0)
    pp = np.clip(pp, 1e-12, 1.0)

    n = len(ps)
    best_pp = np.zeros(n, dtype=np.float64)
    best_ix = np.zeros(n, dtype=np.int64)

    maxv = -1.0
    maxi = n - 1
    for i in range(n - 1, -1, -1):
        if pp[i] >= maxv:
            maxv = pp[i]
            maxi = i
        best_pp[i] = maxv
        best_ix[i] = maxi

    score = ps * best_pp
    start_idx = int(score.argmax())
    peak_idx = int(best_ix[start_idx])
    return start_idx, peak_idx


@torch.no_grad()
def infer_sequence(
    model: VideoMAE2StartPeak,
    frames_tensor: torch.Tensor,
    device: str,
    clip_len: int,
    val_stride: int,
    enforce_order: bool = True,
):
    model.eval()
    frames = frames_tensor.to(device)
    n = frames.shape[0]

    acc_s = torch.zeros(n, device=device)
    acc_p = torch.zeros(n, device=device)
    cnt = torch.zeros(n, device=device)

    for st in range(0, max(1, n - clip_len + 1), val_stride):
        ed = min(n, st + clip_len)
        idxs = list(range(st, ed))
        if len(idxs) < clip_len:
            idxs += [idxs[-1]] * (clip_len - len(idxs))

        clip = frames[idxs].unsqueeze(0)
        log_s, log_p = model(clip)

        ps = torch.sigmoid(log_s[0])
        pp = torch.sigmoid(log_p[0])

        for i_local, i_global in enumerate(idxs):
            acc_s[i_global] += ps[i_local]
            acc_p[i_global] += pp[i_local]
            cnt[i_global] += 1.0

    ps_full = (acc_s / (cnt + 1e-8)).detach().cpu().numpy()
    pp_full = (acc_p / (cnt + 1e-8)).detach().cpu().numpy()

    if enforce_order:
        pred_start, pred_peak = decode_order_constrained(ps_full, pp_full)
    else:
        pred_start = int(ps_full.argmax())
        pred_peak = int(pp_full.argmax())

    return pred_start, pred_peak, ps_full, pp_full


def save_probability_csv(out_dir: Path, seq_name: str, ps: np.ndarray, pp: np.ndarray) -> None:
    csv_path = out_dir / f"{seq_name}_probabilities.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "p_start", "p_peak"])
        for i in range(len(ps)):
            writer.writerow([i, float(ps[i]), float(pp[i])])


def run_inference_on_root(input_root: str | Path, cfg: InferConfig) -> Path:
    input_root = Path(input_root)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_seq_dirs = list_sequence_dirs(input_root)

    min_required_frames = None
    if cfg.reference_total_frames is not None and cfg.min_frame_ratio > 0:
        raw_threshold = cfg.reference_total_frames * cfg.min_frame_ratio
        min_required_frames = math.ceil(raw_threshold)
        comparator_text = ">="
        if cfg.strict_ratio:
            min_required_frames = math.floor(raw_threshold) + 1
            comparator_text = ">"

        print(
            f"[INFO] Sequence filter enabled: n_frames {comparator_text} {min_required_frames} "
            f"(reference_total_frames={cfg.reference_total_frames}, ratio={cfg.min_frame_ratio:.2f})"
        )

        kept_seq_dirs = []
        skipped_results = []
        for seq_dir in all_seq_dirs:
            n_frames = len(list_frames(seq_dir))
            keep = n_frames >= min_required_frames
            if cfg.strict_ratio:
                keep = n_frames > (cfg.reference_total_frames * cfg.min_frame_ratio)

            if keep:
                kept_seq_dirs.append(seq_dir)
            else:
                skipped_results.append(
                    {
                        "seq": seq_dir.name,
                        "path": str(seq_dir),
                        "n_frames": n_frames,
                        "skip_reason": f"n_frames_below_{cfg.min_frame_ratio:.2f}",
                    }
                )
                print(f"[SKIP] {seq_dir.name}: {n_frames} frames < required {min_required_frames}")

        seq_dirs = kept_seq_dirs
    else:
        seq_dirs = all_seq_dirs
        skipped_results = []

    if not seq_dirs:
        skipped_json = out_dir / "skipped_sequences.json"
        skipped_json.write_text(json.dumps(skipped_results, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError("No eligible sequences found for inference after frame-ratio filtering.")

    model, processor, device, resolved_cfg, ckpt_cfg = load_model_and_processor(cfg)

    print(
        "[INFO] Resolved infer config:\n"
        f"       model_name = {resolved_cfg.model_name}\n"
        f"       img_size   = {resolved_cfg.img_size}\n"
        f"       clip_len   = {resolved_cfg.clip_len}\n"
        f"       val_stride = {cfg.val_stride}\n"
        f"       use_fasg   = {resolved_cfg.use_fasg}\n"
        f"       fasg_short = {resolved_cfg.fasg_short}\n"
        f"       fasg_long  = {resolved_cfg.fasg_long}\n"
        f"       fasg_gamma = {resolved_cfg.fasg_gamma}"
    )

    results = []
    for seq_dir in tqdm(seq_dirs, desc="Infer sequences", dynamic_ncols=True):
        frames, frames_tensor = load_frames_tensor(seq_dir, processor)
        pred_s, pred_p, ps, pp = infer_sequence(
            model=model,
            frames_tensor=frames_tensor,
            device=device,
            clip_len=resolved_cfg.clip_len,
            val_stride=cfg.val_stride,
            enforce_order=cfg.enforce_order,
        )

        seq_result = {
            "seq": seq_dir.name,
            "path": str(seq_dir),
            "n_frames": len(frames),
            "pred_start_idx": int(pred_s),
            "pred_peak_idx": int(pred_p),
            "pred_start_name": frames[pred_s].name,
            "pred_peak_name": frames[pred_p].name,
            "order_ok": int(pred_s <= pred_p),
        }
        results.append(seq_result)

        if cfg.save_prob_csv:
            save_probability_csv(out_dir, seq_dir.name, ps, pp)

    json_path = out_dir / "per_sequence_predictions.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "per_sequence_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seq",
                "path",
                "n_frames",
                "pred_start_idx",
                "pred_peak_idx",
                "pred_start_name",
                "pred_peak_name",
                "order_ok",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    skipped_json = out_dir / "skipped_sequences.json"
    skipped_json.write_text(json.dumps(skipped_results, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "total_sequences_before_filter": len(all_seq_dirs),
        "total_sequences_after_filter": len(results),
        "skipped_sequences": len(skipped_results),
        "min_required_frames": min_required_frames,
        "reference_total_frames": cfg.reference_total_frames,
        "min_frame_ratio": cfg.min_frame_ratio,
        "order_ok_ratio": float(np.mean([r["order_ok"] for r in results])) if results else 0.0,
        "input_root": str(input_root),
        "output_dir": str(out_dir),
        "resolved_model_cfg": resolved_cfg.__dict__,
        "ckpt_cfg": ckpt_cfg if isinstance(ckpt_cfg, dict) else {},
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[INFO] Inference finished")
    print(f"[INFO] JSON   -> {json_path}")
    print(f"[INFO] CSV    -> {csv_path}")
    print(f"[INFO] Skipped-> {skipped_json}")
    print(f"[INFO] Summary-> {summary_path}")
    return out_dir
