from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class AggregateConfig:
    pred_json_path: str
    output_dir: str
    time_format: str = "%Y-%m-%d %H:%M:%S"
    tstart_equal_distance_fallback: str = "latest"
    tpeak_tie_break: str = "earliest"


def extract_ts_from_name(name: str) -> Optional[datetime]:
    m = re.search(r"\d{14}", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(0), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def fmt_dt(dt: Optional[datetime], fmt: str) -> Optional[str]:
    if dt is None:
        return None
    return dt.strftime(fmt)


def time_str_to_ts(time_str: str, fmt: str) -> float:
    return datetime.strptime(time_str, fmt).timestamp()


def safe_mode(counter: Counter, prefer: str = "earliest"):
    if not counter:
        return None, [], 0

    max_count = max(counter.values())
    candidates = [k for k, v in counter.items() if v == max_count]
    candidates = sorted(candidates)

    if prefer == "latest":
        primary = candidates[-1]
    else:
        primary = candidates[0]

    return primary, candidates, max_count


def safe_mode_tstart_by_average(
    counter: Counter,
    all_start_times: list[str],
    fmt: str,
    equal_distance_fallback: str = "latest",
):
    if not counter:
        return None, [], 0, None

    max_count = max(counter.values())
    candidates = [k for k, v in counter.items() if v == max_count]
    candidates = sorted(candidates)

    all_ts = [time_str_to_ts(x, fmt) for x in all_start_times if x is not None]
    avg_ts = sum(all_ts) / len(all_ts) if len(all_ts) > 0 else None
    avg_time_str = datetime.fromtimestamp(avg_ts).strftime(fmt) if avg_ts is not None else None

    if len(candidates) == 1:
        return candidates[0], candidates, max_count, avg_time_str

    cand_ts = {c: time_str_to_ts(c, fmt) for c in candidates}
    dists = {c: abs(cand_ts[c] - avg_ts) for c in candidates}
    min_dist = min(dists.values())

    best = [c for c in candidates if abs(dists[c] - min_dist) < 1e-9]
    best = sorted(best)

    if len(best) == 1:
        primary = best[0]
    else:
        primary = best[-1] if equal_distance_fallback == "latest" else best[0]

    return primary, candidates, max_count, avg_time_str


def run_mode_aggregation(cfg: AggregateConfig) -> Path:
    pred_json_path = Path(cfg.pred_json_path)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_json_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_json_path}")

    records = json.loads(pred_json_path.read_text(encoding="utf-8"))
    if not records:
        raise RuntimeError("Prediction results are empty.")

    start_times = []
    peak_times = []
    enriched_records = []

    for r in records:
        start_str = r.get("pred_start_time")
        peak_str = r.get("pred_peak_time")

        if start_str is None:
            start_name = r.get("pred_start_name")
            start_dt = extract_ts_from_name(start_name) if start_name else None
            start_str = fmt_dt(start_dt, cfg.time_format)

        if peak_str is None:
            peak_name = r.get("pred_peak_name")
            peak_dt = extract_ts_from_name(peak_name) if peak_name else None
            peak_str = fmt_dt(peak_dt, cfg.time_format)

        if start_str is not None:
            start_times.append(start_str)
        if peak_str is not None:
            peak_times.append(peak_str)

        enriched = dict(r)
        enriched["pred_start_time"] = start_str
        enriched["pred_peak_time"] = peak_str
        enriched_records.append(enriched)

    start_counter = Counter(start_times)
    peak_counter = Counter(peak_times)

    tstart_mode, tstart_candidates, tstart_count, tstart_mean = safe_mode_tstart_by_average(
        start_counter,
        all_start_times=start_times,
        fmt=cfg.time_format,
        equal_distance_fallback=cfg.tstart_equal_distance_fallback,
    )

    tpeak_mode, tpeak_candidates, tpeak_count = safe_mode(
        peak_counter,
        prefer=cfg.tpeak_tie_break,
    )

    summary = {
        "total_panicles": len(records),
        "valid_start_times": len(start_times),
        "valid_peak_times": len(peak_times),
        "Tstart_mode": tstart_mode,
        "Tstart_mode_count": tstart_count,
        "Tstart_mode_candidates": tstart_candidates,
        "Tstart_mean_of_all_start_times": tstart_mean,
        "Tstart_tie_break_rule": "closest_to_mean_of_all_start_times",
        "Tstart_equal_distance_fallback": cfg.tstart_equal_distance_fallback,
        "Tpeak_mode": tpeak_mode,
        "Tpeak_mode_count": tpeak_count,
        "Tpeak_mode_candidates": tpeak_candidates,
        "Tpeak_tie_break": cfg.tpeak_tie_break,
    }

    enriched_json = out_dir / "per_sequence_predictions_with_time.json"
    enriched_json.write_text(
        json.dumps(enriched_records, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    enriched_csv = out_dir / "per_sequence_predictions_with_time.csv"
    fieldnames = list(enriched_records[0].keys())
    with enriched_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_records)

    summary_json = out_dir / "mode_summary.json"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    summary_csv = out_dir / "mode_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            if isinstance(v, list):
                writer.writerow([k, "; ".join(v)])
            else:
                writer.writerow([k, v])

    print("\n[INFO] Mode aggregation finished")
    print(f"[INFO] Enriched JSON -> {enriched_json}")
    print(f"[INFO] Enriched CSV  -> {enriched_csv}")
    print(f"[INFO] Summary JSON  -> {summary_json}")
    print(f"[INFO] Summary CSV   -> {summary_csv}")
    print(f"[RESULT] Tstart (mode) = {tstart_mode}")
    print(f"[RESULT] Tpeak  (mode) = {tpeak_mode}")
    print(f"[INFO] Mean of all Tstart = {tstart_mean}")

    return summary_json
