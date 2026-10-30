#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures as cf
import traceback
import json
import time
import math

# --------- Helpers ---------

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

def find_videos(root: Path, recursive: bool = True) -> List[Path]:
    if recursive:
        return [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    else:
        return [p for p in root.iterdir() if p.suffix.lower() in VIDEO_EXTS and p.is_file()]

def probe_opencv(path: Path) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[str]]:
    """
    Returns: (frame_count, fps, duration_seconds, error_message_or_None)
    Uses OpenCV only (no ffmpeg shell calls).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, None, None, "OpenCV failed to open"
    try:
        # Some containers expose duration directly, but many don't; frame_count/fps is more reliable.
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # sanitize
        frames_i = int(frames) if frames and frames > 0 else None
        fps_f = float(fps) if fps and fps > 0 else None

        duration = None
        if frames_i is not None and fps_f is not None and fps_f > 0:
            duration = frames_i / fps_f
        else:
            # Fallback: try to step a few frames to infer fps (slow; we avoid in bulk)
            pass

        return frames_i, fps_f, duration, None
    except Exception as e:
        return None, None, None, f"Exception: {e}"
    finally:
        cap.release()

def process_one(path: Path) -> Dict:
    frames, fps, duration, err = probe_opencv(path)
    return {
        "relative_path": str(path),
        "filename": path.name,
        "parent": str(path.parent),
        "frames": frames if frames is not None else np.nan,
        "fps": fps if fps is not None else np.nan,
        "duration_sec": duration if duration is not None else np.nan,
        "error": err if err else ""
    }

def summarize_series(s: pd.Series) -> Dict[str, float]:
    s_valid = s.dropna()
    if len(s_valid) == 0:
        return {}
    return {
        "count": int(s_valid.count()),
        "mean": float(s_valid.mean()),
        "std": float(s_valid.std(ddof=1)) if len(s_valid) > 1 else 0.0,
        "min": float(s_valid.min()),
        "p25": float(s_valid.quantile(0.25)),
        "median": float(s_valid.median()),
        "p75": float(s_valid.quantile(0.75)),
        "max": float(s_valid.max()),
    }

def save_histogram(data: pd.Series, title: str, fname: str, bins="auto", xlab="", ylab="Count", clip_max=None, logy=False):
    d = data.dropna()
    if clip_max is not None:
        d = d.clip(upper=clip_max)
    plt.figure()
    plt.hist(d, bins=bins)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    if logy:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# --------- Main ---------

def main():
    parser = argparse.ArgumentParser(description="Summarize a video folder: per-clip frames & duration + histograms.")
    parser.add_argument("--root", type=str, default="/home/xxxxx/Research/Data/HAA500/videos",
                        help="Directory containing videos (default: HAA500/videos).")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders (default: True).")
    parser.set_defaults(recursive=True)
    parser.add_argument("--max_workers", type=int, default=os.cpu_count() or 8, help="Parallel workers.")
    parser.add_argument("--csv", type=str, default="video_stats.csv", help="Output CSV filename.")
    parser.add_argument("--summary", type=str, default="video_summary.txt", help="Human-readable summary file.")
    parser.add_argument("--json", type=str, default="video_summary.json", help="JSON summary file.")
    parser.add_argument("--dur_hist", type=str, default="duration_hist_seconds.png", help="Duration histogram PNG.")
    parser.add_argument("--frame_hist", type=str, default="frames_hist.png", help="Frames histogram PNG.")
    parser.add_argument("--fps_hist", type=str, default="fps_hist.png", help="FPS histogram PNG.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    print(f"[info] Scanning: {root}")
    vids = find_videos(root, recursive=args.recursive)
    print(f"[info] Found {len(vids)} video files.")

    t0 = time.time()
    rows = []
    errors = 0

    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for res in tqdm(ex.map(process_one, vids), total=len(vids), desc="Probing"):
            rows.append(res)
            if res["error"]:
                errors += 1

    df = pd.DataFrame(rows)
    # Sort for stable CSV
    df = df.sort_values(by=["parent", "filename"]).reset_index(drop=True)

    # Save CSV
    out_csv = Path(args.csv).resolve()
    df.to_csv(out_csv, index=False)
    print(f"[ok] Wrote CSV: {out_csv} (rows={len(df)})")

    # Compute basic summaries
    total_videos = len(df)
    valid_dur = df["duration_sec"].dropna()
    valid_frames = df["frames"].dropna()
    valid_fps = df["fps"].dropna()

    dur_stats = summarize_series(df["duration_sec"])
    frame_stats = summarize_series(df["frames"])
    fps_stats = summarize_series(df["fps"])

    total_duration_sec = float(valid_dur.sum()) if len(valid_dur) > 0 else 0.0
    total_duration_hours = total_duration_sec / 3600.0 if total_duration_sec else 0.0

    summary = {
        "root": str(root),
        "total_videos": int(total_videos),
        "read_errors": int(errors),
        "have_duration": int(valid_dur.count()),
        "have_frames": int(valid_frames.count()),
        "have_fps": int(valid_fps.count()),
        "total_duration_sec": total_duration_sec,
        "total_duration_hours": total_duration_hours,
        "duration_stats_sec": dur_stats,
        "frame_stats": frame_stats,
        "fps_stats": fps_stats,
    }

    # Save summary JSON + TXT
    out_json = Path(args.json).resolve()
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] Wrote JSON summary: {out_json}")

    out_txt = Path(args.summary).resolve()
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Video Dataset Summary for: {root}\n")
        f.write(f"Total videos: {total_videos}\n")
        f.write(f"Readable (no errors): {total_videos - errors}, Read errors: {errors}\n")
        f.write(f"Clips with duration available: {summary['have_duration']}\n")
        f.write(f"Clips with frames available:   {summary['have_frames']}\n")
        f.write(f"Clips with fps available:      {summary['have_fps']}\n")
        f.write(f"Total duration: {total_duration_sec:.2f} sec ({total_duration_hours:.2f} hours)\n\n")

        def dump_stats(name, st):
            if not st:
                f.write(f"{name}: <no data>\n")
                return
            f.write(f"{name}:\n")
            f.write(f"  count : {st['count']}\n")
            f.write(f"  mean  : {st['mean']:.3f}\n")
            f.write(f"  std   : {st['std']:.3f}\n")
            f.write(f"  min   : {st['min']:.3f}\n")
            f.write(f"  25%   : {st['p25']:.3f}\n")
            f.write(f"  median: {st['median']:.3f}\n")
            f.write(f"  75%   : {st['p75']:.3f}\n")
            f.write(f"  max   : {st['max']:.3f}\n\n")

        dump_stats("Duration (sec)", dur_stats)
        dump_stats("Frames", frame_stats)
        dump_stats("FPS", fps_stats)

        f.write("Notes:\n")
        f.write("- Duration is computed as frames / fps when both are available from OpenCV.\n")
        f.write("- If fps is unreliable, consider cross-checking a sample with ffprobe.\n")
        f.write("- Use duration distribution to choose NUM_INPUT_FRAMES and SAMPLING_RATE.\n")
    print(f"[ok] Wrote text summary: {out_txt}")

    # Histograms (saved where script is run from)
    if len(valid_dur) > 0:
        # Clip extreme outliers for a readable plot; also provide a log-scale alternative
        clip_at = np.percentile(valid_dur, 99.5)  # avoid one huge outlier flattening the chart
        save_histogram(df["duration_sec"], "Duration (sec)", args.dur_hist,
                       bins="auto", xlab="Seconds", clip_max=clip_at, logy=False)
        save_histogram(df["duration_sec"], "Duration (sec) [log scale]", "duration_hist_seconds_logy.png",
                       bins="auto", xlab="Seconds", clip_max=clip_at, logy=True)
        print(f"[ok] Wrote duration histograms: {args.dur_hist}, duration_hist_seconds_logy.png")

    if len(valid_frames) > 0:
        clip_at = np.percentile(valid_frames, 99.5)
        save_histogram(df["frames"], "Frames per clip", args.frame_hist,
                       bins="auto", xlab="Frames", clip_max=clip_at)
        print(f"[ok] Wrote frames histogram: {args.frame_hist}")

    if len(valid_fps) > 0:
        clip_at = np.percentile(valid_fps, 99.5)
        save_histogram(df["fps"], "FPS", args.fps_hist,
                       bins="auto", xlab="FPS", clip_max=clip_at)
        print(f"[ok] Wrote FPS histogram: {args.fps_hist}")

    dt = time.time() - t0
    print(f"[done] Completed in {dt:.2f}s")

if __name__ == "__main__":
    main()
