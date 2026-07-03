"""Ghi metadata.json + metadata.csv cho từng video, và gộp vào all_metadata.csv toàn batch."""

import csv
import json
from pathlib import Path

CSV_FIELDS = [
    "video_id", "video_title", "source_url",
    "clip_file", "srt_file",
    "start_sec", "end_sec", "duration_sec",
    "matched_terms", "confidence", "transcript_vi",
    "llm_extra_terms", "transcript_vi_llm_suggested",
    "llm_confidence_rate", "llm_valid",
]


def build_rows(video_meta: dict, exported_clips: list[dict], output_root: Path) -> list[dict]:
    """exported_clips: mỗi phần tử gồm dữ liệu clip (start/end/terms/confidence/transcript_vi)
    gộp với clip_file/srt_file (Path tuyệt đối) từ clip_exporter.export_clip.

    llm_extra_terms/transcript_vi_llm_suggested để trống mặc định — chỉ được điền thêm
    (không ghi đè transcript_vi gốc) nếu pipeline được chạy kèm bước validate LLM."""
    rows = []
    for c in exported_clips:
        rows.append(
            {
                "video_id": video_meta["video_id"],
                "video_title": video_meta["title"],
                "source_url": video_meta["source_url"],
                "clip_file": str(c["clip_file"].relative_to(output_root)),
                "srt_file": str(c["srt_file"].relative_to(output_root)),
                "start_sec": round(c["start"], 2),
                "end_sec": round(c["end"], 2),
                "duration_sec": round(c["end"] - c["start"], 2),
                "matched_terms": ";".join(c["terms"]),
                "confidence": c["confidence"],
                "transcript_vi": c["transcript_vi"],
                "llm_extra_terms": "",
                "transcript_vi_llm_suggested": "",
                "llm_confidence_rate": "",
                "llm_valid": "",
            }
        )
    return rows


def write_video_metadata(rows: list[dict], video_dir: Path) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)

    (video_dir / "metadata.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with open(video_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_aggregate_csv(rows: list[dict], output_root: Path) -> None:
    agg_path = output_root / "all_metadata.csv"
    write_header = not agg_path.exists()

    with open(agg_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
