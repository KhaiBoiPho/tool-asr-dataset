"""Gom các match thuật ngữ liên tiếp thành clip candidate, pad thời gian, giới hạn độ dài."""


def _merge_matches(matches: list[dict], merge_gap_sec: float) -> list[list[dict]]:
    """Gộp các match cách nhau dưới merge_gap_sec thành cùng 1 nhóm (sẽ thành 1 clip)."""
    groups: list[list[dict]] = []
    current: list[dict] = []
    for m in matches:
        if not current:
            current = [m]
            continue
        gap = m["start"] - current[-1]["end"]
        if gap < merge_gap_sec:
            current.append(m)
        else:
            groups.append(current)
            current = [m]
    if current:
        groups.append(current)
    return groups


def _split_oversized(group: list[dict], max_duration_sec: float, pad_sec: float) -> list[list[dict]]:
    """Chia đôi đệ quy nếu nhóm (kèm padding) vượt quá độ dài tối đa cho phép."""
    start = group[0]["start"] - pad_sec
    end = group[-1]["end"] + pad_sec
    if end - start <= max_duration_sec or len(group) == 1:
        return [group]
    mid = len(group) // 2
    return _split_oversized(group[:mid], max_duration_sec, pad_sec) + _split_oversized(
        group[mid:], max_duration_sec, pad_sec
    )


def _collect_subtitle_entries(segments: list[dict], clip_start: float, clip_end: float) -> list[dict]:
    """Lấy các câu (segment-level) chồng lấn với khoảng [clip_start, clip_end], quy đổi
    timestamp về mốc tương đối 0 = clip_start (chuẩn để ghi SRT)."""
    entries = []
    for seg in segments:
        if seg["end"] <= clip_start or seg["start"] >= clip_end:
            continue
        rel_start = max(seg["start"], clip_start) - clip_start
        rel_end = min(seg["end"], clip_end) - clip_start
        entries.append({"start": rel_start, "end": rel_end, "text": seg["text"]})
    return entries


def build_clips(segments: list[dict], matches: list[dict], clip_cfg: dict) -> list[dict]:
    """Trả về list clip: {start, end, terms, confidence, subtitle_entries, transcript_vi}.

    `start`/`end` đã được pad và giới hạn <= max_duration_sec (giây, mốc tuyệt đối trong audio gốc).
    """
    if not matches:
        return []

    max_duration_sec = clip_cfg["max_duration_sec"]
    pad_sec = clip_cfg["pad_sec"]
    merge_gap_sec = clip_cfg["merge_gap_sec"]

    clips = []
    for group in _merge_matches(matches, merge_gap_sec):
        for sub_group in _split_oversized(group, max_duration_sec, pad_sec):
            clip_start = max(0.0, sub_group[0]["start"] - pad_sec)
            clip_end = sub_group[-1]["end"] + pad_sec

            subtitle_entries = _collect_subtitle_entries(segments, clip_start, clip_end)
            transcript_vi = " ".join(e["text"] for e in subtitle_entries)

            clips.append(
                {
                    "start": clip_start,
                    "end": clip_end,
                    "terms": sorted({m["term"] for m in sub_group}),
                    "confidence": round(
                        sum(m["confidence"] for m in sub_group) / len(sub_group), 4
                    ),
                    "subtitle_entries": subtitle_entries,
                    "transcript_vi": transcript_vi,
                }
            )

    return clips
