"""Gom các match thuật ngữ liên tiếp thành clip candidate.

Ưu tiên căn clip theo đúng ranh giới câu hoàn chỉnh (segment-level của Whisper) thay vì
chỉ pad cứng quanh từ khớp — để clip nghe tự nhiên, đủ ngữ cảnh, và không bị cắt cụt
giữa câu. Mở rộng sang câu liền kề nếu ngắn hơn min_duration_sec; không bao giờ vượt
quá max_duration_sec.
"""


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


def _split_oversized_matches(group: list[dict], max_duration_sec: float) -> list[list[dict]]:
    """Chia đôi đệ quy nếu khoảng match thô (chưa mở rộng theo câu) đã vượt quá độ dài
    tối đa — an toàn cho trường hợp hiếm khi nhiều match nằm rải rác trên diện rộng."""
    span = group[-1]["end"] - group[0]["start"]
    if span <= max_duration_sec or len(group) == 1:
        return [group]
    mid = len(group) // 2
    return _split_oversized_matches(group[:mid], max_duration_sec) + _split_oversized_matches(
        group[mid:], max_duration_sec
    )


def _overlapping_segment_range(segments_sorted: list[dict], start: float, end: float):
    """Trả về (start_idx, end_idx) của dải segment chồng lấn [start, end], hoặc None
    nếu không có segment nào bao trùm (hiếm khi xảy ra)."""
    idx = [i for i, s in enumerate(segments_sorted) if s["end"] > start and s["start"] < end]
    return (idx[0], idx[-1]) if idx else None


def _expand_to_min_duration(
    segments_sorted: list[dict], start_idx: int, end_idx: int, min_duration_sec: float, max_duration_sec: float
):
    """Mở rộng sang segment liền kề (ưu tiên bên phải trước) cho tới khi đạt
    min_duration_sec, không vượt quá max_duration_sec, hoặc hết segment lân cận."""
    n = len(segments_sorted)
    while segments_sorted[end_idx]["end"] - segments_sorted[start_idx]["start"] < min_duration_sec:
        extended = False
        if end_idx < n - 1:
            candidate_end = segments_sorted[end_idx + 1]["end"]
            if candidate_end - segments_sorted[start_idx]["start"] <= max_duration_sec:
                end_idx += 1
                extended = True
        if not extended and start_idx > 0:
            candidate_start = segments_sorted[start_idx - 1]["start"]
            if segments_sorted[end_idx]["end"] - candidate_start <= max_duration_sec:
                start_idx -= 1
                extended = True
        if not extended:
            break
    return start_idx, end_idx


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


def _merge_overlapping_windows(windows: list[dict]) -> list[dict]:
    """Gộp các cửa sổ [start, end] chồng lấn/trùng nhau thành 1 — xảy ra khi 2 match
    riêng biệt (cách nhau hơn merge_gap_sec) vẫn cùng được giãn ra trùng 1 ranh giới câu."""
    if not windows:
        return []
    windows_sorted = sorted(windows, key=lambda w: w["start"])
    merged = [dict(windows_sorted[0])]
    for w in windows_sorted[1:]:
        last = merged[-1]
        if w["start"] < last["end"]:
            last["end"] = max(last["end"], w["end"])
            last["terms"] = sorted(set(last["terms"]) | set(w["terms"]))
            last["confidence"] = round((last["confidence"] + w["confidence"]) / 2, 4)
        else:
            merged.append(dict(w))
    return merged


def build_clips(segments: list[dict], matches: list[dict], clip_cfg: dict) -> list[dict]:
    """Trả về list clip: {start, end, terms, confidence, subtitle_entries, transcript_vi}.

    Với mỗi nhóm match, căn clip theo đúng ranh giới câu (segment Whisper) chứa từ khớp,
    mở rộng sang câu liền kề nếu quá ngắn (< min_duration_sec), luôn <= max_duration_sec.
    Nếu không tìm được segment bao trùm, hoặc câu chứa match tự nó đã dài hơn
    max_duration_sec, fallback về cách pad quanh từ khớp như cũ. Các cửa sổ chồng lấn
    (vd 2 match trong cùng 1 câu bị giãn ra trùng nhau) được gộp lại để tránh clip trùng lặp.
    """
    if not matches:
        return []

    max_duration_sec = clip_cfg["max_duration_sec"]
    min_duration_sec = clip_cfg.get("min_duration_sec", 5.0)
    pad_sec = clip_cfg["pad_sec"]
    merge_gap_sec = clip_cfg["merge_gap_sec"]

    segments_sorted = sorted(segments, key=lambda s: s["start"])

    windows = []
    for group in _merge_matches(matches, merge_gap_sec):
        for sub_group in _split_oversized_matches(group, max_duration_sec):
            match_start = sub_group[0]["start"]
            match_end = sub_group[-1]["end"]

            overlap = _overlapping_segment_range(segments_sorted, match_start, match_end)
            if overlap is not None:
                start_idx, end_idx = overlap
                span = segments_sorted[end_idx]["end"] - segments_sorted[start_idx]["start"]
                if span <= max_duration_sec:
                    start_idx, end_idx = _expand_to_min_duration(
                        segments_sorted, start_idx, end_idx, min_duration_sec, max_duration_sec
                    )
                    clip_start = segments_sorted[start_idx]["start"]
                    clip_end = segments_sorted[end_idx]["end"]
                else:
                    # Câu chứa match tự nó đã dài hơn max_duration_sec -> fallback pad quanh match
                    clip_start = match_start - pad_sec
                    clip_end = match_end + pad_sec
            else:
                clip_start = match_start - pad_sec
                clip_end = match_end + pad_sec

            windows.append(
                {
                    "start": max(0.0, clip_start),
                    "end": clip_end,
                    "terms": sorted({m["term"] for m in sub_group}),
                    "confidence": round(
                        sum(m["confidence"] for m in sub_group) / len(sub_group), 4
                    ),
                }
            )

    clips = []
    for w in _merge_overlapping_windows(windows):
        subtitle_entries = _collect_subtitle_entries(segments, w["start"], w["end"])
        transcript_vi = " ".join(e["text"] for e in subtitle_entries)
        clips.append(
            {
                "start": w["start"],
                "end": w["end"],
                "terms": w["terms"],
                "confidence": w["confidence"],
                "subtitle_entries": subtitle_entries,
                "transcript_vi": transcript_vi,
            }
        )

    return clips
