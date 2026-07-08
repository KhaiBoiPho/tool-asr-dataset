"""Gọi Claude Haiku 4.5 qua OpenRouter để validate THÊM (không ghi đè) transcript + thuật ngữ.

Dùng cho dataset nghiên cứu: kết quả LLM luôn nằm ở cột riêng (llm_extra_terms,
transcript_vi_llm_suggested) song song với transcript_vi gốc từ Whisper, để đối chiếu
chứ không thay thế nguồn ASR gốc.
"""

import json
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-haiku-4.5"

SYSTEM_PROMPT = (
    "Bạn là trợ lý kiểm tra transcript tiếng Việt có lẫn thuật ngữ chuyên ngành kỹ "
    "thuật/xây dựng/vật liệu bằng tiếng Anh (ví dụ: composite, FRP, HDPE, resin, nano, "
    "graphene, v.v. — KHÔNG giới hạn trong 1 lĩnh vực cố định, bất kỳ thuật ngữ kỹ "
    "thuật/vật liệu tiếng Anh hợp lệ nào cũng được tính), được tạo ra bởi ASR nên có "
    "thể có lỗi. Với đoạn transcript được cung cấp, hãy:\n"
    "1) liệt kê thêm các thuật ngữ chuyên ngành kỹ thuật/xây dựng/vật liệu bằng tiếng "
    "Anh xuất hiện trong đoạn mà KHÔNG có trong danh sách đã biết (nếu không có thì để "
    "mảng rỗng)\n"
    "2) đề xuất bản transcript đã sửa lỗi chính tả/ngữ pháp rõ ràng do ASR gây ra, "
    "KHÔNG suy diễn thêm nội dung không có trong transcript gốc\n"
    "3) đánh giá mức độ tự tin (0-100) DỰA TRÊN ĐÚNG 2 TIÊU CHÍ SAU:\n"
    "   (a) transcript có MẠCH LẠC, ĐÚNG NGỮ PHÁP tiếng Việt hay không (câu văn có "
    "nghĩa, không rời rạc/vô nghĩa do ASR lỗi)\n"
    "   (b) có ít nhất 1 thuật ngữ kỹ thuật/xây dựng/vật liệu tiếng Anh THẬT SỰ tồn "
    "tại (không phải do ASR bịa ra) và được dùng hợp lý trong câu\n"
    "   KHÔNG đánh giá thấp chỉ vì chủ đề video không thuộc riêng composite/FRP — mọi "
    "thuật ngữ kỹ thuật/vật liệu tiếng Anh hợp lệ đều được chấp nhận như nhau, không "
    "cần liên quan đến 1 chủ đề cụ thể.\n"
    "Lưu ý: đây chỉ là đánh giá dựa trên VĂN BẢN, không phải kiểm chứng lại audio gốc.\n"
    "Chỉ trả về JSON hợp lệ, không thêm giải thích, đúng schema:\n"
    '{"extra_terms": ["..."], "suggested_text": "...", "confidence_percent": 0-100}'
)


def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
    return json.loads(raw)


DETECT_SYSTEM_PROMPT = (
    "Bạn nhận được danh sách câu tiếng Việt (có lẫn thuật ngữ kỹ thuật/khoa học/vật liệu "
    "tiếng Anh, ví dụ composite, FRP, HDPE, resin, nano, graphene, carbon fiber, v.v. — "
    "KHÔNG giới hạn trong 1 lĩnh vực cố định, bất kỳ thuật ngữ kỹ thuật/khoa học/vật liệu "
    "tiếng Anh hợp lệ nào cũng được tính), mỗi câu có 1 chỉ số [index] đứng trước. Với MỖI "
    "câu có chứa ít nhất 1 thuật ngữ kỹ thuật tiếng Anh THẬT SỰ tồn tại (không phải do ASR "
    "bịa ra/lỗi chính tả vô nghĩa), trả về index của câu đó + thuật ngữ tìm được (giữ nguyên "
    "chính tả xuất hiện trong câu) + độ tự tin (0-100) rằng đây đúng là thuật ngữ kỹ thuật "
    "hợp lệ. Bỏ qua câu không có thuật ngữ nào. Nếu 1 câu có nhiều thuật ngữ, trả nhiều dòng "
    "cho cùng 1 index. Chỉ trả về JSON hợp lệ, không thêm giải thích, đúng schema:\n"
    '{"matches": [{"index": 0, "term": "...", "confidence": 0-100}]}'
)


def detect_terms_in_segments(segments: list[dict], api_key: str, batch_size: int = 25) -> list[dict]:
    """Quét toàn bộ transcript bằng Claude Haiku để tự phát hiện thuật ngữ kỹ thuật/khoa học/
    vật liệu tiếng Anh xuất hiện trong câu tiếng Việt, KHÔNG giới hạn theo danh sách terms.yaml
    cố định — thay thế cho term_matcher.find_term_matches khi có OpenRouter API key.

    Trả về matches cùng schema với term_matcher.find_term_matches: {term, matched_text,
    start, end, confidence} — start/end lấy theo ranh giới câu (segment) chứa thuật ngữ, đủ
    để segment_builder căn clip theo câu như bình thường (không cần định vị word-level).

    Không bao giờ raise ra ngoài — batch nào gọi API lỗi thì bỏ qua batch đó (log warning),
    không làm gián đoạn cả video.
    """
    if not segments:
        return []

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    indexed_segments = list(enumerate(segments))
    matches = []

    for batch_start in range(0, len(indexed_segments), batch_size):
        batch = indexed_segments[batch_start : batch_start + batch_size]
        batch_text = "\n".join(f'[{i}] {seg["text"]}' for i, seg in batch)
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": DETECT_SYSTEM_PROMPT},
                    {"role": "user", "content": batch_text},
                ],
            )
            data = _parse_json_response(resp.choices[0].message.content)
            batch_end = batch_start + len(batch)
            for m in data.get("matches", []):
                idx = m.get("index")
                term = m.get("term", "").strip()
                if not isinstance(idx, int) or not (batch_start <= idx < batch_end) or not term:
                    continue
                seg = segments[idx]
                try:
                    confidence = max(0.0, min(100.0, float(m.get("confidence", 0)))) / 100.0
                except (TypeError, ValueError):
                    confidence = 0.0
                matches.append(
                    {
                        "term": term,
                        "matched_text": term,
                        "start": seg["start"],
                        "end": seg["end"],
                        "confidence": round(confidence, 4),
                    }
                )
        except Exception:
            logger.exception(
                "Lỗi khi gọi OpenRouter/Haiku để phát hiện thuật ngữ (segment %d-%d), bỏ qua batch này.",
                batch_start,
                batch_start + len(batch) - 1,
            )

    matches.sort(key=lambda m: m["start"])
    return matches


def validate_clip(transcript_vi: str, known_terms: list[str], api_key: str) -> dict:
    """Trả về {"extra_terms": [...], "suggested_text": "...", "confidence_percent": 0-100 | None}.

    confidence_percent là đánh giá CỦA LLM dựa trên văn bản (transcript có mạch lạc,
    thuật ngữ có hợp ngữ cảnh không) — không phải kiểm chứng lại audio gốc, nên không
    thay thế được việc xác minh audio thực tế.

    Không bao giờ raise ra ngoài — lỗi gọi API trả về dict rỗng (confidence_percent=None)
    để pipeline chính không bị gián đoạn vì một lần gọi LLM thất bại.
    """
    if not transcript_vi.strip():
        return {"extra_terms": [], "suggested_text": "", "confidence_percent": None}

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    user_content = (
        f"Danh sách thuật ngữ đã biết: {', '.join(known_terms)}\n\n"
        f"Transcript cần kiểm tra:\n{transcript_vi}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        data = _parse_json_response(resp.choices[0].message.content)
        confidence = data.get("confidence_percent")
        try:
            confidence = max(0.0, min(100.0, float(confidence))) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        return {
            "extra_terms": data.get("extra_terms", []),
            "suggested_text": data.get("suggested_text", ""),
            "confidence_percent": confidence,
        }
    except Exception:
        logger.exception("Lỗi khi gọi OpenRouter/Haiku để validate clip, bỏ qua.")
        return {"extra_terms": [], "suggested_text": "", "confidence_percent": None}
