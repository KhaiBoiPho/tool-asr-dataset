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
    "Bạn là trợ lý kiểm tra transcript tiếng Việt có lẫn thuật ngữ chuyên ngành "
    "composite/FRP, được tạo ra bởi ASR nên có thể có lỗi. Với đoạn transcript được "
    "cung cấp, hãy:\n"
    "1) liệt kê thêm các thuật ngữ chuyên ngành composite/FRP xuất hiện trong đoạn mà "
    "KHÔNG có trong danh sách đã biết (nếu không có thì để mảng rỗng)\n"
    "2) đề xuất bản transcript đã sửa lỗi chính tả/ngữ pháp rõ ràng do ASR gây ra, "
    "KHÔNG suy diễn thêm nội dung không có trong transcript gốc\n"
    "Chỉ trả về JSON hợp lệ, không thêm giải thích, đúng schema:\n"
    '{"extra_terms": ["..."], "suggested_text": "..."}'
)


def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
    return json.loads(raw)


def validate_clip(transcript_vi: str, known_terms: list[str], api_key: str) -> dict:
    """Trả về {"extra_terms": [...], "suggested_text": "..."}.

    Không bao giờ raise ra ngoài — lỗi gọi API trả về dict rỗng để pipeline chính
    không bị gián đoạn vì một lần gọi LLM thất bại.
    """
    if not transcript_vi.strip():
        return {"extra_terms": [], "suggested_text": ""}

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
        return {
            "extra_terms": data.get("extra_terms", []),
            "suggested_text": data.get("suggested_text", ""),
        }
    except Exception:
        logger.exception("Lỗi khi gọi OpenRouter/Haiku để validate clip, bỏ qua.")
        return {"extra_terms": [], "suggested_text": ""}
