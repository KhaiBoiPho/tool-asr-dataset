"""So khớp thuật ngữ chuyên ngành trên word-level transcript bằng fuzzy matching.

Dùng rapidfuzz thay vì substring match thuần vì Whisper có thể ghi sai chính tả
thuật ngữ tiếng Anh khi bị nói xen trong câu tiếng Việt (vd "gelcoat" -> "gel cot").
"""

from rapidfuzz import fuzz


def flatten_words(segments: list[dict]) -> list[dict]:
    """Gộp toàn bộ word-level entries của mọi segment thành 1 danh sách phẳng, có thứ tự thời gian."""
    words = []
    for seg in segments:
        words.extend(seg["words"])
    return words


def find_term_matches(words: list[dict], terms: list[str], fuzzy_threshold: float = 80.0) -> list[dict]:
    """Trả về list match: {term, matched_text, start, end, confidence}.

    Với mỗi thuật ngữ, trượt cửa sổ có độ dài bằng số từ của thuật ngữ đó qua
    toàn bộ transcript, so khớp fuzzy (0-100) với chính thuật ngữ.
    """
    matches = []
    for term in terms:
        term_word_count = len(term.split())
        term_lower = term.lower()

        for i in range(len(words) - term_word_count + 1):
            window = words[i : i + term_word_count]
            window_text = " ".join(w["word"] for w in window)
            score = fuzz.ratio(window_text.lower(), term_lower)
            if score >= fuzzy_threshold:
                avg_word_prob = sum(w["probability"] for w in window) / len(window)
                matches.append(
                    {
                        "term": term,
                        "matched_text": window_text,
                        "start": window[0]["start"],
                        "end": window[-1]["end"],
                        "confidence": round(avg_word_prob * (score / 100.0), 4),
                    }
                )

    matches.sort(key=lambda m: m["start"])
    return matches
