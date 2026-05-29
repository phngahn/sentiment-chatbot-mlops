"""
LLM caller — Multi-provider (Groq / OpenRouter)
Streaming + Optimized prompting for ABSA-aware RAG
"""
from __future__ import annotations
import os
from groq import Groq

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_FAST_MODEL = "llama-3.1-8b-instant"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

ASPECTS_VI = {
    "description": "Mô tả SP",
    "quality":     "Chất lượng",
    "packaging":   "Đóng gói",
    "delivery":    "Giao hàng",
    "service":     "Dịch vụ",
    "price":       "Giá cả",
}

SYSTEM_PROMPT = """Bạn là Tiki Shopping Assistant — trợ lý mua sắm thông minh.

## Nguyên tắc
- Trả lời tự nhiên như đang trò chuyện, KHÔNG liệt kê khô khan
- Dựa 100% vào data được cung cấp, KHÔNG bịa
- Nếu không có data → nói thẳng "mình chưa có thông tin về sản phẩm này"

## Cách diễn đạt số liệu ABSA
KHÔNG BAO GIỜ viết raw numbers như "score 0.87" hay "94 pos / 47 neg".
Hãy diễn đạt tự nhiên:

- Nhiều khen (>70%): "được đa số khách hàng đánh giá tích cực"
- Trái chiều (40-70%): "nhận được đánh giá trái chiều"
- Nhiều chê (<40%): "nhiều khách hàng phàn nàn"
- Ít đánh giá (< 10): "chưa có nhiều đánh giá để kết luận"

Khi cần cụ thể, dùng phần trăm tự nhiên:
✅ "Khoảng 70% khách hài lòng về giao hàng"
✅ "Chất lượng là điểm yếu — chỉ 1/3 khách hài lòng"
❌ "absa_quality_score: 0.333, pos: 94, neg: 47"

## Cấu trúc trả lời
- Mở đầu: trả lời thẳng vào câu hỏi (1-2 câu)
- Thân: phân tích với dẫn chứng tự nhiên
- Kết: khuyến nghị ngắn gọn
- Tối đa 250 từ
- Highlight cả ưu VÀ nhược điểm

## Khi gợi ý sản phẩm
- Nêu tên + giá + lý do cụ thể
- Gợi ý ai nên/không nên mua

## Khi so sánh
- Tóm tắt khác biệt chính
- Kết luận: "nếu cần X thì chọn A, nếu cần Y thì chọn B"

## Ngôn ngữ
- Tiếng Việt tự nhiên, thân thiện
- Dùng "mình/bạn"
- Emoji nhẹ nhàng, không quá nhiều"""


def build_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.get("metadata", {})
        text = doc.get("text", "")
        doc_type = doc.get("doc_type", "")

        header = f"[Nguồn {i}] ({doc_type})"
        if m.get("name"):
            header += f" {m['name']}"
            if m.get("price"):
                header += f" — {m['price']:,.0f}đ"
            if m.get("rating_average"):
                header += f" — {m['rating_average']}/5 ({m.get('review_count', '?')} đánh giá)"

        absa_lines = []
        for asp_en, asp_vi in ASPECTS_VI.items():
            score = m.get(f"absa_{asp_en}_score")
            pos = m.get(f"absa_{asp_en}_pos", 0)
            neg = m.get(f"absa_{asp_en}_neg", 0)
            if score is not None:
                total = pos + neg
                pct = round(pos / total * 100) if total > 0 else 0
                absa_lines.append(f"  {asp_vi}: {pct}% hài lòng ({pos} khen / {neg} chê)")

        absa_block = ""
        if absa_lines:
            absa_block = "\nPhân tích đánh giá:\n" + "\n".join(absa_lines)

        content = text if text else "(không có nội dung text)"
        parts.append(f"{header}\n{content}{absa_block}")

    return "\n\n" + "=" * 50 + "\n\n".join(parts)


def needs_new_search(query: str, history: list[dict]) -> bool:
    """Dùng LLM nhỏ để detect có cần search Qdrant mới không."""
    if not history:
        return True

    messages = [
        {"role": "system", "content": "Trả lời chỉ YES hoặc NO. Câu hỏi người dùng có cần tìm kiếm sản phẩm MỚI không, hay chỉ hỏi thêm về kết quả vừa trả lời trước đó?"},
    ]
    messages.extend(history[-4:])
    messages.append({"role": "user", "content": query})

    try:
        resp = client.chat.completions.create(
            model=GROQ_FAST_MODEL,
            messages=messages,
            max_tokens=5,
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip().upper()
        return "YES" in answer
    except Exception:
        return True  # fallback: search mới nếu lỗi


def ask(query: str, docs: list[dict], history: list[dict] = None) -> str:
    context = build_context(docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Thông tin sản phẩm:\n\n{context}\n\nCâu hỏi: {query}"})
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
    )
    return resp.choices[0].message.content # type: ignore


def ask_stream(query: str, docs: list[dict], history: list[dict] = None):
    context = build_context(docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Thông tin sản phẩm:\n\n{context}\n\nCâu hỏi: {query}"})
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
        stream=True,
    ) # type: ignore
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def ask_url_recommendation(query: str, report: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Báo cáo phân tích sản phẩm:\n\n{report}\n\nCâu hỏi của người dùng: {query}"},
    ]
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.3,
    )
    return resp.choices[0].message.content # type: ignore

def rewrite_query(query: str, history: list[dict]) -> str:
    """Viết lại query dựa vào history để search Qdrant chính xác hơn."""
    if not history:
        return query

    messages = [
        {"role": "system", "content": "Viết lại câu hỏi của người dùng thành một câu tìm kiếm sản phẩm đầy đủ ngữ cảnh, dựa vào lịch sử hội thoại. Chỉ trả về câu tìm kiếm mới, không giải thích gì thêm. Nếu câu hỏi đã đủ rõ ràng thì giữ nguyên."},
    ]
    messages.extend(history[-4:])
    messages.append({"role": "user", "content": query})

    try:
        resp = client.chat.completions.create(
            model=GROQ_FAST_MODEL,
            messages=messages, # type: ignore
            max_tokens=100,
            temperature=0,
        )
        rewritten = resp.choices[0].message.content.strip() # type: ignore
        return rewritten if rewritten else query
    except Exception:
        return query