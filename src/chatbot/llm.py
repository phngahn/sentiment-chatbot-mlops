"""
LLM caller — Groq (Llama 3.3 70B)
"""
from __future__ import annotations
import os
from groq import Groq

GROQ_MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """Bạn là trợ lý mua sắm thông minh của Tiki. 
Dựa vào thông tin sản phẩm được cung cấp, hãy trả lời câu hỏi của khách hàng một cách ngắn gọn, chính xác.
Ưu tiên dùng số liệu từ đánh giá (ABSA scores) để đưa ra nhận xét cụ thể.
Nếu không có đủ thông tin, hãy nói thẳng là không biết, đừng bịa."""

def build_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.get("metadata", {})
        text = doc.get("text", "")
        parts.append(f"[{i}] {text}" if text else f"[{i}] {m.get('name', '')} — (metadata only)")
    return "\n\n".join(parts)

def ask(query: str, docs: list[dict]) -> str:
    context = build_context(docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Thông tin sản phẩm:\n{context}\n\nCâu hỏi: {query}"},
    ]
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
    )
    return resp.choices[0].message.content