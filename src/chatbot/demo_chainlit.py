"""
Chainlit UI — Tiki Shopping Assistant
Supports: normal chat (streaming) + URL analysis (3-tier) + multi-turn memory
"""
import re
from chainlit.input_widget import Select
import chainlit as cl
from dotenv import load_dotenv
load_dotenv()

from src.chatbot.retrieval import TikiRAG
from src.chatbot.llm import ask_stream
from src.chatbot.url_analyzer import analyze_url

rag = TikiRAG()

TIKI_URL_PATTERN = re.compile(r'(https?://(?:www\.)?tiki\.vn/\S+)')
MAX_HISTORY = 6


@cl.set_starters
async def starters(user: cl.User | None = None, message: str | None = None):
    return [
        cl.Starter(label="🥤 Cốc giữ nhiệt tốt", message="Cốc giữ nhiệt chất lượng tốt, giữ nhiệt lâu"),
        cl.Starter(label="🏠 Đồ gia dụng dưới 300k", message="Gợi ý đồ gia dụng phổ biến có giá dưới 300.000 VND"),
        cl.Starter(label="🚚 Giao hàng nhanh", message="Sản phẩm nào được khách đánh giá giao hàng nhanh"),
        cl.Starter(label="🔗 Phân tích URL", message="Phân tích sản phẩm: https://tiki.vn/binh-giu-nhiet-lock-lock-energetic-one-touch-tumbler-lhc3249-550ml-p83412126.html"),
    ]


@cl.on_chat_start
async def start():
    await cl.ChatSettings([
    Select(
        id="absa_model",
        label="🤖 ABSA Model (khi phân tích URL mới)",
        values=["logreg", "phobert_onnx", "phobert"],
        initial_value="logreg",
        description="⚡ LogReg: nhanh ~2s | 🚀 PhoBERT ONNX: nhanh ~5s | 🎯 PhoBERT: chính xác ~44s",
    ),
]).send()

    cl.user_session.set("absa_model", "logreg")
    cl.user_session.set("history", [])
    cl.user_session.set("last_docs", [])

    await cl.Message(
        content="👋 Xin chào! Mình là **Tiki Shopping Assistant**.\n\n"
        "Bạn có thể:\n"
        "- 💬 Hỏi về sản phẩm trên Tiki\n"
        "- 🔗 Paste link Tiki để phân tích chi tiết\n\n"
        "⚙️ Click icon cài đặt góc trên để chọn ABSA model!",
    ).send()


@cl.on_settings_update
async def update_settings(settings):
    model = settings.get("absa_model", "logreg")
    cl.user_session.set("absa_model", model)
    label = "LogReg (nhanh)" if model == "logreg" else "PhoBERT (chính xác)"
    await cl.Message(content=f"Đã chuyển sang **{label}**").send()


@cl.on_message
async def main(message: cl.Message):
    query = message.content
    url_match = TIKI_URL_PATTERN.search(query)

    if url_match:
        await handle_url_analysis(url_match.group(1), query=query)
    else:
        await handle_chat(query)


async def handle_url_analysis(url: str, query: str = ""):
    msg = cl.Message(content="")
    await msg.send()

    absa_model = cl.user_session.get("absa_model", "logreg")
    progress_lines = []

    async def progress_callback(text: str):
        progress_lines.append(text)
        msg.content = "\n".join(progress_lines)
        await msg.update()

    report = await analyze_url(url, progress_callback=progress_callback, absa_model=absa_model, user_query=query)
    msg.content = "\n".join(progress_lines) + "\n\n---\n\n" + report
    await msg.update()


async def handle_chat(query: str):
    msg = cl.Message(content="")
    await msg.send()

    try:
        history = cl.user_session.get("history", [])

        # Rewrite query nếu có history
        from src.chatbot.llm import rewrite_query
        search_query = rewrite_query(query, history)

        async with cl.Step(name="🔍 Tìm kiếm", type="retrieval") as step:
            docs = rag.search(search_query, top_k=5)
            if search_query != query:
                step.output = f"Tìm: \"{search_query}\" → {len(docs)} sản phẩm"
            else:
                step.output = f"{len(docs)} sản phẩm"

        full_answer = ""
        for chunk in ask_stream(query, docs, history=history): # type: ignore
            full_answer += chunk
            msg.content = full_answer
            await msg.update()

        history.append({"role": "user", "content": query}) # type: ignore
        history.append({"role": "assistant", "content": full_answer}) # type: ignore
        if len(history) > MAX_HISTORY: # type: ignore
            history = history[-MAX_HISTORY:] # type: ignore
        cl.user_session.set("history", history)

        sources = [
            {"doc_type": d["doc_type"], "name": d["metadata"].get("name", ""), "score": round(d["score"], 3)}
            for d in docs
        ]
        if sources:
            source_text = "\n".join([
                f"**{i+1}.** {s['name']} · `{s['doc_type']}` · ⭐ {s['score']:.3f}"
                for i, s in enumerate(sources)
            ])
            await cl.Message(content=f"### 📚 Nguồn tham khảo\n\n{source_text}").send()

    except Exception as e:
        msg.content = f"❌ Lỗi: {e}"
        await msg.update()
        