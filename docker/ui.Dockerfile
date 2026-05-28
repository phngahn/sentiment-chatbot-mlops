FROM python:3.11-slim

WORKDIR /app

COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt
RUN pip install --no-cache-dir sniffio anyio

COPY src/ ./src/

ENV PYTHONPATH=/app

EXPOSE 8001
CMD ["chainlit", "run", "src/chatbot/demo_chainlit.py", "--host", "0.0.0.0", "--port", "8001", "--headless"]