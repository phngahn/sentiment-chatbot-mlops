# 🛒 Tiki Sentiment Chatbot — CS317 MLOps

> Hệ thống hỏi đáp sản phẩm kết hợp phân tích cảm xúc người mua trên nền tảng thương mại điện tử Tiki

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688)](https://fastapi.tiangolo.com/)
[![Airflow](https://img.shields.io/badge/Airflow-2.7-017CEE)](https://airflow.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-EC2-FF9900)](https://aws.amazon.com/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

**Môn học:** CS317 - Phát triển và vận hành hệ thống máy học  
**GVHD:** TS. Đỗ Văn Tiến | **GVTH:** CN. Lê Trần Trọng Khiêm  
**Trường:** Đại học Công nghệ Thông tin - ĐHQG TP.HCM

---

## 👥 Thành viên nhóm 7

| MSSV | Họ và tên | Phụ trách |
|------|-----------|-----------|
| 23521330 | Trần Phi Quyên | Xây dựng module crawl dữ liệu từ Tiki API gồm 3 module (product list, product detail, reviews) phục vụ cả luồng bulk ban đầu và incremental hàng ngày (Airflow DAG 1). Thiết lập CI/CD pipeline tự động hóa quy trình deploy qua GitHub Actions. Viết báo cáo và làm slide. |
| 23520439 | Văn Thị Bảo Hân | Xây dựng pipeline tiền xử lý và gán nhãn dữ liệu ABSA dựa trên LLM qua kiến trúc AWS Lambda + SQS, quản lý dữ liệu trên AWS S3, đồng bộ chỉ số lên W&B để giám sát và lưu vết; xây dựng module tiền xử lý phục vụ DAG 2. Viết báo cáo và làm slide. |
| 23520078 | Trần Nhật Phương Anh | Chuẩn bị dữ liệu cho mô hình ABSA, huấn luyện và tối ưu siêu tham số (Sweep) cho các mô hình ABSA (PhoBERT, LogReg, RF, CNN, BiGRU), đánh giá hiệu năng mô hình, sử dụng W&B để ghi nhận và quản lý toàn bộ quá trình huấn luyện. Viết báo cáo. |
| 23520556 | Đoàn Nhật Hưng | Xây dựng RAG Chatbot và triển khai hạ tầng MLOps: thiết kế RAG pipeline (FastAPI, Chainlit, hybrid search Qdrant, Redis embedding cache, multi-turn conversation), Airflow DAG 3 (ABSA Inference) và DAG 4 (KB Incremental Rebuild), tích hợp PhoBERT ONNX + BGE-M3 ONNX, Prometheus + Grafana monitoring, RAGAS evaluation, Docker Compose 12 services. [Thực hành] AWS EC2 cloud deployment, AWS CloudWatch cloud logging, Grafana alerting → Telegram, CI/CD mở rộng auto-deploy lên EC2. |

---

## 📌 Mô tả đề tài

Xây dựng hệ thống RAG Chatbot hỏi đáp sản phẩm Tiki tích hợp Aspect-Based Sentiment Analysis (ABSA), cho phép người dùng:
- Hỏi về sản phẩm thuộc ngành **Nhà cửa & Đời sống** trên Tiki
- Nhận phân tích cảm xúc theo **6 khía cạnh**: chất lượng, giá cả, mô tả sản phẩm, dịch vụ, đóng gói, giao hàng
- Paste link sản phẩm Tiki để phân tích real-time qua **URL Analyzer**
- Hội thoại multi-turn với **query rewriting** tự động

**Dữ liệu:** ~13,000 product reviews, 1,000 product details từ Tiki  
**Domain:** Ngành hàng Nhà cửa & Đời sống

---

## 🏗️ Kiến trúc tổng quan hệ thống

![Architecture Overview](docs/images/architecture_overview.jpg)

---

## 🛠️ Tech Stack

![Tech Stack](docs/images/tech_stack.jpg)

| Component | Technology | Config |
|-----------|-----------|--------|
| **Orchestration** | Apache Airflow 2.7 | CeleryExecutor, Redis broker, PostgreSQL backend |
| **Vector DB** | Qdrant | Collection `tiki_kb`, hybrid search (dense + sparse), port 6333 |
| **Embedding** | BGE-M3 ONNX | ONNX Runtime CPU, 1024-dim vectors |
| **ABSA (offline)** | PhoBERT v2 ONNX | 6 classification heads, avg F1-macro 84.8% |
| **ABSA (realtime)** | Logistic Regression | TF-IDF, realtime URL analysis |
| **LLM** | Groq LLaMA 3.3 70B | Paid Developer tier + OpenRouter fallback |
| **Cache** | Redis 7 | TTL 1hr, smart warm cache on startup (~100 queries) |
| **API** | FastAPI | 2 uvicorn workers, port 8000 |
| **UI** | Chainlit | Multi-turn memory, URL Analyzer, port 8001 |
| **Storage** | AWS S3 | Bucket: tiki-crawl-data |
| **Experiment Tracking** | MLflow + W&B | MLflow port 5000, W&B org: cs317-mlops-org |
| **Monitoring** | Prometheus + Grafana | Metrics scrape /metrics, dashboards port 3000 |
| **Load Testing** | Locust | Port 8089, target /chat endpoint |
| **Infra** | Docker Compose | 12 services, shared network |
| **CI/CD** | GitHub Actions | Self-hosted runner + AWS EC2, trigger on push main |

---

## 🔄 Data Pipeline

![Data Pipeline](docs/images/data_pipeline.jpg)

4 Airflow DAGs chạy hàng ngày:

| DAG | Tên | Mô tả | Schedule |
|-----|-----|-------|----------|
| DAG 1 | `tiki_only_sync_s3` | Crawl sản phẩm & reviews từ Tiki API → upload S3 | Daily |
| DAG 2 | `preprocess_pipeline` | Làm sạch data, generate training set, label | Daily |
| DAG 3 | `absa_inference_pipeline` | ABSA inference (PhoBERT ONNX), check drift, W&B alert | Daily |
| DAG 4 | `kb_rebuild_pipeline` | Incremental KB update, embed với BGE-M3, upsert Qdrant | Daily |

---

## 🤖 Serving & Response Flow

![Serving Flow](docs/images/serving_flow.jpg)

---

## ✨ Phần bổ sung cho học phần Thực hành

> Phần này **thêm mới hoàn toàn** so với đồ án lý thuyết, đáp ứng yêu cầu thực hành CS317.

### 1. 🌩️ Cloud Infrastructure Deployment (AWS EC2)

Triển khai toàn bộ 12 services lên **AWS EC2 t3.xlarge** (4 vCPU, 16GB RAM, Singapore region):

- **Chatbot UI:** `http://13.251.141.210:8001`
- **API:** `http://13.251.141.210:8000`
- **Static Elastic IP:** `13.251.141.210`
- Hệ thống accessible từ mọi nơi, không cần VPN

**Cách triển khai:**
```bash
# SSH vào EC2
ssh -i your-key.pem ubuntu@13.251.141.210

# Deploy toàn bộ stack
cd ~/sentiment-chatbot-mlops
docker compose up -d

# Kiểm tra 12 services
docker compose ps
```

### 2. 📋 Cloud Logging (AWS CloudWatch)

Tích hợp **AWS CloudWatch Agent** ship logs từ tất cả 12 Docker containers lên cloud:

- **Log group:** `tiki-mlops` (region: ap-southeast-1)
- **Log stream:** `docker-containers` — tổng hợp log toàn bộ services
- Accessible từ AWS Console mọi lúc, không cần SSH

```bash
# Kiểm tra CloudWatch Agent
sudo systemctl status amazon-cloudwatch-agent

# Xem log groups
aws logs describe-log-groups --region ap-southeast-1
```

### 3. 🔔 Grafana Alerting → Telegram

3 alert rules tự động gửi notification qua **Telegram bot** khi có sự cố:

| Alert Rule | Condition | Pending |
|-----------|-----------|---------|
| API Down | `up{job="tiki-api"} < 1` | 1 phút |
| High Error Rate | Error rate > 5% trong 5 phút | 2 phút |
| High Latency | p95 latency > 10s | 2 phút |

### 4. 🔄 CI/CD mở rộng sang AWS EC2

GitHub Actions tự động deploy lên **cả 2 môi trường** khi push lên `main`:

```
push to main
    ↓
Job 1: deploy (server thầy — self-hosted runner)
    ↓ nếu thành công
Job 2: deploy-ec2 (AWS EC2 — appleboy/ssh-action)  ← MỚI
    → git pull + docker compose build + restart services
```

---

## 📊 Kết quả đánh giá

### ABSA Model Comparison

| Model | Avg F1 (macro) | Speed | Dùng cho |
|-------|---------------|-------|---------|
| **PhoBERT v2 ONNX** | **84.8%** | Offline | KB rebuild (DAG 3) |
| Logistic Regression | ~72% | ~1ms | Realtime URL Analyzer |
| MultiHeadTextCNN | ~68% | Fast | Baseline comparison |
| MultiHeadBiGRU | ~70% | Fast | Baseline comparison |

### RAG Evaluation (RAGAS, 20 queries)

| Metric | Score |
|--------|-------|
| Faithfulness | 0.494 |
| Answer Relevancy | 0.145 |

### Load Testing (Locust)

- Max **20 concurrent users** tại 0% failure rate
- Bottleneck: Groq API rate limit (25 RPM), không phải infrastructure
- Redis cache hit: **~16ms** vs cold start **~10s**

---

## 🚀 Hướng dẫn cài đặt

### Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|-----------|---------|
| OS | Ubuntu 20.04+ hoặc Windows (WSL2) |
| RAM | ≥ 8GB (khuyến nghị 16GB) |
| Disk | ≥ 50GB |
| Docker | ≥ 24.x + Docker Compose v2 |
| Python | 3.8+ |

### Bước 1 — Clone repo

```bash
git clone https://github.com/phngahn/sentiment-chatbot-mlops
cd sentiment-chatbot-mlops
```

### Bước 2 — Tạo file .env

```bash
cp .env.example .env
```

Điền các biến bắt buộc trong `.env`:

```env
# LLM
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_key   # fallback LLM

# AWS S3
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=ap-southeast-1
S3_BUCKET_NAME=tiki-crawl-data

# W&B (cho ABSA team)
WANDB_API_KEY=your_wandb_key
WANDB_ENTITY=cs317-mlops-org

# Airflow
AIRFLOW_UID=1000

# MLflow
MLFLOW_TRACKING_URI=http://tiki-mlflow:5000
```

### Bước 3 — Copy model files

ONNX model files không được commit lên git do kích thước lớn. Cần copy thủ công:

```bash
# Cấu trúc cần có:
models/
├── absa/
│   └── v2/
│       ├── phobert_onnx/
│       │   └── phobert_absa.onnx     # PhoBERT ONNX
│       └── logreg/
│           └── logreg_model.pkl      # LogReg model
└── bge-m3-onnx/
    └── bge_m3_dense.onnx             # BGE-M3 ONNX

# Hoặc export lại từ script:
python scripts/convert_phobert_onnx.py
python scripts/convert_bgem3_onnx.py
```

### Bước 4 — Build và start

```bash
# Build tất cả images
docker compose build

# Start 12 services
docker compose up -d

# Kiểm tra trạng thái
docker compose ps
```

### Bước 5 — Khởi tạo Knowledge Base (lần đầu)

```bash
# Trigger DAG 4 để build KB từ đầu (cần có data trong S3)
# Hoặc restore từ Qdrant snapshot:
curl -X POST "http://localhost:6333/collections/tiki_kb/snapshots/upload?priority=snapshot" \
  -H "Content-Type: multipart/form-data" \
  -F "snapshot=@tiki_kb.snapshot"

# Kiểm tra KB
curl http://localhost:6333/collections/tiki_kb | python3 -m json.tool
```

---

## 🖥️ Truy cập services

| Service | URL (Local) | URL (Cloud) | Credentials |
|---------|------------|-------------|-------------|
| 🤖 Chatbot UI | http://localhost:8001 | http://13.251.141.210:8001 | — |
| ⚡ API Docs | http://localhost:8000/docs | http://13.251.141.210:8000/docs | — |
| 🌀 Airflow | http://localhost:8080 | SSH tunnel | admin/admin |
| 📊 MLflow | http://localhost:5000 | SSH tunnel | — |
| 📈 Grafana | http://localhost:3000 | http://13.251.141.210:3000 | admin/admin |
| 🔥 Prometheus | http://localhost:9090 | SSH tunnel | — |
| 🦗 Locust | http://localhost:8089 | SSH tunnel | — |

**SSH tunnel để truy cập nội bộ:**
```bash
ssh -i your-key.pem -L 8080:localhost:8080 ubuntu@13.251.141.210
# Sau đó vào http://localhost:8080
```

---

## 🧪 Kiểm thử

### Health check

```bash
# API health
curl http://localhost:8000/health
# Expected: {"status":"ok"}

# Qdrant KB
curl -s http://localhost:6333/collections/tiki_kb | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])"
# Expected: 2962
```

### Test chatbot

```bash
# Hỏi về sản phẩm
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Nồi cơm điện nào tốt nhất?", "session_id": "test-001"}'

# URL Analyzer
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "https://tiki.vn/noi-com-dien-abc", "session_id": "test-002"}'
```

### Test ABSA inference

```bash
docker exec tiki-api python3 -c "
from src.absa.inference import PhoBERTONNXPredictor
model = PhoBERTONNXPredictor()
result = model.predict('Sản phẩm tốt, giao hàng nhanh, giá hơi cao')
print(result)
"
```

### RAGAS Evaluation

```bash
docker exec tiki-api python /app/scripts/ragas_eval.py
# Results logged to MLflow experiment: ragas_evaluation
```

### Load Testing

```bash
# Mở Locust UI
open http://localhost:8089
# Hoặc headless:
docker compose exec locust locust \
  --headless -u 20 -r 2 --run-time 120s \
  --host http://tiki-api:8000
```

### Kiểm tra CloudWatch logs (cloud)

```bash
# Xem log groups
aws logs describe-log-groups --region ap-southeast-1

# Xem log stream
aws logs get-log-events \
  --log-group-name tiki-mlops \
  --log-stream-name docker-containers \
  --region ap-southeast-1 \
  --limit 20
```

---

## 📁 Cấu trúc thư mục

```
sentiment-chatbot-mlops/
├── src/
│   ├── chatbot/           # RAG pipeline
│   │   ├── api.py         # FastAPI endpoints, warm cache
│   │   ├── retrieval.py   # Qdrant hybrid search, Redis cache
│   │   └── llm.py         # Groq LLM, query rewriting
│   ├── absa/              # ABSA inference
│   │   └── inference.py   # PhoBERT ONNX + LogReg predictor
│   ├── crawling/          # Tiki crawler
│   └── kb/                # Knowledge base builder
│       ├── index_qdrant.py       # Full KB rebuild
│       └── index_qdrant_delta.py # Incremental update
├── dags/                  # Airflow DAGs (4 pipelines)
├── scripts/               # ONNX export, RAGAS eval, drift check
├── models/                # ONNX model files (gitignored)
├── monitoring/            # Prometheus config, Grafana dashboards
│   └── promtail-config.yaml    # CloudWatch Agent config ← MỚI
├── docker/                # Dockerfiles
├── tests/                 # Load tests (Locust)
├── docs/
│   └── images/            # Architecture diagrams
├── .github/workflows/
│   └── deploy.yml         # CI/CD: server thầy + AWS EC2 ← MỚI
└── docker-compose.yml     # 12 services definition
```

---

## ⚙️ MLOps Pipeline hoàn chỉnh

```
Tiki API
    ↓ DAG 1 (Airflow)
AWS S3 (raw data)
    ↓ DAG 2 (Airflow)
Cleaned + Labeled CSV
    ↓ DAG 3 (Airflow)
ABSA Inference (PhoBERT ONNX)
    ├── Log F1 drift → W&B Alert (notify team)
    └── Log metrics → MLflow
    ↓ DAG 4 (Airflow)
Qdrant KB (2,962 vectors, BGE-M3)
    ↓
FastAPI (2 workers) + Redis Cache
    ↓
Chainlit UI (multi-turn, URL Analyzer)
    ↓
interactions.jsonl → RAGAS Eval → MLflow

Monitoring:
    Prometheus → Grafana → Telegram Alert ← MỚI
    CloudWatch (cloud logs) ← MỚI

Deployment:
    GitHub Actions → server thầy + AWS EC2 ← MỚI
```

---

## ⚠️ Limitations

- Domain giới hạn: chỉ Nhà cửa & Đời sống (~2,962 sản phẩm)
- Groq rate limit (25 RPM) giới hạn ~20 concurrent users
- PhoBERT auto-retrain cần GPU — dùng human-in-the-loop với W&B alert
- RAGAS answer_relevancy thấp (0.145) do test set có out-of-domain queries

---

## 📄 License

MIT License — CS317 Group 7, UIT 2026
