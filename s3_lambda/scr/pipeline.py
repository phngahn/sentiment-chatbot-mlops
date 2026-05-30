import os
import re
import time
import json
import glob
import logging
import pandas as pd
from datetime import datetime
from underthesea import word_tokenize
from google import genai
from google.genai.errors import ServerError
from tqdm import tqdm
import boto3
from botocore.exceptions import ClientError

os.environ.setdefault("WANDB_CACHE_DIR", "/tmp/wandb_cache")
os.environ.setdefault("WANDB_DATA_DIR",  "/tmp/wandb_data")
os.environ.setdefault("WANDB_DIR",       "/tmp/wandb")

import wandb

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# WW&B HELPERS
def create_wandb_evaluation_table(df: pd.DataFrame) -> wandb.Table:
    """
    Biến đổi dataframe kết quả thành wandb.Table để trực quan hóa trên UI W&B.
    Tách cột 'label' chứa list dict ABSA thành các cột khía cạnh riêng biệt.
    """
    aspect_list = ["description", "quality", "packaging", "delivery", "service", "price"]
    columns = [
        "product_id", "customer_name", "rating", 
        "content", "clean_content", "tokens"
    ] + [f"aspect/{a}" for a in aspect_list]

    table = wandb.Table(columns=columns)

    for _, row in df.iterrows():
        labels = row.get("label", "[]")
        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except Exception:
                labels = []
        
        sentiment_map = {}
        if isinstance(labels, list):
            for item in labels:
                if isinstance(item, dict):
                    sentiment_map[item.get("aspect")] = item.get("sentiment")
        
        row_data = [
            str(row.get("product_id", "")),
            str(row.get("customer_name", "")),
            int(row.get("rating", 0)) if pd.notnull(row.get("rating")) else 0,
            str(row.get("content", "")),
            str(row.get("clean_content", "")),
            str(row.get("tokens", ""))
        ]
        
        for aspect in aspect_list:
            row_data.append(sentiment_map.get(aspect, "none"))
            
        table.add_data(*row_data)
        
    return table


def log_pipeline_errors_to_table(df_raw: pd.DataFrame, df_processed: pd.DataFrame, df_labeled: pd.DataFrame) -> wandb.Table:
    """
    Gom nhóm và phân loại tất cả các dòng xử lý thất bại (Preprocess) 
    và dán nhãn thất bại (Label) dựa trên cột tạm _old_idx.
    """
    columns = ["product_id", "customer_name", "created_at", "content", "error_stage"]
    error_table = wandb.Table(columns=columns)
    
    processed_indices = set(df_processed["_old_idx"].dropna().astype(int).tolist()) if "_old_idx" in df_processed.columns else set(df_processed.index)
    labeled_indices = set(df_labeled["_old_idx"].dropna().astype(int).tolist()) if "_old_idx" in df_labeled.columns else set(df_labeled.index)
    
    df_failed_preprocess = df_raw[~df_raw.index.isin(processed_indices)]
    for _, row in df_failed_preprocess.iterrows():
        error_table.add_data(
            str(row.get("product_id", "")),
            str(row.get("customer_name", "")),
            str(row.get("created_at", "")),
            str(row.get("content", "")),
            "FAILED_PREPROCESS"
        )
        
    df_failed_label = df_processed[~df_processed["_old_idx"].isin(labeled_indices)] if "_old_idx" in df_processed.columns else df_processed[~df_processed.index.isin(labeled_indices)]
    for _, row in df_failed_label.iterrows():
        error_table.add_data(
            str(row.get("product_id", "")),
            str(row.get("customer_name", "")),
            str(row.get("created_at", "")),
            str(row.get("content", "")),
            "FAILED_LABELING"
        )
        
    return error_table

def log_advanced_pipeline_plots(df_raw: pd.DataFrame, df_labeled: pd.DataFrame):
    """
    Tạo và đẩy toàn bộ các biểu đồ thống kê (Độ dài, Aspect Sentiment) lên W&B.
    """

    # Histogram số từ
    if "clean_content" in df_labeled.columns:
        word_counts = df_labeled["clean_content"].dropna().apply(lambda x: len(str(x).split())).tolist()
        word_count_table = wandb.Table(data=[[wc] for wc in word_counts], columns=["word_count"])
        wandb.log({
            "plots/review_length_distribution": wandb.plot.histogram(
                word_count_table, "word_count", title="Phân phối Độ dài Review (Số từ)"
            )
        })

    # Bar Chart
    aspect_list = ["description", "quality", "packaging", "delivery", "service", "price"]
    
    aspect_plot_data = []
    
    counts = {a: {"positive": 0, "negative": 0, "neutral": 0} for a in aspect_list}
    
    for _, row in df_labeled.iterrows():
        labels = row.get("label", "[]")
        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except Exception:
                labels = []
        
        if isinstance(labels, list):
            for item in labels:
                if isinstance(item, dict):
                    asp = item.get("aspect")
                    sent = item.get("sentiment")
                    if asp in counts and sent in counts[asp]:
                        counts[asp][sent] += 1
                        
    for asp in aspect_list:
        for sent in ["positive", "negative", "neutral"]:
            aspect_plot_data.append([asp, sent, counts[asp][sent]])
            
    aspect_table = wandb.Table(data=aspect_plot_data, columns=["Aspect", "Sentiment", "Count"])
    
    wandb.log({
        "plots/aspect_sentiment_distribution": wandb.plot.bar(
            aspect_table, "Aspect", "Count", title="Phân phối cảm xúc theo Khía cạnh"
        )
    })

# Config
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
INPUT_BUCKET    = os.environ.get("INPUT_BUCKET",  "my-reviews-data")
OUTPUT_BUCKET   = os.environ.get("OUTPUT_BUCKET", "my-reviews-data")
OUTPUT_PREFIX   = os.environ.get("OUTPUT_PREFIX", "processed/")
CHECKPOINT_DIR  = os.environ.get("CHECKPOINT_DIR", "/tmp/checkpoints")
GEMINI_MODEL    = os.environ.get("GEMINI_MODEL",  "gemini-3.1-flash-lite")

# Rate limit: 15 req/min -> sleep 4s
GEMINI_SLEEP    = float(os.environ.get("GEMINI_SLEEP", "4"))

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

client    = genai.Client(api_key=GEMINI_API_KEY)
s3_client = boto3.client("s3")


#  S3 CHECKPOINT HELPERS
def _ckpt_s3_key(base_name: str, kind: str) -> str:
    folder = "preprocessed" if kind == "preprocess" else "labeled"
    return f"checkpoints/{folder}/{kind}_{base_name}.csv"


def upload_checkpoint(local_path: str, base_name: str, kind: str) -> None:
    """Upload checkpoint lên S3 sau mỗi batch."""
    try:
        s3_client.upload_file(local_path, OUTPUT_BUCKET, _ckpt_s3_key(base_name, kind))
    except Exception as e:
        log.warning("Không upload được checkpoint lên S3: %s", e)


def download_checkpoint(local_path: str, base_name: str, kind: str) -> bool:
    """Download checkpoint từ S3 về /tmp. Trả về True nếu có."""
    try:
        s3_client.download_file(OUTPUT_BUCKET, _ckpt_s3_key(base_name, kind), local_path)
        log.info("Đã tải checkpoint %s từ S3.", kind)
        return True
    except ClientError:
        return False

#  HELPERS
def call_gemini(prompt: str, retries: int = 5, delay: int = 5) -> str:
    for i in range(retries):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={"temperature": 0.2},
            )
            return resp.text
        except ServerError:
            log.warning("Gemini 500 – retry %d/%d", i + 1, retries)
            time.sleep(delay * (i + 1))
    log.error("Gemini unreachable after %d retries – returning prompt as-is.", retries)
    return prompt


def clean_text(text: str) -> str:
    text = str(text).lower().replace("\n", " ")
    text = re.sub(r'(?<!\d)\.|\.(?!\d)', ' ', text)
    text = re.sub(r'(?<!\d)/|/(?!\d)', ' ', text)
    text = re.sub(
        r'[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ'
        r'ìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữ'
        r'ỳýỵỷỹđ\s\.\/]', '', text,
    )
    return re.sub(r'\s+', ' ', text).strip()

#  S3 FILE HELPERS
def download_from_s3(bucket: str, key: str, local_path: str) -> None:
    log.info("Downloading s3://%s/%s → %s", bucket, key, local_path)
    s3_client.download_file(bucket, key, local_path)


def upload_to_s3(local_path: str, bucket: str, key: str) -> None:
    log.info("Uploading %s → s3://%s/%s", local_path, bucket, key)
    s3_client.upload_file(local_path, bucket, key)

#  PREPROCESS
def _process_batch(batch: list[str]) -> list[str]:
    prompt = (
        "Bạn là một công cụ chuẩn hóa chính tả.\n"
        "Hãy sửa chính tả, ngữ pháp và dấu câu cho các câu đánh giá sản phẩm sau.\n"
        "Giữ nguyên nội dung, ý nghĩa và phong cách tự nhiên của người mua hàng, "
        "chỉ sửa lỗi chính tả, chuẩn hóa từ viết tắt cho review.\n"
        f"Trả về đúng {len(batch)} dòng, mỗi câu một dòng, theo đúng thứ tự, "
        "không giải thích, mô tả hay dẫn dắt thêm gì.\n\n"
        + "\n".join(f"{i+1}. {line}" for i, line in enumerate(batch))
    )
    corrected = call_gemini(prompt)
    time.sleep(GEMINI_SLEEP)

    lines = [
        re.sub(r"^\d+\.\s*", "", l).strip()
        for l in corrected.split("\n") if l.strip()
    ]
    if len(lines) > len(batch):
        lines = lines[: len(batch)]
    elif len(lines) < len(batch):
        lines = lines + batch[len(lines):]
    return lines

def preprocess(df: pd.DataFrame, checkpoint_path: str, base_name: str, already_downloaded=False) -> pd.DataFrame:
    """Deduplicate -> clean -> spell-correct -> tokenise. Checkpoint trên S3."""
    df["_old_idx"] = df.index

    id_cols = ["product_id", "customer_name", "rating", "content"]
    df = df.drop_duplicates(subset=id_cols).reset_index(drop=True)
    df["rating"]     = df["rating"].astype(int)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["normalized_content"] = df["content"].apply(clean_text)
    df = df[df["normalized_content"].str.strip().str.len() > 0].reset_index(drop=True)
    df["clean_content"] = None
    df["tokens"]        = None

    # Resume S3 checkpoint
    if not already_downloaded:
        download_checkpoint(checkpoint_path, base_name, "preprocess")
        
    if os.path.exists(checkpoint_path):
        ckpt = pd.read_csv(checkpoint_path)
        done = ckpt[ckpt["clean_content"].notna()][
            id_cols + ["created_at", "clean_content", "tokens"]
        ].drop_duplicates(subset=id_cols).copy()

        if not done.empty:
            done["created_at"] = pd.to_datetime(done["created_at"])
            done["rating"]     = done["rating"].astype(int)
            df = df.merge(done, on=id_cols, how="left", suffixes=("", "_ckpt"))
            df = df.drop_duplicates(subset=id_cols).reset_index(drop=True)
            for col in ["clean_content", "tokens"]:
                ckpt_col = f"{col}_ckpt"
                if ckpt_col in df.columns:
                    df[col] = df[ckpt_col].combine_first(df[col])
                    df.drop(columns=[ckpt_col], inplace=True)
            log.info("Resumed preprocess: %d/%d rows từ checkpoint.", df["clean_content"].notna().sum(), len(df))

    # Process remaining
    todo_mask = df["clean_content"].isna()
    todo_df   = df[todo_mask]

    if todo_df.empty:
        log.info("Preprocess: nothing to do.")
    else:
        batch_size = 20
        contents   = todo_df["normalized_content"].tolist()
        positions  = todo_df.index.tolist()

        with tqdm(total=len(todo_df), desc="Preprocessing", unit="row") as pbar:
            for i in range(0, len(contents), batch_size):
                batch_c = contents[i : i + batch_size]
                batch_p = positions[i : i + batch_size]

                corrected   = _process_batch(batch_c)
                clean_batch = [clean_text(line) for line in corrected]
                token_batch = [word_tokenize(line, format="text") for line in clean_batch]

                df.loc[batch_p, "clean_content"] = clean_batch
                df.loc[batch_p, "tokens"]        = token_batch
                pbar.update(len(batch_p))

                # Lưu checkpoint local + upload S3
                df.to_csv(checkpoint_path, index=False)
                upload_checkpoint(checkpoint_path, base_name, "preprocess")
    
    df.drop(columns=["normalized_content"], inplace=True, errors="ignore")

    return df

#  LABEL
ASPECTS = ["description", "quality", "packaging", "delivery", "service", "price"]


def _batch_labeling(batch: list[str]) -> list[list | None]:
    if not batch:
        return []

    prompt = (
        "Bạn là chuyên gia Phân tích Cảm xúc Dựa trên Khía cạnh (ABSA).\n"
        "Nhiệm vụ: Trích xuất các cặp Khía cạnh - Cảm xúc từ danh sách review dưới đây.\n\n"
        "ĐỊNH NGHĨA KHÍA CẠNH (Chỉ dùng 6 nhãn này):\n"
        "1. description: Sự khớp giữa quảng cáo và thực tế.\n"
        "2. quality: Ngoại hình, công năng, độ bền, vật liệu, hiệu quả sử dụng.\n"
        "3. packaging: Bao bì, hộp, chống sốc, độ an toàn của kiện hàng.\n"
        "4. delivery: Tốc độ vận chuyển và thái độ của người giao hàng.\n"
        "5. service: Thái độ tư vấn của shop, hỗ trợ đổi trả, quà tặng.\n"
        "6. price: Sự đắt/rẻ, tính xứng đáng với số tiền bỏ ra.\n"
        "ĐỊNH NGHĨA CẢM XÚC: positive, negative, neutral\n\n"
        "QUY TẮC:\n"
        "1. Phân tích chi tiết từng từ ngữ để ánh xạ chính xác vào 6 khía cạnh.\n"
        "2. Nếu câu hoàn toàn vô nghĩa hoặc là ký tự rác, trả về aspect_sentiments rỗng: [].\n"
        "3. Nếu review chỉ chứa duy nhất một từ ngữ mà không nói gì thêm (Ví dụ: 'tuyệt vời', 'tốt', 'tệ'),\n"
        "thì gán cảm xúc đó cho tất cả 6 khía cạnh. "
        "4. Nếu review có nhắc đến khía cạnh cụ thể, gán cảm xúc theo nội dung."
        "5. Nếu một khía cạnh có nhiều ý kiến trái chiều, ưu tiên 'negative'.\n"
        "6. Chỉ gán 'neutral' cho một khía cạnh cụ thể NẾU VÀ CHỈ NẾU review hoàn toàn KHÔNG NHẮC ĐẾN "
        "bất kỳ từ ngữ nào liên quan đến khía cạnh đó.\n"
        "hãy gán cảm xúc đó (positive/negative) cho CẢ 6 KHÍA CẠNH.\n"
        "7. Đầu ra là DUY NHẤT một mảng JSON. Mỗi phần tử:\n"
        '   {"id": <stt>, "aspect_sentiments": [{"aspect": "...", "sentiment": "..."}]}\n'
        f"8. Trả về đúng {len(batch)} phần tử, đúng thứ tự.\n\n"
        "DANH SÁCH REVIEW:\n"
        + "\n".join(f"{i+1}. {line}" for i, line in enumerate(batch))
    )

    try:
        response = call_gemini(prompt)
        time.sleep(GEMINI_SLEEP)  # rate limit

        match   = re.search(r'\[.*\]', response, re.DOTALL)
        results = json.loads(match.group() if match else response.strip())

        if not isinstance(results, list) or len(results) != len(batch):
            return [None] * len(batch)

        return [
            item.get("aspect_sentiments", []) if isinstance(item, dict) else []
            for item in results
        ]
    except Exception as exc:
        log.warning("Labeling batch failed: %s", exc)
        return [None] * len(batch)


def _post_filter(df: pd.DataFrame) -> pd.DataFrame:
    new_labels = []
    for val in df["label"]:
        try:
            aspects = json.loads(val) if pd.notna(val) else []
        except Exception:
            aspects = []

        if not aspects:
            new_labels.append(None)
            continue

        aspect_dict = {item["aspect"].strip().lower(): item["sentiment"] for item in aspects if "aspect" in item}
        fixed_aspects = [{"aspect": asp, "sentiment": aspect_dict.get(asp, "neutral")} for asp in ASPECTS]
        new_labels.append(json.dumps(fixed_aspects, ensure_ascii=False))

    df["label"] = new_labels
    return df.dropna(subset=["label"]).reset_index(drop=True)


def label(df: pd.DataFrame, checkpoint_path: str, base_name: str) -> pd.DataFrame:
    """ABSA labeling with checkpoint S3."""

    id_cols = ["product_id", "customer_name", "created_at", "clean_content"]
    
    # Ép kiểu cột định danh về chuỗi ngay từ đầu để đồng bộ tuyệt đối
    for col in id_cols:
        df[col] = df[col].astype(str).fillna("Unknown").str.strip()

    if "label" not in df.columns:
        df["label"] = None

    # Resume S3 checkpoint
    download_checkpoint(checkpoint_path, base_name, "label")

    if os.path.exists(checkpoint_path):
        ckpt = pd.read_csv(checkpoint_path)
        
        # Ép kiểu chuỗi cho các cột định danh của checkpoint tương tự như df
        for col in id_cols:
            ckpt[col] = ckpt[col].astype(str).fillna("Unknown").str.strip()

        # Lọc lấy những bản ghi đã gán nhãn
        done = ckpt[ckpt["label"].notna()][id_cols + ["label"]].drop_duplicates(subset=id_cols)
        
        if not done.empty:
            df = df.merge(done, on=id_cols, how="left", suffixes=("", "_ckpt"))
            if "label_ckpt" in df.columns:
                df["label"] = df["label_ckpt"].combine_first(df["label"])
                df.drop(columns=["label_ckpt"], inplace=True)
            log.info("Resumed label: %d labels từ checkpoint.", df["label"].notna().sum())

    todo_mask = df["label"].isna()
    todo_df   = df[todo_mask]

    if todo_df.empty:
        log.info("Labeling: nothing to do.")
    else:
        batch_size = 50
        contents   = todo_df["clean_content"].tolist()
        positions  = todo_df.index.tolist()

        with tqdm(total=len(todo_df), desc="Labeling", unit="row") as pbar:
            for i in range(0, len(contents), batch_size):
                batch_c = contents[i : i + batch_size]
                batch_p = positions[i : i + batch_size]

                labels = _batch_labeling(batch_c)
                df.loc[batch_p, "label"] = [
                    json.dumps(res, ensure_ascii=False) if res is not None else None
                    for res in labels
                ]
                pbar.update(len(batch_p))

                # Lưu checkpoint local + upload S3
                df.to_csv(checkpoint_path, index=False)
                upload_checkpoint(checkpoint_path, base_name, "label")

    return _post_filter(df)

def delete_all_checkpoints_for_base(base_name: str):
    """
    Xóa tất cả các file checkpoint, file staged, file raw/chunks 
    và cấu hình meta liên quan đến một base_name cụ thể trên S3.
    """
    folders_to_clean = [
        "checkpoints/preprocessed/", 
        "checkpoints/labeled/", 
        "checkpoints/stage/",
        "checkpoints/chunks/",
        CHUNK_PREFIX,
    ]
    
    for folder in folders_to_clean:
        prefix = folder
        paginator = s3_client.get_paginator("list_objects_v2")
        
        try:
            for page in paginator.paginate(Bucket=OUTPUT_BUCKET, Prefix=prefix):
                objects_to_delete = []
                
                for obj in page.get("Contents", []):
                    if base_name in obj["Key"]:
                        objects_to_delete.append({"Key": obj["Key"]})
                        
                if objects_to_delete:
                    s3_client.delete_objects(
                        Bucket=OUTPUT_BUCKET,
                        Delete={"Objects": objects_to_delete}
                    )
                    for item in objects_to_delete:
                        log.info("Đã dọn dẹp triệt để file tạm trên S3: %s", item["Key"])
                        
        except Exception as e:
            log.error("Lỗi khi dọn dẹp thư mục %s: %s", folder, e)

#  CHUNK HELPERS

CHUNK_SIZE      = int(os.environ.get("CHUNK_SIZE", "5000"))
CHUNK_PREFIX    = os.environ.get("CHUNK_PREFIX", "raw/chunks/")


def _s3_exists(key: str) -> bool:
    """Kiểm tra key đã tồn tại trên OUTPUT_BUCKET chưa."""
    try:
        s3_client.head_object(Bucket=OUTPUT_BUCKET, Key=key)
        return True
    except ClientError:
        return False

def _chunkmeta_s3_key(base_name: str) -> str:
    return f"checkpoints/chunks/meta_{base_name}.json"


def _chunkdone_s3_key(base_name: str, chunk_idx: int) -> str:
    return f"checkpoints/chunks/done_{base_name}_{chunk_idx:03d}"


def _load_chunkmeta(base_name: str) -> dict | None:
    """Load metadata của file gốc từ S3. None nếu chưa có."""
    try:
        obj = s3_client.get_object(Bucket=OUTPUT_BUCKET, Key=_chunkmeta_s3_key(base_name))
        return json.loads(obj["Body"].read())
    except ClientError:
        return None


def _save_chunkmeta(base_name: str, meta: dict) -> None:
    s3_client.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=_chunkmeta_s3_key(base_name),
        Body=json.dumps(meta, ensure_ascii=False),
        ContentType="application/json",
    )
    log.info("Saved chunkmeta for %s: %s", base_name, meta)


def _mark_chunk_done(base_name: str, chunk_idx: int, preprocessed_key: str, labeled_key: str) -> None:
    s3_client.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=_chunkdone_s3_key(base_name, chunk_idx),
        Body=json.dumps({
            "preprocessed_key": preprocessed_key,
            "labeled_key":      labeled_key,
            "done_date":        datetime.utcnow().strftime("%Y-%m-%d"),  # ngày done theo UTC
        }),
        ContentType="application/json",
    )


def _get_done_chunks(base_name: str, total_chunks: int) -> list[int]:
    """Trả về danh sách chunk index đã xong (tất cả các ngày)."""
    done = []
    for i in range(total_chunks):
        try:
            s3_client.head_object(Bucket=OUTPUT_BUCKET, Key=_chunkdone_s3_key(base_name, i))
            done.append(i)
        except ClientError:
            pass
    return done

def split_and_upload_chunks(bucket: str, key: str, df: pd.DataFrame, base_name: str) -> dict:
    """
    Chia df thành chunks, lưu tất cả local, chỉ upload chunk_000 lên S3.
    Các chunk sau sẽ được upload tuần tự sau khi chunk trước done.
    Trả về meta dict.
    """
    chunks = [df.iloc[i: i + CHUNK_SIZE] for i in range(0, len(df), CHUNK_SIZE)]
    total  = len(chunks)
    log.info("Splitting %d rows → %d chunks of max %d", len(df), total, CHUNK_SIZE)

    chunk_keys = []
    for idx, chunk_df in enumerate(chunks):
        chunk_filename = f"{base_name}_chunk_{idx:03d}.csv"
        chunk_key      = f"{CHUNK_PREFIX}{chunk_filename}"
        local_path     = f"/tmp/{chunk_filename}"
        chunk_df.to_csv(local_path, index=False)
        chunk_keys.append(chunk_key)

        if idx == 0:
            upload_to_s3(local_path, bucket, chunk_key)
            log.info("Uploaded chunk_000 → s3://%s/%s", bucket, chunk_key)

    meta = {
        "original_key":  key,
        "base_name":     base_name,
        "total_chunks":  total,
        "total_rows":    len(df),
        "chunk_size":    CHUNK_SIZE,
        "chunk_keys":    chunk_keys,
        "created_at":    datetime.utcnow().isoformat(),
    }
    _save_chunkmeta(base_name, meta)
    return meta


def concat_chunks_and_upload(bucket: str, meta: dict) -> tuple:
    """
    Khi TẤT CẢ chunks đã labeled xong, download về, concat lại, lưu local rồi upload final.
    Upload là lần DUY NHẤT các chunk được ghi vào processed/ và labeled/.
    Trả về (preprocessed_key, labeled_key, pre_local_out, lbl_local_out).
    """
    base_name     = meta["base_name"]
    total_chunks  = meta["total_chunks"]
    original_name = os.path.basename(meta["original_key"])
    pre_dfs, lbl_dfs = [], []

    for idx in range(total_chunks):
        done_obj  = s3_client.get_object(Bucket=OUTPUT_BUCKET, Key=_chunkdone_s3_key(base_name, idx))
        done_info = json.loads(done_obj["Body"].read())

        pre_local = f"/tmp/concat_pre_{idx:03d}.csv"
        s3_client.download_file(OUTPUT_BUCKET, done_info["preprocessed_key"], pre_local)
        pre_dfs.append(pd.read_csv(pre_local))

        lbl_local = f"/tmp/concat_lbl_{idx:03d}.csv"
        s3_client.download_file(OUTPUT_BUCKET, done_info["labeled_key"], lbl_local)
        lbl_dfs.append(pd.read_csv(lbl_local))

    df_pre = pd.concat(pre_dfs, ignore_index=True)
    df_lbl = pd.concat(lbl_dfs, ignore_index=True)

    final_cols = ["product_id", "customer_name", "rating", "content", "created_at", "clean_content", "tokens"]
    
    pre_cols = [c for c in final_cols if c in df_pre.columns]
    df_pre = df_pre[pre_cols]
    
    lbl_cols = [c for c in final_cols + ["label"] if c in df_lbl.columns]
    df_lbl = df_lbl[lbl_cols]

    pre_local_out = f"/tmp/preprocessed_{original_name}"
    lbl_local_out = f"/tmp/labeled_{original_name}"
    df_pre.to_csv(pre_local_out, index=False)
    df_lbl.to_csv(lbl_local_out, index=False)

    preprocessed_key = f"{OUTPUT_PREFIX}preprocessed/preprocessed_{original_name}"
    labeled_key      = f"{OUTPUT_PREFIX}labeled/labeled_{original_name}"
    upload_to_s3(pre_local_out, OUTPUT_BUCKET, preprocessed_key)
    upload_to_s3(lbl_local_out, OUTPUT_BUCKET, labeled_key)
    log.info(
        "Chunk concat: pre=%d rows -> %s | lbl=%d rows → %s",
        len(df_pre), preprocessed_key, len(df_lbl), labeled_key,
    )

    return preprocessed_key, labeled_key, pre_local_out, lbl_local_out


#  INCREMENTAL FINAL CONCAT

FINAL_PREFIX            = os.environ.get("FINAL_PREFIX", "final/")
FINAL_LABELED_PREFIX    = f"{FINAL_PREFIX}labeled/"
FINAL_PRE_PREFIX        = f"{FINAL_PREFIX}preprocessed/"
_DEDUP_COLS             = ["product_id", "customer_name", "rating", "content"]


def _final_labeled_key_for(original_filename: str) -> str:
    stem = os.path.splitext(original_filename)[0]
    return f"{FINAL_LABELED_PREFIX}all_labeled_{stem}.csv"


def _final_pre_key_for(original_filename: str) -> str:
    stem = os.path.splitext(original_filename)[0]
    return f"{FINAL_PRE_PREFIX}all_preprocessed_{stem}.csv"


def _latest_final_key(prefix: str) -> str | None:
    """Tìm file mới nhất trong prefix trên OUTPUT_BUCKET."""
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        objects   = []
        for page in paginator.paginate(Bucket=OUTPUT_BUCKET, Prefix=prefix):
            objects.extend(page.get("Contents", []))
        if not objects:
            return None
        return max(objects, key=lambda o: o["LastModified"])["Key"]
    except ClientError:
        return None


def _concat_incremental(new_local: str, prefix: str, new_final_key: str, dedup: bool = True) -> tuple:
    """
    Download file mới nhất trong prefix làm base, concat với file mới, dedup.
    Kiểm tra real-time để tránh Race Condition.
    """
    df_new = pd.read_csv(new_local)
    latest_key = _latest_final_key(prefix)

    if latest_key:
        base_local = "/tmp/_base_final_realtime.csv"
        if os.path.exists(base_local):
            os.remove(base_local)
            
        try:
            s3_client.download_file(OUTPUT_BUCKET, latest_key, base_local)
            df_base = pd.read_csv(base_local)
            df_final = pd.concat([df_base, df_new], ignore_index=True)
            log.info("Cập nhật Real-time: Concat %s (%d rows) + mới (%d rows).", latest_key, len(df_base), len(df_new))
        except ClientError:
            df_final = df_new.copy()
            log.info("Không tải được base, tạo mới final từ %d rows.", len(df_new))
    else:
        df_final = df_new.copy()
        log.info("Chưa có file final trên S3. Tạo mới final hoàn toàn từ %d rows.", len(df_new))

    if dedup:
        before = len(df_final)
        df_final = df_final.drop_duplicates(
            subset=[c for c in _DEDUP_COLS if c in df_final.columns]
        ).reset_index(drop=True)
        log.info("Dedup final: %d → %d rows.", before, len(df_final))

    return df_final, latest_key


def append_to_final(new_labeled_local: str, new_pre_local: str, original_filename: str, wandb_run) -> tuple:
    """
    Tạo/cập nhật 2 file final:
      final/labeled/all_labeled_<stem>.csv
      final/preprocessed/all_preprocessed_<stem>.csv
    Trả về (labeled_key, pre_key).
    """
    stem = os.path.splitext(original_filename)[0]

    lbl_key       = _final_labeled_key_for(original_filename)
    lbl_local_out = f"/tmp/all_labeled_{stem}.csv"
    df_lbl, lbl_base = _concat_incremental(new_labeled_local, FINAL_LABELED_PREFIX, lbl_key)
    
    if not lbl_base and _s3_exists(lbl_key):
        log.warning("Phát hiện Race Condition! Luồng khác đã tạo file final trước. Tiến hành gộp lại...")
        df_lbl, lbl_base = _concat_incremental(new_labeled_local, FINAL_LABELED_PREFIX, lbl_key)

    lbl_cols = [c for c in ["product_id", "customer_name", "rating", "content", "created_at", "clean_content", "tokens", "label"] if c in df_lbl.columns]
    df_lbl = df_lbl[lbl_cols]
    
    df_lbl.to_csv(lbl_local_out, index=False)
    upload_to_s3(lbl_local_out, OUTPUT_BUCKET, lbl_key)

    lbl_artifact = wandb.Artifact(
        name="all-labeled-final",
        type="dataset",
        description="Tích lũy reviews đã labeled theo từng file gốc",
        metadata={
            "s3_key":      lbl_key,
            "based_on":    lbl_base,
            "total_rows":  len(df_lbl),
            "source_file": original_filename,
        },
    )
    lbl_artifact.add_file(lbl_local_out)
    wandb_run.log_artifact(lbl_artifact)
    wandb_run.log({"rows/final_labeled": len(df_lbl), "s3/final_labeled_key": lbl_key})

    pre_key       = _final_pre_key_for(original_filename)
    pre_local_out = f"/tmp/all_preprocessed_{stem}.csv"
    df_pre, pre_base = _concat_incremental(new_pre_local, FINAL_PRE_PREFIX, pre_key, dedup=False)

    if not pre_base and _s3_exists(pre_key):
        log.warning("Phát hiện Race Condition! Luồng khác đã tạo file final trước. Tiến hành gộp lại...")
        df_pre, pre_base = _concat_incremental(new_pre_local, FINAL_PRE_PREFIX, pre_key, dedup=False)
        
    pre_cols = [c for c in ["product_id", "customer_name", "rating", "content", "created_at", "clean_content", "tokens"] if c in df_pre.columns]
    df_pre = df_pre[pre_cols]
    
    df_pre.to_csv(pre_local_out, index=False)
    upload_to_s3(pre_local_out, OUTPUT_BUCKET, pre_key)

    pre_artifact = wandb.Artifact(
        name="all-preprocessed-final",
        type="dataset",
        description="Tích lũy reviews đã preprocessed theo từng file gốc",
        metadata={
            "s3_key":      pre_key,
            "based_on":    pre_base,
            "total_rows":  len(df_pre),
            "source_file": original_filename,
        },
    )
    pre_artifact.add_file(pre_local_out)
    wandb_run.log_artifact(pre_artifact)
    wandb_run.log({"rows/final_preprocessed": len(df_pre), "s3/final_pre_key": pre_key})

    log.info("Final labeled: %d rows -> %s", len(df_lbl), lbl_key)
    log.info("Final preprocessed: %d rows -> %s", len(df_pre), pre_key)
    return lbl_key, pre_key


#  ENTRY POINT

def run_pipeline(bucket: str, key: str) -> str:
    filename  = os.path.basename(key)
    base_name = os.path.splitext(filename)[0]
    local_raw = f"/tmp/{filename}"
    pre_local, lbl_local = None, None

    if "_chunk_" not in base_name:
        download_from_s3(bucket, key, local_raw)
        df_check = pd.read_csv(local_raw)
        meta = split_and_upload_chunks(bucket, key, df_check, base_name)
        return f"chunked:{meta['total_chunks']} chunks created"

    parts     = base_name.rsplit("_chunk_", 1)
    orig_base = parts[0]
    chunk_idx = int(parts[1])
    is_chunk  = True

    preprocess_ckpt  = os.path.join(CHECKPOINT_DIR, f"preprocess_{base_name}.csv")
    label_ckpt       = os.path.join(CHECKPOINT_DIR, f"label_{base_name}.csv")
    preprocessed_out = f"/tmp/processed_{filename}"
    labeled_out      = f"/tmp/labeled_processed_{filename}"

    is_resume = download_checkpoint(preprocess_ckpt, base_name, "preprocess")
    run_type  = "resume" if is_resume else "new"

    CURRENT_PROMPT_VERSION = "v1.2_fixed_aspects"
    CURRENT_SYSTEM_PROMPT = "Bạn là chuyên gia phân tích ABSA..." 

    # W&B init
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "reviews-pipeline"),
        name=f"{base_name}_{run_type}_{datetime.utcnow().strftime('%H%M%S')}",
        group=orig_base,   
        config={
            "bucket":       bucket,
            "key":          key,
            "gemini_model": GEMINI_MODEL,
            "gemini_sleep": GEMINI_SLEEP,
            "is_chunk":     is_chunk,
            "chunk_size":   CHUNK_SIZE if is_chunk else None,
            "chunk_idx":    chunk_idx,
            "prompt_version": CURRENT_PROMPT_VERSION,
            "system_prompt": CURRENT_SYSTEM_PROMPT,
            "gemini_temperature": 0.2, 
            "environment": "AWS_Lambda_Production"
        },
    )
    log_step    = chunk_idx if is_chunk else 0
    start_total = time.time()

    if is_chunk:
        run.log({"chunk/index": chunk_idx}, step=log_step)

    # Download raw file
    if not os.path.exists(local_raw):
        download_from_s3(bucket, key, local_raw)
    df_raw     = pd.read_csv(local_raw)
    total_rows = len(df_raw)
    log.info("Loaded %d rows from %s", total_rows, key)
    wandb.log({"rows/input": total_rows})

    # Preprocess
    start           = time.time()
    df_processed = preprocess(df_raw, preprocess_ckpt, base_name, already_downloaded=is_resume)
    preprocess_time = time.time() - start
    df_processed.to_csv(preprocessed_out, index=False)

    rows_after_preprocess = len(df_processed)
    wandb.log({
        "rows/after_preprocess":   rows_after_preprocess,
        "rows/dropped_preprocess": total_rows - rows_after_preprocess,
        "time/preprocess_sec":     round(preprocess_time, 2),
    })
    log.info("Preprocess done: %d rows in %.1fs", rows_after_preprocess, preprocess_time)

    # Label
    start      = time.time()
    df_labeled = label(df_processed.copy(), label_ckpt, base_name)
    label_time = time.time() - start
    df_labeled.to_csv(labeled_out, index=False)

    rows_labeled       = len(df_labeled)
    rows_failed        = rows_after_preprocess - rows_labeled
    label_success_rate = rows_labeled / rows_after_preprocess * 100 if rows_after_preprocess > 0 else 0

    ESTIMATED_INPUT_TOKENS_PER_ROW = 150
    ESTIMATED_OUTPUT_TOKENS_PER_ROW = 80
    total_input_tokens = rows_labeled * ESTIMATED_INPUT_TOKENS_PER_ROW
    total_output_tokens = rows_labeled * ESTIMATED_OUTPUT_TOKENS_PER_ROW
    estimated_cost_usd = ((total_input_tokens / 1_000_000) * 0.075) + ((total_output_tokens / 1_000_000) * 0.30)

    wandb.log({
        "rows/labeled":                   rows_labeled,
        "rows/label_failed":              rows_failed,
        "metrics/label_success_rate_pct": round(label_success_rate, 2),
        "time/label_sec":                 round(label_time, 2),
        "time/total_sec":                 round(time.time() - start_total, 2),
        "gemini/estimated_cost_usd":      round(estimated_cost_usd, 5),
        "gemini/total_tokens_processed":  total_input_tokens + total_output_tokens,
    })
    log.info("Labeling done: %d rows in %.1fs", rows_labeled, label_time)

    # Upload S3 + chunk logic
    if is_chunk:
        staged_pre_key = f"checkpoints/stage/pre_{filename}"
        staged_lbl_key = f"checkpoints/stage/lbl_{filename}"
        upload_to_s3(preprocessed_out, OUTPUT_BUCKET, staged_pre_key)
        upload_to_s3(labeled_out,      OUTPUT_BUCKET, staged_lbl_key)
        wandb.log({"s3/chunk_staged": staged_lbl_key})

        _mark_chunk_done(orig_base, chunk_idx, staged_pre_key, staged_lbl_key)
        log.info("Chunk %d staged. Kiểm tra xem tất cả chunks đã xong chưa.", chunk_idx)

        meta = _load_chunkmeta(orig_base)
        if meta:
            total_chunks = meta["total_chunks"]
            done_chunks  = _get_done_chunks(orig_base, total_chunks)
            log.info("Done %d/%d chunks.", len(done_chunks), total_chunks)

            if len(done_chunks) == total_chunks:
                log.info("Tất cả chunks xong! Concat -> processed/ và labeled/")
                pre_key, lbl_key, pre_local, lbl_local = concat_chunks_and_upload(bucket, meta)
                wandb.log({"s3/preprocessed_key": pre_key, "s3/labeled_key": lbl_key})

                try:
                    try:
                        df_final_processed = pd.read_csv(pre_local)
                        df_final_labeled = pd.read_csv(lbl_local)
                            
                        if not os.path.exists(local_raw):
                            download_from_s3(bucket, meta["original_key"], local_raw)
                                
                        df_final_raw = pd.read_csv(local_raw) 

                        eval_table_all = create_wandb_evaluation_table(df_final_labeled)
                        run.log({"evaluation/absa_predictions_all": eval_table_all})

                        error_table_all = log_pipeline_errors_to_table(df_final_raw, df_final_processed, df_final_labeled)
                        run.log({"evaluation/pipeline_all_errors_all_chunks": error_table_all})

                        log_advanced_pipeline_plots(df_final_raw, df_final_labeled)

                        final_cols = ["product_id", "customer_name", "rating", "content", "created_at", "clean_content", "tokens"]
                            
                        df_final_processed = df_final_processed[[c for c in final_cols if c in df_final_processed.columns]]
                        df_final_labeled = df_final_labeled[[c for c in final_cols + ["label"] if c in df_final_labeled.columns]]
                            
                        df_final_processed.to_csv(pre_local, index=False) 
                        df_final_labeled.to_csv(lbl_local, index=False)
                            
                        log.info("Đã đồng bộ và hiển thị đầy đủ bảng dữ liệu, biểu đồ lỗi cho file lớn.")
                            
                    except Exception as e:
                        log.error("Lỗi khi cố gắng log các nội dung trực quan cho file lớn sau concat: %s", str(e))

                    try:
                        pre_artifact = wandb.Artifact(
                            name="preprocessed-reviews", type="dataset",
                            description="Cleaned + tokenized CSV (concat từ chunks)",
                            metadata={"rows": len(df_final_processed) if 'df_final_processed' in locals() else 0, "source_chunks": total_chunks},
                        )
                        pre_artifact.add_file(pre_local)
                        run.log_artifact(pre_artifact)

                        lbl_artifact = wandb.Artifact(
                            name="labeled-reviews", type="dataset",
                            description="ABSA labeled CSV (concat từ chunks)",
                            metadata={"source_chunks": total_chunks, "s3_key": lbl_key},
                        )
                        lbl_artifact.add_file(lbl_local)
                        run.log_artifact(lbl_artifact)

                        append_to_final(lbl_local, pre_local, os.path.basename(meta["original_key"]), run)
                    except Exception as e:
                        log.error("Lỗi trong quá trình push Artifact hoặc append file: %s", str(e))
                    finally:
                        run.finish()

                finally:
                    delete_all_checkpoints_for_base(orig_base)

                log.info("Pipeline hoàn tất (chunk concat) cho %s", key)
                return lbl_key
            else:
                pending = [i for i in range(total_chunks) if i not in done_chunks]
                log.info("Còn %d chunks chưa xong: %s.", len(pending), pending)

        run.finish()
        return staged_lbl_key
        
    try:
        for f in [local_raw, preprocessed_out, labeled_out, pre_local, lbl_local]:
            if f and os.path.exists(f):
                os.remove(f)
    except Exception:
        log.warning("Không thể xóa file tạm.")

# CLI fallback
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python pipeline.py <bucket> <s3-key>")
        sys.exit(1)
    run_pipeline(sys.argv[1], sys.argv[2])
