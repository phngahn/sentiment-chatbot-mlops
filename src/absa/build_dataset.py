import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = BASE_DIR / "data/processed/labeled_processed_products_reviews.csv"
OUTPUT_DIR = BASE_DIR / "data/absa"

ASPECT_MAPPING = {
    "MÔ TẢ": "description",
    "CHẤT LƯỢNG": "quality",
    "ĐÓNG GÓI": "packing",
    "GIAO HÀNG": "delivery",
    "DỊCH VỤ": "service",
    "GIÁ CẢ": "price"
}

def convert_labels(label_str):
    labels = json.loads(label_str)

    row = {
        "description": "none",
        "quality": "none",
        "packing": "none",
        "delivery": "none",
        "service": "none",
        "price": "none"
    }

    for item in labels:
        aspect = item["aspect"]
        sentiment = item["sentiment"]

        mapped_aspect = ASPECT_MAPPING.get(aspect)

        if mapped_aspect:
            row[mapped_aspect] = sentiment

    return row

def build_dataset(df):
    records = []

    for _, row in df.iterrows():
        label_data = convert_labels(row["label"])

        sample = {
            "text": row["clean_content"],
            **label_data
        }

        records.append(sample)

    return pd.DataFrame(records)

def split_dataset(dataset):
    train_df, temp_df = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42
    )

    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42
    )

    return train_df, dev_df, test_df

def save_dataset(train_df, dev_df, test_df, full_df, version):
    version_dir = OUTPUT_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_json(
        version_dir / "train.json",
        orient="records",
        force_ascii=False,
        indent=4
    )

    dev_df.to_json(
        version_dir / "dev.json",
        orient="records",
        force_ascii=False,
        indent=4
    )

    test_df.to_json(
        version_dir / "test.json",
        orient="records",
        force_ascii=False,
        indent=4
    )

    full_df.to_json(
        version_dir / "full_dataset.json",
        orient="records",
        force_ascii=False,
        indent=4
    )
    
def update_dataset(old_version, new_version, new_data_path):
    old_dataset_path = (
        OUTPUT_DIR /
        old_version /
        "full_dataset.json"
    )

    old_df = pd.read_json(old_dataset_path)

    new_reviews_df = pd.read_csv(new_data_path)

    new_dataset = build_dataset(new_reviews_df)

    full_dataset = pd.concat(
        [old_df, new_dataset],
        ignore_index=True
    )

    train_df, dev_df, test_df = split_dataset(
        full_dataset
    )

    save_dataset(
        train_df,
        dev_df,
        test_df,
        full_dataset,
        new_version
    )

    print(
        f"Updated dataset: {old_version} -> {new_version}"
    )

def main(version):
    df = pd.read_csv(INPUT_PATH)

    dataset = build_dataset(df)

    train_df, dev_df, test_df = split_dataset(dataset)

    save_dataset(
        train_df,
        dev_df,
        test_df,
        dataset,
        version
    )

if __name__ == "__main__":
    update_dataset("v1", "v2", BASE_DIR / "data/processed/labeled_processed_reviews_20260508.csv")