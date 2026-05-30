"""
Convert PhoBERT ABSA model to ONNX format.
Run: docker exec tiki-airflow-worker python scripts/convert_phobert_onnx.py
"""
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "absa" / "v2" / "phobert"
ONNX_DIR = BASE_DIR / "models" / "absa" / "v2" / "phobert_onnx"
MODEL_NAME = "vinai/phobert-base"
ASPECTS = ["description", "quality", "packaging", "delivery", "service", "price"]


class MultiHeadPhoBERT(nn.Module):
    def __init__(self, model_name, hidden_size=768, num_classes=3, num_aspects=6, dropout=0.3):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, num_classes) for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask):
        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = self.dropout(out.last_hidden_state[:, 0])
        # Stack all head outputs into single tensor [batch, num_aspects, num_classes]
        logits = torch.stack([head(cls_out) for head in self.heads], dim=1)
        return logits


def main():
    print("Loading PyTorch model...")
    device = torch.device("cpu")
    checkpoint = torch.load(MODEL_DIR / "phobert.pt", map_location=device)

    model = MultiHeadPhoBERT(MODEL_NAME)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    # Create dummy input
    dummy_text = ["sản phẩm chất lượng tốt"]
    enc = tokenizer(
        dummy_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Export
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = ONNX_DIR / "phobert_absa.onnx"

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Save tokenizer alongside
    tokenizer.save_pretrained(str(ONNX_DIR))

    # Verify
    print("Verifying ONNX model...")
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path))
    outputs = session.run(None, {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    })
    logits_onnx = outputs[0]
    print(f"ONNX output shape: {logits_onnx.shape}")  # [1, 6, 3]

    # Compare with PyTorch
    with torch.no_grad():
        logits_pt = model(input_ids, attention_mask).numpy()

    diff = abs(logits_pt - logits_onnx).max()
    print(f"Max diff PyTorch vs ONNX: {diff:.6f}")
    if diff < 0.01:
        print("PASS — ONNX model matches PyTorch")
    else:
        print("WARNING — outputs differ significantly")

    # File size
    pt_size = (MODEL_DIR / "phobert.pt").stat().st_size / 1024 / 1024
    onnx_size = onnx_path.stat().st_size / 1024 / 1024
    print(f"\nPyTorch: {pt_size:.1f}MB → ONNX: {onnx_size:.1f}MB")
    print("Done!")


if __name__ == "__main__":
    main()