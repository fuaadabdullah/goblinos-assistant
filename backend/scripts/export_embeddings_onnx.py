#!/usr/bin/env python3
"""
Export a SentenceTransformers embedding model to ONNX (feature-extraction) and optionally int8-quantize.

The exported ONNX model returns `last_hidden_state`; sentence embeddings are produced by
mean-pooling in the runtime (see services/rag/onnx_embedder.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-id",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF model id or local path (SentenceTransformers model recommended).",
    )
    p.add_argument(
        "--out-dir",
        default="apps/goblin-assistant-root/backend/models/onnx/all-MiniLM-L6-v2",
        help="Output directory for ONNX model + tokenizer files.",
    )
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    p.add_argument(
        "--quantize-int8",
        action="store_true",
        help="Also write model.int8.onnx using onnxruntime dynamic quantization.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    model_id = str(args.model_id)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Missing deps. Install sentence-transformers and torch to export."
        ) from exc

    st = SentenceTransformer(model_id)
    transformer = st._first_module()  # sentence_transformers.models.Transformer
    hf_model = getattr(transformer, "auto_model", None) or getattr(transformer, "model", None)
    tokenizer = getattr(transformer, "tokenizer", None) or getattr(st, "tokenizer", None)
    if hf_model is None or tokenizer is None:
        raise SystemExit("Could not locate HF model/tokenizer from SentenceTransformer")

    hf_model.eval()

    # Dummy inputs
    dummy = tokenizer(
        "Export to ONNX.",
        padding="max_length",
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )

    has_token_type = "token_type_ids" in dummy

    if has_token_type:
        input_names = ["input_ids", "attention_mask", "token_type_ids"]

        class _Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask, token_type_ids):
                out = self.m(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    return_dict=True,
                )
                return out.last_hidden_state

        wrapper = _Wrapper(hf_model)
        inputs = (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"])
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        }
    else:
        input_names = ["input_ids", "attention_mask"]

        class _Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask):
                out = self.m(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                return out.last_hidden_state

        wrapper = _Wrapper(hf_model)
        inputs = (dummy["input_ids"], dummy["attention_mask"])
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        }

    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        inputs,
        str(onnx_path),
        input_names=input_names,
        output_names=["last_hidden_state"],
        dynamic_axes=dynamic_axes,
        opset_version=int(args.opset),
    )

    # Save tokenizer/config so runtimes can load from the exported directory.
    tokenizer.save_pretrained(out_dir)
    try:
        hf_model.config.save_pretrained(out_dir)
    except Exception:
        pass

    meta = {
        "source_model": model_id,
        "onnx_file": "model.onnx",
        "pooling": "mean",
        "normalize": True,
        "inputs": input_names,
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.quantize_int8:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                "onnxruntime is required for --quantize-int8 (pip install onnxruntime)"
            ) from exc

        int8_path = out_dir / "model.int8.onnx"
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )

    print(f"Wrote: {onnx_path}")
    if args.quantize_int8:
        print(f"Wrote: {out_dir / 'model.int8.onnx'}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
