from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


class OnnxEmbedder:
    """Minimal sentence-embedding wrapper around an ONNX-exported Transformer.

    This class runs the base Transformer in onnxruntime and applies mean pooling
    over the last hidden state using the attention mask, matching the common
    SentenceTransformers pooling behavior.
    """

    def __init__(
        self,
        *,
        model_dir: str,
        model_file: str | None = None,
        provider: str = "CPUExecutionProvider",
        max_length: int = 512,
        normalize_embeddings: bool = True,
    ):
        self.model_dir = str(model_dir)
        self.model_file = (model_file or "").strip()
        self.provider = str(provider or "CPUExecutionProvider")
        self.max_length = int(max_length)
        self.normalize_embeddings = bool(normalize_embeddings)

        # Lazy imports so the rest of the backend can run without ONNX deps.
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "onnxruntime is required for rag_embedding_backend=onnx"
            ) from exc

        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "transformers is required for rag_embedding_backend=onnx"
            ) from exc

        self._ort = ort
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        model_path = self._resolve_model_path()
        providers = [self.provider] if self.provider else None
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_names = {i.name for i in self.session.get_inputs()}

        logger.info("Initialized OnnxEmbedder model=%s provider=%s", model_path, self.provider)

    def _resolve_model_path(self) -> Path:
        base = Path(self.model_dir)
        if self.model_file:
            p = base / self.model_file
            if not p.exists():
                raise FileNotFoundError(f"ONNX model file not found: {p}")
            return p

        # Prefer int8 model if present.
        candidates = [
            base / "model.int8.onnx",
            base / "model.quant.onnx",
            base / "model.onnx",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"No ONNX model found in {base} (expected one of: model.int8.onnx, model.quant.onnx, model.onnx)"
        )

    def encode_query(self, texts: str | Sequence[str], **kwargs: Any):
        # Query/passage prompting is handled by PromptAwareEmbedder when desired.
        return self.encode(texts, **kwargs)

    def encode_passage(self, texts: str | Sequence[str], **kwargs: Any):
        return self.encode(texts, **kwargs)

    def encode(
        self,
        texts: str | Sequence[str],
        *,
        normalize_embeddings: bool | None = None,
        batch_size: int = 32,
        **_: Any,
    ):
        try:
            import numpy as np  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("numpy is required for rag_embedding_backend=onnx") from exc

        if isinstance(texts, str):
            is_single = True
            items = [texts]
        else:
            is_single = False
            items = list(texts)

        do_norm = self.normalize_embeddings if normalize_embeddings is None else bool(normalize_embeddings)

        embeddings: list["np.ndarray"] = []
        bs = max(1, int(batch_size))
        for start in range(0, len(items), bs):
            batch = items[start : start + bs]

            # Tokenize to numpy.
            tok = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )

            # Build feed dict based on ONNX model inputs.
            feed: dict[str, "np.ndarray"] = {}
            for name in self._input_names:
                if name in tok:
                    feed[name] = tok[name]

            outputs = self.session.run(None, feed)
            if not outputs:
                raise RuntimeError("ONNX session returned no outputs")

            last_hidden = outputs[0]  # (batch, seq, hidden)
            attention_mask = tok.get("attention_mask")
            if attention_mask is None:
                # No mask available; average across sequence dimension.
                pooled = last_hidden.mean(axis=1)
            else:
                mask = attention_mask.astype(np.float32)
                mask = np.expand_dims(mask, axis=-1)  # (batch, seq, 1)
                summed = (last_hidden * mask).sum(axis=1)
                denom = mask.sum(axis=1)
                denom = np.clip(denom, a_min=1e-9, a_max=None)
                pooled = summed / denom

            if do_norm:
                norm = np.linalg.norm(pooled, axis=1, keepdims=True)
                norm = np.clip(norm, a_min=1e-12, a_max=None)
                pooled = pooled / norm

            embeddings.append(pooled.astype(np.float32))

        out = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 0), dtype=np.float32)
        if is_single:
            return out[0]
        return out


__all__ = ["OnnxEmbedder"]

