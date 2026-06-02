from __future__ import annotations

from pathlib import Path

import torch

from src.models.encoder import ContrastiveDistilBertEncoder


def export_encoder_onnx(model: ContrastiveDistilBertEncoder, path: str | Path, max_length: int = 256) -> None:
    """Export the encoder graph for ONNX Runtime inference."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    input_ids = torch.ones((1, max_length), dtype=torch.long)
    attention_mask = torch.ones((1, max_length), dtype=torch.long)
    model.eval()
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output,
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
