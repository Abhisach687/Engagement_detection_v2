import argparse
import json
from pathlib import Path

import torch

from config import MODEL_DIR, IMG_SIZE, NUM_CLASSES, SEQ_LEN
from models.mobilenetv2_tcn import MobileNetV2_TCN
from utils.affect import AFFECT_COLUMNS, NUM_AFFECTS


MODEL_VARIANTS = {
    "engagement": {
        "stem": "mobilenetv2_tcn_distilled",
        "num_heads": 1,
    },
    "multiaffect": {
        "stem": "mobilenetv2_tcn_multiaffect_distilled",
        "num_heads": NUM_AFFECTS,
    },
}


def _onnx_meta_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_onnx_meta.json")


def _resolve_variant(variant: str) -> str:
    if variant != "auto":
        return variant
    if (MODEL_DIR / f"{MODEL_VARIANTS['multiaffect']['stem']}.ts").exists() or (
        MODEL_DIR / f"{MODEL_VARIANTS['multiaffect']['stem']}.pt"
    ).exists():
        return "multiaffect"
    return "engagement"


def load_source_model(device: torch.device, variant: str = "auto"):
    resolved_variant = _resolve_variant(variant)
    model_info = MODEL_VARIANTS[resolved_variant]
    ts_path = MODEL_DIR / f"{model_info['stem']}.ts"
    pt_path = MODEL_DIR / f"{model_info['stem']}.pt"

    if ts_path.exists():
        model = torch.jit.load(ts_path, map_location=device)
        return model.eval(), ts_path, resolved_variant

    if pt_path.exists():
        model = MobileNetV2_TCN(num_classes=NUM_CLASSES, num_heads=int(model_info["num_heads"]))
        state = torch.load(pt_path, map_location=device)
        model.load_state_dict(state)
        return model.to(device).eval(), pt_path, resolved_variant

    raise FileNotFoundError(
        f"No distilled model found. Expected either {ts_path.name} or {pt_path.name} in {MODEL_DIR}."
    )


def _write_onnx_metadata(output_path: Path, resolved_variant: str, source_path: Path):
    model_info = MODEL_VARIANTS[resolved_variant]
    head_names = ["Engagement"] if resolved_variant == "engagement" else list(AFFECT_COLUMNS)
    meta = {
        "variant": resolved_variant,
        "stem": model_info["stem"],
        "num_heads": int(model_info["num_heads"]),
        "head_names": head_names,
        "num_classes": NUM_CLASSES,
        "seq_len": SEQ_LEN,
        "img_size": IMG_SIZE,
        "source_path": str(source_path),
        "input_name": "frames",
        "output_name": "logits",
    }
    meta_path = _onnx_meta_path(output_path)
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)
    return meta_path


def export_onnx(output_path: Path | None = None, opset: int = 17, variant: str = "auto"):
    device = torch.device("cpu")
    model, source_path, resolved_variant = load_source_model(device, variant=variant)
    dummy = torch.randn(1, SEQ_LEN, 3, IMG_SIZE, IMG_SIZE, device=device)
    if output_path is None:
        output_path = MODEL_DIR / f"{MODEL_VARIANTS[resolved_variant]['stem']}.onnx"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": ["frames"],
        "output_names": ["logits"],
        "dynamic_axes": {"frames": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}},
        # PyTorch 2.10+ defaults to the dynamo exporter, which pulls in onnxscript.
        # The legacy exporter is sufficient for this model and works with only `onnx`.
        "dynamo": False,
    }
    try:
        torch.onnx.export(model, dummy, str(output_path), **export_kwargs)
    except TypeError:
        export_kwargs.pop("dynamo", None)
        torch.onnx.export(model, dummy, str(output_path), **export_kwargs)
    meta_path = _write_onnx_metadata(output_path, resolved_variant, source_path)
    print(f"Exported {resolved_variant} ONNX model from {source_path.name} to {output_path} ({meta_path.name})")


def main():
    parser = argparse.ArgumentParser(description="Export the distilled TCN model to ONNX.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ONNX file path",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--variant", choices=["auto", "engagement", "multiaffect"], default="auto")
    args = parser.parse_args()

    export_onnx(args.output, opset=args.opset, variant=args.variant)


if __name__ == "__main__":
    main()
