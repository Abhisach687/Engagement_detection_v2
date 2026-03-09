import argparse
from pathlib import Path

import torch

from config import MODEL_DIR, IMG_SIZE, NUM_CLASSES, SEQ_LEN
from models.mobilenetv2_tcn import MobileNetV2_TCN


def load_source_model(device: torch.device):
    ts_path = MODEL_DIR / "mobilenetv2_tcn_distilled.ts"
    pt_path = MODEL_DIR / "mobilenetv2_tcn_distilled.pt"

    if ts_path.exists():
        model = torch.jit.load(ts_path, map_location=device)
        return model.eval(), ts_path

    if pt_path.exists():
        model = MobileNetV2_TCN(num_classes=NUM_CLASSES)
        state = torch.load(pt_path, map_location=device)
        model.load_state_dict(state)
        return model.to(device).eval(), pt_path

    raise FileNotFoundError(
        f"No distilled model found. Expected either {ts_path.name} or {pt_path.name} in {MODEL_DIR}."
    )


def export_onnx(output_path: Path, opset: int = 17):
    device = torch.device("cpu")
    model, source_path = load_source_model(device)
    dummy = torch.randn(1, SEQ_LEN, 3, IMG_SIZE, IMG_SIZE, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["frames"],
        output_names=["logits"],
        dynamic_axes={"frames": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Exported ONNX model from {source_path.name} to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export the distilled TCN model to ONNX.")
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_DIR / "mobilenetv2_tcn_distilled.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    export_onnx(args.output, opset=args.opset)


if __name__ == "__main__":
    main()
