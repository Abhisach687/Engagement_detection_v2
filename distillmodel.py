import argparse

from training.distill_tcn import distill


def main():
    parser = argparse.ArgumentParser(description="Distill ensemble teachers into MobileNetV2-TCN student.")
    parser.add_argument("--alpha", type=float, default=0.5, help="KD loss weight")
    parser.add_argument("--temperature", type=float, default=4.0, help="Softmax temperature")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--search", action="store_true", help="Grid search alpha/temperature and pick best Validation accuracy")
    args = parser.parse_args()

    distill(
        alpha=args.alpha,
        temperature=args.temperature,
        batch_size=args.batch_size,
        lr=args.lr,
        search=args.search,
    )


if __name__ == "__main__":
    main()
