import argparse

from evaluation.test_model import evaluate_xgb, evaluate_seq, evaluate_distilled


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on a specified split.")
    parser.add_argument("--model", choices=["xgb", "lstm", "bilstm", "tcn", "tcn_distilled", "all"], default="all")
    parser.add_argument("--split", choices=["Validation", "Test"], default="Validation")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    targets = ["xgb", "lstm", "bilstm", "tcn", "tcn_distilled"] if args.model == "all" else [args.model]
    for m in targets:
        if m == "xgb":
            evaluate_xgb(args.split)
        elif m in {"lstm", "bilstm", "tcn"}:
            evaluate_seq(m, split=args.split, batch_size=args.batch_size)
        else:
            evaluate_distilled(split=args.split, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
