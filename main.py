import argparse
import json

from src.deepfake_audio_project.config import TrainingConfig
from src.deepfake_audio_project.evaluation import evaluate_robustness_on_dataset, test_on_test_set
from src.deepfake_audio_project.inference import secure_predict_single_audio, test_single_audio
from src.deepfake_audio_project.llm_insights import generate_llm_review
from src.deepfake_audio_project.model_io import (
    calculate_model_checksum,
    create_default_preprocessor,
    load_trained_model,
    verify_model_checksum,
)
from src.deepfake_audio_project.pipeline import main_training_pipeline
from src.deepfake_audio_project.reporting import generate_testing_report


def run_train(args):
    config = TrainingConfig(
        dataset_path=args.dataset,
        output_dir=args.output,
        use_enhanced_features=not args.basic,
        max_files_per_class=args.max_files,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    main_training_pipeline(config)


def run_predict(args):
    model = load_trained_model(args.model)
    preprocessor = create_default_preprocessor()
    result = test_single_audio(model, preprocessor, args.audio, use_enhanced=not args.basic)
    print(result)


def run_evaluate(args):
    model = load_trained_model(args.model)
    preprocessor = create_default_preprocessor()
    metrics = test_on_test_set(
        model,
        preprocessor,
        args.dataset,
        use_enhanced=not args.basic,
        show_plots=not args.no_plots,
    )
    if args.robustness_check:
        metrics["robustness"] = evaluate_robustness_on_dataset(
            model,
            preprocessor,
            args.dataset,
            use_enhanced=not args.basic,
            max_samples=args.max_robustness_samples,
        )
    generate_testing_report(metrics, save_dir=args.output)
    if args.llm_insights:
        generate_llm_review(metrics, save_dir=args.output, model_name=args.llm_model)


def run_secure_predict(args):
    model = load_trained_model(args.model)
    preprocessor = create_default_preprocessor()

    if args.expected_sha256:
        ok, actual = verify_model_checksum(args.model, args.expected_sha256)
        if not ok:
            raise ValueError(f"Model checksum mismatch. expected={args.expected_sha256} actual={actual}")

    result = secure_predict_single_audio(
        model,
        preprocessor,
        args.audio,
        use_enhanced=not args.basic,
        ood_confidence_threshold=args.ood_confidence_threshold,
        ood_entropy_threshold=args.ood_entropy_threshold,
        low_risk_threshold=args.low_risk_threshold,
        high_risk_threshold=args.high_risk_threshold,
        audit_log_path=args.audit_log,
    )
    print(json.dumps(result, indent=2))


def run_checksum(args):
    checksum = calculate_model_checksum(args.model)
    print(f"sha256: {checksum}")


def build_parser():
    parser = argparse.ArgumentParser(description="Deepfake audio project runner")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Train model")
    train_parser.add_argument("--dataset", required=True, help="Dataset folder with real/ and fake/")
    train_parser.add_argument("--output", default="outputs", help="Output directory")
    train_parser.add_argument("--max-files", type=int, default=500, help="Max files per class")
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--basic", action="store_true", help="Use basic features only")
    train_parser.set_defaults(func=run_train)

    predict_parser = sub.add_parser("predict", help="Predict one audio file")
    predict_parser.add_argument("--model", required=True, help="Path to trained .h5 model")
    predict_parser.add_argument("--audio", required=True, help="Path to audio file")
    predict_parser.add_argument("--basic", action="store_true", help="Use basic features only")
    predict_parser.set_defaults(func=run_predict)

    eval_parser = sub.add_parser("evaluate", help="Run advanced test-set evaluation")
    eval_parser.add_argument("--model", required=True, help="Path to trained .h5 model")
    eval_parser.add_argument("--dataset", required=True, help="Dataset folder with real/ and fake/")
    eval_parser.add_argument("--output", default="outputs", help="Output directory for reports")
    eval_parser.add_argument("--basic", action="store_true", help="Use basic features only")
    eval_parser.add_argument("--no-plots", action="store_true", help="Disable confusion matrix/curve plots")
    eval_parser.add_argument("--llm-insights", action="store_true", help="Generate optional LLM research review")
    eval_parser.add_argument("--llm-model", default="gpt-4o-mini", help="Model name for LLM insights")
    eval_parser.add_argument("--robustness-check", action="store_true", help="Run robustness tests on perturbations")
    eval_parser.add_argument("--max-robustness-samples", type=int, default=200, help="Max samples for robustness test")
    eval_parser.set_defaults(func=run_evaluate)

    secure_parser = sub.add_parser("secure-predict", help="Run security-aware prediction with OOD/risk/audit")
    secure_parser.add_argument("--model", required=True, help="Path to trained .h5 model")
    secure_parser.add_argument("--audio", required=True, help="Path to audio file")
    secure_parser.add_argument("--basic", action="store_true", help="Use basic features only")
    secure_parser.add_argument("--expected-sha256", help="Optional expected SHA256 checksum for model verification")
    secure_parser.add_argument("--ood-confidence-threshold", type=float, default=0.60)
    secure_parser.add_argument("--ood-entropy-threshold", type=float, default=0.68)
    secure_parser.add_argument("--low-risk-threshold", type=float, default=0.30)
    secure_parser.add_argument("--high-risk-threshold", type=float, default=0.75)
    secure_parser.add_argument("--audit-log", default="outputs/security_audit.jsonl", help="Audit log path")
    secure_parser.set_defaults(func=run_secure_predict)

    checksum_parser = sub.add_parser("checksum", help="Calculate model SHA256 checksum")
    checksum_parser.add_argument("--model", required=True, help="Path to trained .h5 model")
    checksum_parser.set_defaults(func=run_checksum)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
