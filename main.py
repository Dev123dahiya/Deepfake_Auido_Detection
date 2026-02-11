import argparse

from src.deepfake_audio_project.config import TrainingConfig
from src.deepfake_audio_project.inference import test_single_audio
from src.deepfake_audio_project.model_io import create_default_preprocessor, load_trained_model
from src.deepfake_audio_project.pipeline import main_training_pipeline


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

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

