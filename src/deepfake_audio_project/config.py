from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    dataset_path: str
    output_dir: str = "outputs"
    use_enhanced_features: bool = True
    max_files_per_class: int | None = 500
    epochs: int = 30
    batch_size: int = 32
    random_seed: int = 42

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

