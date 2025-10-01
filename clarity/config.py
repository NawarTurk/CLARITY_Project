from dataclasses import dataclass, field
from pathlib import Path


def project_root() -> Path:
    """Return repository root (one level above the `clarity` package)."""
    return Path(__file__).resolve().parents[1]


@dataclass
class Config:
    model_name: str = "qwen-0.5b"
    template_name: str = "prompt_t1_01_base-template.txt"
    max_new_tokens: int = 100
    do_sample: bool = False
    checkpoint_interval: int = 10
    batch_size: int = 1
    sleep_sec: float = 0.0
    seed: int = 42
    root: Path = field(default_factory=project_root)
    prompts_dir: Path = field(init=False)
    test_csv: Path = field(init=False)
    out_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.prompts_dir = self.root / "prompts"
        self.test_csv = self.root / "datasets" / "test_dataset.csv"
        self.out_dir = self.root / "results" / "predictions"

    def output_csv(self) -> Path:
        template_slug = Path(self.template_name).stem
        return self.out_dir / f"test_{self.model_name}_{template_slug}.csv"
