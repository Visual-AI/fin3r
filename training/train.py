"""Train from the release configuration with one process per GPU."""
from pathlib import Path

from omegaconf import OmegaConf
from fin3r.trainer import Fin3RTrainer


if __name__ == "__main__":
    config = OmegaConf.load(Path(__file__).parent / "config" / "release.yaml")
    Fin3RTrainer(**config).run()
