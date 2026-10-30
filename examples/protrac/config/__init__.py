from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
OmegaConf.register_new_resolver("protrac_root", lambda: str(ROOT), replace=True)


def load_config(path):
    return OmegaConf.load(path)
