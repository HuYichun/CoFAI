import json
import platform
from pathlib import Path

import timm
import torch
from omegaconf import OmegaConf


def save_results(cfg, results):
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "ProTRAC",
        "config": OmegaConf.to_container(cfg, resolve=True),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__},
        "results": results,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


def infer(model, images, task, real_codec=False, **kwargs):
    if real_codec:
        coded = model.compress(images, tasks=[task], **kwargs)
        return model.decompress(coded, tasks=[task], **kwargs)[task]
    _, output = model.forward_test(images, tasks=[task], **kwargs)
    return output[task]
