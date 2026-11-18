"""Wrap the released inference weights as a starting point for `tiny_voca.train --init-from`.

The release (tiny-voca.pt) has the speech model only: no critics, optimizer or
mel head. train.py's --init-from wants a full training state, so this builds a
minimal one at step 0. The missing parts (mel head, critics) start fresh, which
is correct for a step-0 branch: critics are only used after --gan-start.

The phoneme embedding is grown from 76 English symbols to English + Urdu by
train.py's load_compatible (new rows start fresh). Run it through run.py so the
Urdu inventory is installed:

    python urdu/run.py make_init_checkpoint --weights weights/tiny-voca.pt --output runs/init.pt
"""

import argparse
import json
from pathlib import Path

import torch

from tiny_voca.config import CHECKPOINT_FORMAT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    released = torch.load(a.weights, map_location="cpu", weights_only=True)
    if released.get("format_version") != CHECKPOINT_FORMAT:
        raise SystemExit(f"unexpected checkpoint format {released.get('format_version')}")
    from tiny_voca import config as config_module   # the Urdu-extended Config (installed by run.py)
    target = config_module.Config().to_dict()
    if {**released["config"], "vocab_size": target["vocab_size"]} != target:
        raise SystemExit("released config differs from tiny_voca.config.Config() defaults")
    state = {
        "format_version": CHECKPOINT_FORMAT, "config": target,
        "frontend": released["frontend"], "annotation": released.get("annotation"),
        "step": 0, "datasets": [], "recipe": {"gan_start": 0, "note": "English tiny-voca, inference weights only"},
        "system": {"model." + k: v for k, v in released["model"].items()},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, a.output)
    print(json.dumps({"tensors": len(released["model"]), "vocab": f"76 -> {target['vocab_size']}", "output": str(a.output)}))


if __name__ == "__main__":
    main()
