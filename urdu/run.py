"""Run a tiny_voca module with the Urdu front end installed.

    python urdu/run.py tiny_voca.annotate --features ...
    python urdu/run.py tiny_voca.train ...
    python urdu/run.py tiny_voca.infer --checkpoint ... --text "aap kaise hain?"

The patch has to happen before the target module is imported, because
tiny-voca modules do ``from .text import phonemize``.
"""

import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import frontend  # noqa: E402

frontend.install()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    module = sys.argv[1]
    sys.argv = [module] + sys.argv[2:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
