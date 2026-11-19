import json
from pathlib import Path
R = Path(__file__).parent
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}
def wf(name): return code(f"%%writefile urdu/{name}\n" + (R / "urdu" / name).read_text())
cells = [
md("# tiny-voca -> Roman Urdu (1000 UrduTTS clips)\nKaggle: turn **Internet ON** and pick a **GPU** accelerator. Run top to bottom."),
code("import torch\nassert torch.cuda.is_available(), 'No GPU: set Settings -> Accelerator -> GPU (before Save Version too)'\n")
,code("STEPS = 25000     # fine-tune steps (~3 h on a T4)\nN_TRAIN = 5000    # ~8 h of audio; whole run ~8-8.5 h\nN_VAL = 64"),
code("""%%bash
set -e
mkdir -p /kaggle/working/tiny-urdu && cd /kaggle/working/tiny-urdu
[ -d tiny-voca ] || git clone -q https://github.com/n4ze3m/tiny-voca
git -C tiny-voca checkout -q ca93ad1e35a834a9247347c6821cfeb4c82c6a96
pip install -q --no-deps -e tiny-voca
# one consistent set: transformers<5 needs huggingface-hub<1, so datasets must be <4
pip install -q "huggingface-hub>=0.34,<1" "transformers>=4.40,<5" "datasets<4" librosa pyarrow scipy soundfile 2>&1 | grep -v -i -E "dependency resolver|requires|^$" || true
python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download("facebook/wav2vec2-lv-60-espeak-cv-ft", allow_patterns=["*.json", "pytorch_model.bin", "*.txt"])
hf_hub_download("n4ze3m/tiny-voca", "tiny-voca.pt", local_dir="weights")
import datasets, transformers, tiny_voca; print("imports ok", datasets.__version__, transformers.__version__)
PY"""),
code("%cd /kaggle/working/tiny-urdu\n!mkdir -p urdu runs"),
md("## Urdu front end + helper scripts"),
wf("frontend.py"), wf("run.py"), wf("prepare_data.py"), wf("make_init_checkpoint.py"),
md("## 1. Data: stream clips, resample to 24 kHz (skipped if already done)"),
code("""import os
def sh(cmd):
    import subprocess; subprocess.run(cmd, shell=True, check=True)
if not os.path.exists("data/validation.jsonl"):
    sh(f"python urdu/prepare_data.py --train {N_TRAIN} --validation {N_VAL} --output data")
else: print("data cached")"""),
md("## 2. Acoustic features (skipped if already done)"),
code("""import json
for split in ("train", "validation"):
    meta = f"features/{split}.preparation.json"
    if os.path.exists(meta) and json.load(open(meta)).get("complete"):
        print(split, "features cached"); continue
    resume = "--resume" if os.path.exists(meta) else ""   # finish a half-done run instead of restarting
    sh(f"python urdu/run.py tiny_voca.prepare --inputs data/{split}.jsonl --split {split} --output features --speaker-column speaker {resume}")"""),
md("""## 3. Forced alignment (durations are cached per clip; re-running only re-applies the quality gate)
The default gate rejected 998/1000 real-speech clips (it is tuned for synthetic English), so the thresholds are relaxed here. Check the printout: `accepted` should be close to 1000."""),
code("""import json
GATE = "--max-weak-phone-fraction 0.8 --min-token-probability 0.1 --min-vowel-voiced 0.2 --max-phone-frames 120 --min-phones-per-second 3 --max-phones-per-second 35"
resume = "--resume" if os.path.exists("annotations-v2/annotations.json") else ""
sh(f"python urdu/run.py tiny_voca.annotate --features features --output annotations-v2 --splits train validation "
   f"--dataset-revision UrduTTS-streamed --device cuda {GATE} {resume}")
# annotate only marks a run 'complete' when it also saw a 'test' split; we don't need one.
meta = json.load(open("annotations-v2/annotations.json"))
counts = meta.get("counts") or {}
summary = json.load(open("annotations-v2/summary.json"))
for k, v in summary["splits"].items():
    print(k, "accepted", v["accepted"], "rejected", v["rejected"], v["rejection_reasons"])
assert summary["splits"]["train"]["accepted"] >= 200, "too few clips accepted; see qc_quantiles in annotations-v2/summary.json"
meta["complete"] = True
json.dump(meta, open("annotations-v2/annotations.json", "w"))"""),
md("## 4. Fine-tune from English tiny-voca (resumes from runs/urdu2/last.pt if it exists)"),
code("""if not os.path.exists("runs/init-v2.pt"):
    sh("python urdu/run.py make_init_checkpoint --weights weights/tiny-voca.pt --output runs/init-v2.pt")
start = "--resume runs/urdu2/last.pt" if os.path.exists("runs/urdu2/last.pt") else "--init-from runs/init-v2.pt --new-data"
sh(f"python urdu/run.py tiny_voca.train --train annotations-v2/train.jsonl --validation annotations-v2/validation.jsonl "
   f"--output runs/urdu2 {start} --device cuda --steps {STEPS} --max-seconds 86400 --learning-rate 2e-4 --warmup 200 "
   f"--gan-start {STEPS//2} --gan-ramp {STEPS//8} --teacher-hold {STEPS//6} --teacher-ramp {STEPS//3} --teacher-final 0 "
   f"--save-every 1000 --keep-every 5000 --log-every 25 --validation-size {N_VAL}")"""),
md("## 5. Speak Roman Urdu"),
code("""import json, subprocess, IPython.display as ipd
voice = json.loads(open("data/train.jsonl").readline())["speaker"]
for i, t in enumerate(["aap kaise hain?", "mujhe chai pasand hai.", "shukriya, bohat shukriya!"]):
    subprocess.run(["python","urdu/run.py","tiny_voca.infer","--checkpoint","runs/urdu2/inference.pt",
                    "--voice",voice,"--text",t,"--output",f"out{i}.wav","--device","cuda"],check=True)
    print(t); ipd.display(ipd.Audio(f"out{i}.wav"))"""),
md("""## 6. Export for the browser (ONNX)
Download `/kaggle/working/web/model.onnx` and `model.json` from the Output tab and put them in `web/public/` of the Next.js app.
**New session, no retraining:** Add Input -> Your Work -> this notebook's output, then run only the setup, `%cd`, `%%writefile` cells and this one."""),
code("""import glob, os, shutil, subprocess
subprocess.run("pip install -q onnx", shell=True)
ckpt = next(p for p in ["runs/urdu2/inference.pt", *glob.glob("/kaggle/input/**/runs/urdu2/inference.pt", recursive=True)] if os.path.exists(p))
print("checkpoint:", ckpt)
os.makedirs("/kaggle/working/web", exist_ok=True)
subprocess.run(["python","urdu/run.py","tiny_voca.export","--checkpoint",ckpt,"--output","/kaggle/working/web/model.onnx"], check=True)
for p in glob.glob("/kaggle/working/web/*"): print(p, round(os.path.getsize(p)/1e6, 2), "MB")"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
(R / "tiny_urdu_kaggle.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
