import json
from pathlib import Path
R = Path(__file__).parent
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}
def wf(name): return code(f"%%writefile urdu/{name}\n" + (R / "urdu" / name).read_text())
cells = [
md("# tiny-voca -> Roman Urdu (1000 UrduTTS clips)\nKaggle: turn **Internet ON** and pick a **GPU** accelerator. Run top to bottom."),
code("STEPS = 6000      # fine-tune steps\nN_TRAIN = 1000\nN_VAL = 32"),
code("""%%bash
mkdir -p /kaggle/working/tiny-urdu && cd /kaggle/working/tiny-urdu
[ -d tiny-voca ] || git clone -q https://github.com/n4ze3m/tiny-voca
git -C tiny-voca checkout -q ca93ad1e35a834a9247347c6821cfeb4c82c6a96
pip install -q --no-deps -e tiny-voca
pip install -q librosa pyarrow "huggingface-hub>=0.34" "transformers>=4.40,<5" scipy datasets soundfile
hf download facebook/wav2vec2-lv-60-espeak-cv-ft > /dev/null
hf download n4ze3m/tiny-voca tiny-voca.pt --local-dir weights > /dev/null
echo ok"""),
code("%cd /kaggle/working/tiny-urdu\n!mkdir -p urdu runs"),
md("## Urdu front end + helper scripts"),
wf("frontend.py"), wf("run.py"), wf("prepare_data.py"), wf("make_init_checkpoint.py"),
md("## 1. Data: stream clips, resample to 24 kHz"),
code("!python urdu/prepare_data.py --train {N_TRAIN} --validation {N_VAL} --output data"),
md("## 2. Acoustic features"),
code("""!python urdu/run.py tiny_voca.prepare --inputs data/train.jsonl --split train --output features --speaker-column speaker
!python urdu/run.py tiny_voca.prepare --inputs data/validation.jsonl --split validation --output features --speaker-column speaker"""),
md("## 3. Forced alignment (phoneme durations)"),
code("!python urdu/run.py tiny_voca.annotate --features features --output annotations --splits train validation --dataset-revision UrduTTS-streamed --device cuda"),
md("## 4. Fine-tune from English tiny-voca"),
code("""!python urdu/make_init_checkpoint.py --weights weights/tiny-voca.pt --output runs/init.pt
!python urdu/run.py tiny_voca.train \\
  --train annotations/train.jsonl --validation annotations/validation.jsonl \\
  --output runs/urdu1 --init-from runs/init.pt --new-data --device cuda \\
  --steps {STEPS} --max-seconds 86400 --learning-rate 2e-4 --warmup 200 \\
  --gan-start {STEPS//2} --gan-ramp {STEPS//8} --teacher-hold {STEPS//6} --teacher-ramp {STEPS//3} --teacher-final 0 \\
  --save-every 500 --keep-every 2000 --log-every 25 --validation-size {N_VAL}"""),
md("## 5. Speak Roman Urdu"),
code("""import json, subprocess, IPython.display as ipd
voice = json.loads(open("data/train.jsonl").readline())["speaker"]
for i, t in enumerate(["aap kaise hain?", "mujhe chai pasand hai.", "shukriya, bohat shukriya!"]):
    subprocess.run(["python","urdu/run.py","tiny_voca.infer","--checkpoint","runs/urdu1/inference.pt",
                    "--voice",voice,"--text",t,"--output",f"out{i}.wav","--device","cuda"],check=True)
    print(t); ipd.display(ipd.Audio(f"out{i}.wav"))"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
(R / "tiny_urdu_kaggle.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
