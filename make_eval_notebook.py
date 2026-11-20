"""Builds tiny_urdu_eval.ipynb: compares the 1k and 5k models by word error rate (WER)."""
import json
from pathlib import Path
R = Path(__file__).parent
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}

cells = [
md("""# tiny-urdu: compare the 1k and 5k models (WER)
Each model speaks the same unseen UrduTTS sentences. Whisper (speech-to-text) writes down what it hears, and we count the wrong words.
**Lower is better.** The real recordings give the ceiling: even perfect speech is not scored 0 by Whisper.

Setup on Kaggle:
1. **New Dataset** -> upload `models/1k` and `models/5k` (each with `model.onnx` and `model.json`) -> name it `tiny-urdu-models`.
2. In this notebook: **Add Input** -> that dataset. Turn **Internet ON** and pick a **GPU**.
3. Run all cells."""),
code("!pip install -q \"huggingface-hub>=0.34,<1\" \"transformers>=4.40,<5\" \"datasets<4\" jiwer onnxruntime librosa soundfile 2>&1 | grep -v -i -E 'dependency resolver|requires|^$' || true\n!mkdir -p urdu"),
code("N = 150        # test sentences\nSKIP = 200     # skip the first test rows: some were used for validation during training\nWHISPER = 'openai/whisper-large-v3'"),
md("## Urdu front end (same file as training)"),
code("%%writefile urdu/frontend.py\n" + (R / "urdu" / "frontend.py").read_text()),
md("## Find the two models"),
code("""import glob, json, os
def find(tag):
    hits = [p for p in glob.glob("/kaggle/input/**/model.onnx", recursive=True) if f"/{tag}/" in p]
    assert hits, f"no {tag}/model.onnx under /kaggle/input: add the dataset as Input"
    return hits[0]
MODELS = {"1k": find("1k"), "5k": find("5k")}
for k, v in MODELS.items(): print(k, v, json.load(open(v.replace(".onnx", ".json")))["step"], "steps")"""),
md("## Unseen test sentences"),
code("""import io, itertools, numpy as np, soundfile as sf, librosa
from datasets import load_dataset, Audio
ds = load_dataset("kaab4321/UrduTTS", split="test", streaming=True).cast_column("audio", Audio(decode=False))
rows = []
for r in itertools.islice(ds, SKIP, None):
    wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
    wav = wav.mean(1) if wav.ndim > 1 else wav
    secs = len(wav) / sr
    if 1.5 <= secs <= 12 and r["phonemes"].strip() and r["transcription"].strip():
        rows.append({"ipa": r["phonemes"], "ref": r["transcription"], "real": librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)})
    if len(rows) == N: break
print(len(rows), "sentences")"""),
md("## Each model speaks the sentences (on CPU, from the exported ONNX files)"),
code("""import sys, onnxruntime as ort
sys.path.insert(0, "urdu"); import frontend

def speak_all(onnx_path):
    meta = json.load(open(onnx_path.replace(".onnx", ".json")))
    sym = meta["symbols"]; ids = {s: i for i, s in enumerate(sym)}
    special = {"<pad>", "<bos>", "<eos>", "<sp>", "<unk>", ",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…"}
    frontend._inventory = frozenset(s for s in sym if s not in special)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    voice = meta["voices"].index("M2")
    out = []
    for r in rows:
        words = frontend.pieces("ipa:" + r["ipa"])[0][1]
        toks = ["<bos>"]
        for w in words:
            if toks[-1] not in ("<bos>", "<sp>"): toks.append("<sp>")
            toks += [p for p, _ in w]
        toks.append("<eos>")
        t = [ids.get(s, ids["<unk>"]) for s in toks]
        if len(t) > meta["max_tokens"] - 12: out.append(None); continue
        arr = lambda v: np.array([v], dtype=np.int64)
        audio = sess.run(None, {"tokens": arr(t), "stress": np.zeros((1, len(t)), np.int64), "voice": np.array([voice], np.int64),
                                "speed": np.array([1.0], np.float32), "pitch_shift": np.array([0.0], np.float32),
                                "pitch_range": np.array([1.0], np.float32)})[0][0]
        peak = np.abs(audio).max()
        if peak > 0.99: audio = audio * 0.99 / peak
        out.append(librosa.resample(audio.astype("float32"), orig_sr=meta["sample_rate"], target_sr=16000))
    return out
AUDIO = {k: speak_all(v) for k, v in MODELS.items()}
print({k: sum(a is not None for a in v) for k, v in AUDIO.items()}, "spoken")"""),
md("## Whisper listens and writes it down"),
code("""import torch
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model=WHISPER, torch_dtype=torch.float16, device=0)
gen = {"language": "urdu", "task": "transcribe"}
def transcribe(clips):
    res = asr([{"raw": c, "sampling_rate": 16000} for c in clips], batch_size=8, generate_kwargs=gen)
    return [x["text"] for x in res]
keep = [i for i in range(len(rows)) if all(AUDIO[k][i] is not None for k in AUDIO)]
HEARD = {"real recordings": transcribe([rows[i]["real"] for i in keep])}
for k in AUDIO: HEARD[f"{k} model"] = transcribe([AUDIO[k][i] for i in keep])
REFS = [rows[i]["ref"] for i in keep]
print(len(keep), "sentences scored")"""),
md("## Word error rate"),
code("""import re, jiwer
def norm(s):
    s = re.sub(r"[\\u064B-\\u065F\\u0670\\u0640]", "", s)                       # diacritics, tatweel
    s = s.translate(str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ہ", "ة": "ہ", "ه": "ہ", "ۃ": "ہ"}))
    s = re.sub(r"[^\\w\\s]|_", " ", s)                                          # punctuation (including ، ۔ ؟)
    return re.sub(r"\\s+", " ", s).strip()
refs = [norm(r) for r in REFS]
print(f"{'':18} {'WER %':>8} {'CER %':>8}")
for name, heard in HEARD.items():
    h = [norm(x) for x in heard]
    print(f"{name:18} {100*jiwer.wer(refs, h):8.1f} {100*jiwer.cer(refs, h):8.1f}")
# sentence by sentence: which model is closer?
e = {n: [jiwer.wer(r, norm(x)) for r, x in zip(refs, h)] for n, h in HEARD.items()}
better = sum(a < b for a, b in zip(e["5k model"], e["1k model"])); worse = sum(a > b for a, b in zip(e["5k model"], e["1k model"]))
print(f"\\n5k better on {better} sentences, worse on {worse}, same on {len(refs)-better-worse}")"""),
md("## Listen to a few (1k vs 5k)"),
code("""import IPython.display as ipd
for j in range(5):
    i = keep[j]; print(REFS[j])
    for k in AUDIO:
        print(" ", k, "->", HEARD[f"{k} model"][j]); ipd.display(ipd.Audio(AUDIO[k][i], rate=16000))"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
(R / "tiny_urdu_eval.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
