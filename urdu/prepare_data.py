"""Stream N UrduTTS clips, resample 44.1k -> 24k, write wavs + tiny-voca JSONL.

    python urdu/prepare_data.py --train 1000 --validation 32 --output data

Training text is "ipa:" + the dataset's `phonemes` column (see urdu/frontend.py
for why the vowel-less `roman` column is not used).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 24000
SOURCE_SR = 44100
REPO = "kaab4321/UrduTTS"


def stream(split, count, min_s, max_s, out_dir, speaker, prefix):
    from datasets import Audio, load_dataset
    ds = load_dataset(REPO, split=split, streaming=True).cast_column("audio", Audio(decode=False))
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for index, row in enumerate(ds):
        if len(rows) >= count:
            break
        phonemes = (row.get("phonemes") or "").strip()
        if not phonemes:
            skipped += 1
            continue
        wave, sr = sf.read(__import__("io").BytesIO(row["audio"]["bytes"]), dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        seconds = len(wave) / sr
        if not min_s <= seconds <= max_s:
            skipped += 1
            continue
        if sr != SR:
            g = np.gcd(sr, SR)
            wave = resample_poly(wave, SR // g, sr // g).astype(np.float32)
        peak = float(np.max(np.abs(wave)))
        if peak > 0.99:
            wave = wave * (0.99 / peak)
        name = f"{prefix}{len(rows):05d}"
        sf.write(wav_dir / f"{name}.wav", wave, SR, subtype="PCM_16")
        rows.append({"id": name, "text": "ipa:" + phonemes, "speaker": speaker, "audio": f"wavs/{name}.wav",
                     "origin": "kaab4321/UrduTTS", "source_id": f"{split}:{index}",
                     "urdu": row.get("transcription", "").strip(), "seconds": round(seconds, 2)})
        if len(rows) % 100 == 0:
            print(f"[{split}] {len(rows)}/{count} (skipped {skipped})", flush=True)
    return rows


def median_f0(out_dir, rows, n=20):
    import librosa
    f0s = []
    for row in rows[:n]:
        wave, _ = sf.read(out_dir / row["audio"], dtype="float32")
        f0 = librosa.yin(wave, fmin=60, fmax=500, sr=SR, frame_length=2048)
        rms = librosa.feature.rms(y=wave, frame_length=2048, hop_length=512)[0][:len(f0)]
        f0s.extend(f0[:len(rms)][rms > np.percentile(rms, 60)])
    return float(np.median(f0s))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=int, default=1000)
    p.add_argument("--validation", type=int, default=32)
    p.add_argument("--output", type=Path, default=Path("data"))
    p.add_argument("--min-seconds", type=float, default=1.5)
    p.add_argument("--max-seconds", type=float, default=12.0, help="shorter clips = cheaper CPU training")
    p.add_argument("--speaker", choices=("auto", "F2", "M2"), default="auto",
                   help="which tiny-voca voice slot the Urdu speaker is written into (auto = by median pitch)")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    train = stream("train", a.train, a.min_seconds, a.max_seconds, a.output, "F2", "tr")
    speaker = a.speaker
    if speaker == "auto":
        f0 = median_f0(a.output, train)
        speaker = "F2" if f0 > 165 else "M2"
        print(f"median f0 {f0:.0f} Hz -> voice slot {speaker}")
    val = stream("test", a.validation, a.min_seconds, a.max_seconds, a.output, speaker, "va")
    for row in train:
        row["speaker"] = speaker
    for name, rows in (("train", train), ("validation", val)):
        with (a.output / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    hours = sum(r["seconds"] for r in train) / 3600
    print(json.dumps({"train": len(train), "validation": len(val), "train_hours": round(hours, 2), "speaker": speaker}))


if __name__ == "__main__":
    main()
