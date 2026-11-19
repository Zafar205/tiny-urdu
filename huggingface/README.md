---
license: apache-2.0
language:
- ur
base_model: n4ze3m/tiny-voca
datasets:
- kaab4321/UrduTTS
pipeline_tag: text-to-speech
library_name: onnx
tags:
- text-to-speech
- tts
- urdu
- urdu-tts
- roman-urdu
- speech-synthesis
- small-model
- pakistan
- onnx
- tiny
- on-device
- browser
---

# tiny-urdu-tts: Small Urdu Text-to-Speech (TTS) Model 🇵🇰🎙️

**tiny-urdu-tts** is a small, open-source **Urdu text-to-speech (TTS)** model for **Roman Urdu**. It has only **4.5 million parameters (21 MB)** and runs on a CPU or right in your web browser, with no GPU, server or API key.

اردو ٹیکسٹ ٹو اسپیچ: رومن اردو لکھیں اور سنیں

**4.5M parameters · 21 MB · Urdu TTS · runs on CPU / in the browser**

Type `aap kaise hain?` and it speaks it in Urdu.

- **Try it:** https://web-hazel-seven-89.vercel.app/
- **Code:** https://github.com/Zafar205/tiny-urdu

## Audio samples

| Text | Audio |
|---|---|
| aap kaise hain? | <audio controls src="https://huggingface.co/zafar205/tiny-urdu-tts-4.5-million-parameters/resolve/main/sample1.wav"></audio> |
| mujhe chai pasand hai. | <audio controls src="https://huggingface.co/zafar205/tiny-urdu-tts-4.5-million-parameters/resolve/main/sample2.wav"></audio> |
| shukriya, bohat shukriya! | <audio controls src="https://huggingface.co/zafar205/tiny-urdu-tts-4.5-million-parameters/resolve/main/sample3.wav"></audio> |

## Model details

- **Base model:** [tiny-voca](https://github.com/n4ze3m/tiny-voca) by [@n4ze3m](https://x.com/n4ze3m), an English TTS model
- **Fine-tuning:** full fine-tune (all 4.5M parameters) on 1,000 clips from [UrduTTS](https://huggingface.co/datasets/kaab4321/UrduTTS), on a free Kaggle T4 GPU
- **Input:** Roman Urdu text, converted to phonemes by a small rule-based phonemizer
- **Output:** 24 kHz mono audio
- **Voice:** one male Urdu voice (`M2`)

## Files

| File | Use |
|---|---|
| `model.onnx` | Browser and CPU inference (onnxruntime / onnxruntime-web) |
| `model.json` / `config.json` | Config: sample rate, voices, phoneme symbols |
| `sample1-3.wav` | Audio samples |
| `inference.pt` | PyTorch checkpoint, for further fine-tuning with tiny-voca |

## How to use

The model takes phoneme IDs, not raw text. The Roman Urdu phonemizer is in the GitHub repo:

- **Browser / JavaScript:** [`web/lib/urdu.ts`](https://github.com/Zafar205/tiny-urdu/blob/main/web/lib/urdu.ts) and [`web/app/page.tsx`](https://github.com/Zafar205/tiny-urdu/blob/main/web/app/page.tsx)
- **Python:** `python urdu/run.py tiny_voca.infer --checkpoint inference.pt --voice M2 --text "aap kaise hain?" --output out.wav`

ONNX inputs: `tokens`, `stress` (int64 `[1, N]`), `voice` (int64 `[1]`), `speed`, `pitch_shift`, `pitch_range` (float32 `[1]`). Output: `audio` (float32 `[1, samples]`).

## Writing tips

- Write it the way you'd text: `mujhe chai pasand hai`, `shukriya, bohat shukriya!`
- A capital **T, D, R** inside a word marks the retroflex sound: `baRa` (big) vs `bura` (bad)

## Limitations

- Trained on only about 1,000 clips from a single speaker, so expect some mispronunciations and a robotic tone.
- Roman Urdu spelling varies, so some words may be read wrong.
- Urdu script (اردو) input is not supported yet.

## Credits

- [tiny-voca](https://github.com/n4ze3m/tiny-voca) by [@n4ze3m](https://x.com/n4ze3m) (Apache-2.0)
- [UrduTTS dataset](https://huggingface.co/datasets/kaab4321/UrduTTS)
- [facebook/wav2vec2-lv-60-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft), used for alignment
