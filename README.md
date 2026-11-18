# tiny-urdu

A tiny text-to-speech model that speaks **Roman Urdu**. You type `aap kaise hain?` and it speaks in Urdu. It runs entirely in the browser.

It is [tiny-voca](https://github.com/n4ze3m/tiny-voca) (4.5M parameters, English only), fine-tuned on **1,000 clips** of the [UrduTTS](https://huggingface.co/datasets/kaab4321/UrduTTS) dataset with free Kaggle GPU time.

```
"mujhe chai pasand hai."
  -> Roman Urdu phonemizer  ->  m ʊ ɟʰ eː <sp> c aː eː <sp> p a s a n d <sp> h ɛ .
  -> model.onnx (21 MB)     ->  24 kHz audio
```

## How it works

1. **Phonemes, not letters.** Roman Urdu has no standard spelling (`hai`, `hay`, `he`), so the model never sees letters. It is trained on the dataset's IPA phonemes. At inference, a rule-based phonemizer turns Roman Urdu into those same phonemes.
2. **Native Urdu phones.** tiny-voca's 76 English symbols are kept, and 44 Urdu phones are appended: retroflex `ʈ ɖ ɽ`, aspirated `kʰ tʰ bʰ …`, `q x ɣ ʋ`, and long and nasal vowels `aː eː ẽ ã …`. The English IDs don't change, so the English checkpoint is a good starting point. The embedding grows from 76 to 120.
   - The first attempt mapped Urdu sounds onto English ones (`kʰ → k h`, `ʈ → t`). The forced aligner rejected **998 of 1,000 clips**. Using the aligner's own Urdu labels fixed it.
3. **Fine-tune.** The pipeline is tiny-voca's own: prepare → annotate (wav2vec2 forced alignment) → train. It ran for 6,000 steps on a Kaggle T4. Training uses the single speaker's voice in the `M2` slot.
4. **Browser.** The model is exported to ONNX. The phonemizer is ported to TypeScript, and its output matches the Python version on every test sentence. Inference runs on the visitor's device with onnxruntime-web.

## Writing Roman Urdu

- Write it the way you'd text it: `kya haal hai`, `shukriya, bohat shukriya!`
- **Capital `T`, `D`, `R` inside a word mean retroflex:** `baRa` is big and `bura` is bad; `gaaRi`, `Thanda`. A capital first letter is just sentence case.
- `kh`, `gh`, `ch`, `chh`, `th`, `bh` … are aspirated or digraph sounds, `aa`/`ee`/`oo` are long vowels, and `ai` is `ɛ` (as in `hai`).
- Is a word pronounced wrong? Add it to `LEXICON_EXTRA` in [web/lib/urdu.ts](web/lib/urdu.ts) **and** `_LEXICON_EXTRA` in [urdu/frontend.py](urdu/frontend.py). No retraining is needed.

## Layout

| Path | What |
|---|---|
| `urdu/frontend.py` | Urdu phone inventory and Roman Urdu → phonemes. It patches tiny-voca's text module. |
| `urdu/run.py` | Runs any `tiny_voca.*` module with the Urdu frontend installed |
| `urdu/prepare_data.py` | Streams UrduTTS, resamples to 24 kHz and writes `data/*.jsonl` |
| `urdu/make_init_checkpoint.py` | Turns the released English weights into a step-0 training state |
| `make_notebook.py` → `tiny_urdu_kaggle.ipynb` | The whole pipeline as a resumable Kaggle notebook |
| `web/` | Next.js app: `lib/urdu.ts` (phonemizer), `public/model.onnx` and `model.json` |
| `tests/test_frontend.py` | Frontend tests |

## Train it yourself (Kaggle)

1. Upload `tiny_urdu_kaggle.ipynb` to Kaggle and set **Accelerator: GPU** and **Internet: on**.
2. Click **Save Version** to run everything. It took about 1h 45m on a T4. Each stage is cached, so a re-run continues where it stopped.
3. Cell 6 exports `web/model.onnx` and `model.json`.

To change the number of training clips, set `N_TRAIN`. To change the training length, set `STEPS`.

## Export locally

If you downloaded `runs/urdu2/inference.pt`:

```bash
python urdu/run.py tiny_voca.export --checkpoint runs/urdu2/inference.pt --output web/public/model.onnx
```

This needs `torch`, `onnx` and tiny-voca installed (`pip install -e vendor/tiny-voca`).

## Run the web app

```bash
cd web && npm install && npm run dev
```

To deploy on Vercel, import the repo with **Root Directory = `web`**. The site is fully static, with no server or GPU.

## Limits

- 1,000 clips from one speaker is a small amount of data. Expect some mispronunciations and a robotic tone.
- The phonemizer is rules plus a small lexicon. Roman Urdu spelling varies a lot, so some words will be read wrong.
- Roman Urdu input only. Urdu script (اردو) is not supported yet.

## Credits

- [tiny-voca](https://github.com/n4ze3m/tiny-voca) by n4ze3m (Apache-2.0): the model, training recipe and ONNX export.
- [UrduTTS](https://huggingface.co/datasets/kaab4321/UrduTTS): audio and IPA transcriptions.
- [facebook/wav2vec2-lv-60-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft): forced alignment.
