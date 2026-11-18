"use client";
import { useEffect, useRef, useState } from "react";
import { Frontend } from "../lib/urdu";

const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
type Config = { sample_rate: number; voices: string[]; symbols: string[]; max_tokens: number };
type Phones = ReturnType<Frontend["phonemize"]>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare global { interface Window { ort: any } }

function loadOrt(): Promise<void> {
  if (window.ort) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = ORT + "ort.wasm.min.js";
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("could not load onnxruntime-web"));
    document.head.appendChild(s);
  });
}

function wav(samples: Float32Array, rate: number): Blob {
  const buf = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const str = (o: number, s: string) => [...s].forEach((c, i) => buf.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); buf.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVEfmt ");
  buf.setUint32(16, 16, true); buf.setUint16(20, 1, true); buf.setUint16(22, 1, true);
  buf.setUint32(24, rate, true); buf.setUint32(28, rate * 2, true); buf.setUint16(32, 2, true);
  buf.setUint16(34, 16, true); str(36, "data"); buf.setUint32(40, samples.length * 2, true);
  samples.forEach((v, i) => buf.setInt16(44 + i * 2, Math.max(-1, Math.min(1, v)) * 0x7fff, true));
  return new Blob([buf], { type: "audio/wav" });
}

export default function Home() {
  const [text, setText] = useState("aap kaise hain? mujhe chai pasand hai.");
  const [status, setStatus] = useState("Loading model…");
  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(false);
  const [voices, setVoices] = useState<string[]>([]);
  const [voice, setVoice] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [url, setUrl] = useState("");
  const model = useRef<{ session: any; config: Config; frontend: Frontend } | null>(null); // eslint-disable-line @typescript-eslint/no-explicit-any

  useEffect(() => {
    (async () => {
      try {
        await loadOrt();
        const ort = window.ort;
        ort.env.wasm.wasmPaths = ORT;
        ort.env.wasm.numThreads = 1;
        const config: Config = await (await fetch("/model.json")).json();
        const session = await ort.InferenceSession.create("/model.onnx", { executionProviders: ["wasm"] });
        model.current = { session, config, frontend: new Frontend(config.symbols) };
        setVoices(config.voices);
        const m2 = config.voices.indexOf("M2");
        setVoice(m2 >= 0 ? m2 : 0);
        setStatus("");
        setReady(true);
      } catch (e) {
        setError(true);
        setStatus("Could not load the model: " + (e as Error).message);
      }
    })();
  }, []);

  // One utterance holds at most max_tokens phonemes: cut long sentences at commas, then by words.
  function fit(sentence: string): Phones[] {
    const { frontend, config } = model.current!;
    const p = frontend.phonemize(sentence);
    if (!p.symbols.some((s) => frontend.phones.has(s))) return [];
    if (p.ids.length <= config.max_tokens - 12) return [p];
    const parts = sentence.split(/(?<=[,;:—])\s+/);
    if (parts.length > 1) return parts.flatMap(fit);
    const words = sentence.split(" ");
    if (words.length < 2) return [];
    const half = Math.ceil(words.length / 2);
    return [...fit(words.slice(0, half).join(" ")), ...fit(words.slice(half).join(" "))];
  }

  async function speak() {
    if (!model.current || busy) return;
    const { session, config } = model.current;
    const ort = window.ort;
    setBusy(true);
    setError(false);
    setStatus("");
    try {
      const started = performance.now();
      const utterances = text.split(/(?<=[.?!…])\s+/).filter((s) => s.trim()).flatMap(fit);
      if (!utterances.length) throw new Error("nothing to pronounce: type some Roman Urdu");
      const gap = new Float32Array(Math.round(config.sample_rate * 0.15));
      const chunks: Float32Array[] = [];
      for (const u of utterances) {
        const n = u.ids.length;
        const ints = (v: number[]) => new ort.Tensor("int64", BigInt64Array.from(v.map(BigInt)), [1, n]);
        const f = (v: number) => new ort.Tensor("float32", Float32Array.from([v]), [1]);
        const out = await session.run({
          tokens: ints(u.ids), stress: ints(u.stress),
          voice: new ort.Tensor("int64", BigInt64Array.from([BigInt(voice)]), [1]),
          speed: f(speed), pitch_shift: f(0), pitch_range: f(1),
        });
        if (chunks.length) chunks.push(gap);
        chunks.push(out.audio.data as Float32Array);
      }
      const all = new Float32Array(chunks.reduce((a, c) => a + c.length, 0));
      chunks.reduce((o, c) => (all.set(c, o), o + c.length), 0);
      const peak = all.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
      if (peak > 0.99) all.forEach((v, i) => (all[i] = (v * 0.99) / peak));
      if (url) URL.revokeObjectURL(url);
      setUrl(URL.createObjectURL(wav(all, config.sample_rate)));
      setStatus(`${(all.length / config.sample_rate).toFixed(1)}s of speech in ${((performance.now() - started) / 1000).toFixed(2)}s`);
    } catch (e) {
      setError(true);
      setStatus((e as Error).message);
    }
    setBusy(false);
  }

  return (
    <main>
      <header>
        <span className="badge">On-device text-to-speech</span>
        <h1>Tiny <span>Urdu</span></h1>
        <p className="urdu" lang="ur" dir="rtl">چھوٹی اردو</p>
      </header>

      <section className="card">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) speak(); }}
          placeholder="Kuch likhiye… e.g. aap kaise hain?"
          rows={4}
          spellCheck={false}
        />

        <div className="controls">
          <div className="field">
            <span className="label">Voice</span>
            <div className="segmented">
              {voices.map((v, i) => (
                <button key={v} className={i === voice ? "on" : ""} onClick={() => setVoice(i)}>{v}</button>
              ))}
            </div>
          </div>
          <div className="field grow">
            <span className="label">Speed <b>{speed.toFixed(1)}×</b></span>
            <input type="range" min={0.6} max={1.6} step={0.1} value={speed} onChange={(e) => setSpeed(+e.target.value)} />
          </div>
          <button className="speak" onClick={speak} disabled={!ready || busy || !text.trim()}>
            {busy || !ready ? <span className="spinner" /> : <PlayIcon />}
            {!ready && !error ? "Loading" : busy ? "Speaking" : "Speak"}
          </button>
        </div>

        {url && <audio src={url} controls autoPlay />}
        {status && <p className={error ? "status error" : "status"}>{status}</p>}
      </section>

      <footer>
        21 MB model · runs in your browser · built on{" "}
        <a href="https://github.com/n4ze3m/tiny-voca" target="_blank" rel="noreferrer">tiny-voca</a> ·{" "}
        <a href="https://github.com/Zafar205/tiny-urdu" target="_blank" rel="noreferrer">source</a>
      </footer>
    </main>
  );
}

function PlayIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M8 5.14v13.72a1 1 0 0 0 1.5.86l11-6.86a1 1 0 0 0 0-1.72l-11-6.86A1 1 0 0 0 8 5.14z" /></svg>;
}
