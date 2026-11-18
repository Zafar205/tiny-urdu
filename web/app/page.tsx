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
  const [busy, setBusy] = useState(true);
  const [voice, setVoice] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [url, setUrl] = useState("");
  const [phones, setPhones] = useState("");
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
        const m2 = config.voices.indexOf("M2");
        setVoice(m2 >= 0 ? m2 : 0);
        setStatus("Ready");
        setBusy(false);
      } catch (e) {
        setStatus("Failed to load: " + (e as Error).message + " (are model.onnx and model.json in web/public?)");
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
    if (!model.current) return;
    const { session, config } = model.current;
    const ort = window.ort;
    setBusy(true);
    setStatus("Speaking…");
    try {
      const started = performance.now();
      const utterances = text.split(/(?<=[.?!…])\s+/).filter((s) => s.trim()).flatMap(fit);
      setPhones(utterances.map((u) => u.symbols.join(" ")).join("\n"));
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
      setStatus(`Done: ${(all.length / config.sample_rate).toFixed(2)}s of audio in ${((performance.now() - started) / 1000).toFixed(2)}s`);
    } catch (e) {
      setStatus("Error: " + (e as Error).message);
    }
    setBusy(false);
  }

  return (
    <main>
      <h1>Tiny Urdu</h1>
      <p className="hint">Type Roman Urdu. Capital T, D, R mean retroflex (baRa = big). Runs fully in your browser.</p>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4} />
      <div className="row">
        <label>Voice{" "}
          <select value={voice} onChange={(e) => setVoice(+e.target.value)}>
            {model.current?.config.voices.map((v, i) => <option key={v} value={i}>{v}</option>)}
          </select>
        </label>
        <label>Speed {speed.toFixed(1)}{" "}
          <input type="range" min={0.6} max={1.6} step={0.1} value={speed} onChange={(e) => setSpeed(+e.target.value)} />
        </label>
        <button onClick={speak} disabled={busy || !text.trim()}>Speak</button>
      </div>
      <p className="status">{status}</p>
      {url && <audio src={url} controls autoPlay />}
      {phones && <pre>{phones}</pre>}
    </main>
  );
}
