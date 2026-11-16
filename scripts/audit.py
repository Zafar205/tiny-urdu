import io, random, collections, statistics as st, time, numpy as np, soundfile as sf
from datasets import load_dataset, Audio
random.seed(0)
def get(split, n):
    ds = load_dataset("kaab4321/UrduTTS", split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    t=time.time(); rows=[]
    for r in ds.take(n):
        rows.append(r)
        if len(rows)%250==0: print(f"[{split}] {len(rows)}/{n} rows, {time.time()-t:.0f}s",flush=True)
    print(f"[{split}] done {len(rows)} rows in {time.time()-t:.0f}s",flush=True); return rows
train=get("train",1500); test=get("test",300)
cols=list(train[0].keys()); print("columns:",cols)
for r in train[:3]: print({k:v for k,v in r.items() if k!="audio"})
rows=train+test
def isar(s): return sum('؀'<=c<='ۿ' for c in s)/max(1,len(s))
strc=[c for c in cols if c!="audio" and isinstance(train[0][c],str)]
ur=max(strc,key=lambda c:np.mean([isar(r[c]) for r in train[:50]]))
ph=[c for c in strc if c!=ur and any(ch in "".join(r[c] for r in train[:50]) for ch in "ɪʊɑəɾ")]
print("urdu col:",ur,"| phoneme cols:",ph,"| speaker col:",[c for c in cols if "spk" in c.lower() or "speaker" in c.lower()])
# durations, sr, channels
def meta(r):
    d,sr=sf.read(io.BytesIO(r["audio"]["bytes"]),dtype="float32"); return len(d)/sr,sr,1 if d.ndim==1 else d.shape[1]
for name,rs in [("train[:1500]",train),("test[:300]",test)]:
    m=[meta(r) for r in rs]; du=[x[0] for x in m]
    print(f"{name}: mean {np.mean(du):.2f} med {np.median(du):.2f} min {min(du):.2f} max {max(du):.2f} >15s {np.mean([d>15 for d in du]):.3%} sr {set(x[1] for x in m)} ch {set(x[2] for x in m)}")
alld=[meta(r)[0] for r in rows]
print(f"pooled mean {np.mean(alld):.2f}s -> est hours = {np.mean(alld)*57873/3600:.1f} (57,873 rows from README; README says 91.9 h)")
for p in ph:
    cnt=collections.Counter("".join(r[p] for r in rows)); print(f"\n[{p}] top60:",cnt.most_common(60))
    print("targets:",{c:cnt[c] for c in "ʈɖɽʰʱɣxqʃ"},"stress:",{c:cnt[c] for c in "ˈˌ"})
    sel=[r for r in rows if any(c in r[ur] for c in "ٹڈڑھ")]
    print(f"{len(sel)}/{len(rows)} rows contain ٹ ڈ ڑ ھ; 10 examples:")
    for r in random.sample(sel,10): print(" ",r[ur],"|",r[p])
for i,r in enumerate(random.sample(rows,10)):
    d,sr=sf.read(io.BytesIO(r["audio"]["bytes"])); sf.write(f"samples/clip{i}.wav",d,sr)
print("saved 10 wavs")
