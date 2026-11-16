import io, collections, random, numpy as np, soundfile as sf, pyarrow as pa
from huggingface_hub import hf_hub_download
random.seed(1)
f=hf_hub_download("kaab4321/UrduTTS","train/data-00020-of-00041.arrow",repo_type="dataset",local_dir="/Users/muhammadzafar/urdu-audit/shard")
print("downloaded",f,flush=True)
src=pa.memory_map(f); 
try: rd=pa.ipc.open_stream(src)
except Exception: rd=pa.ipc.open_file(src)
rows=[]
for b in (rd if hasattr(rd,'schema') and not hasattr(rd,'num_record_batches') else (rd.get_batch(i) for i in range(rd.num_record_batches))):
    rows+=b.to_pylist()
    if len(rows)>=500: break
rows=rows[:500]; cols=list(rows[0]); print("columns",cols,"rows",len(rows))
def isar(s): return sum('؀'<=c<='ۿ' for c in s)/max(1,len(s))
strc=[c for c in cols if c!="audio" and isinstance(rows[0][c],str)]
ur=max(strc,key=lambda c:np.mean([isar(r[c]) for r in rows[:50]]))
ph=[c for c in strc if c!=ur and any(ch in "".join(r[c] for r in rows[:50]) for ch in "ɪʊɑəɾ")]
def meta(r):
    d,sr=sf.read(io.BytesIO(r["audio"]["bytes"]),dtype="float32"); return len(d)/sr,sr,1 if d.ndim==1 else d.shape[1]
m=[meta(r) for r in rows]; du=[x[0] for x in m]
print(f"mid-shard: mean {np.mean(du):.2f} med {np.median(du):.2f} min {min(du):.2f} max {max(du):.2f} >15s {np.mean([d>15 for d in du]):.3%} sr {set(x[1] for x in m)} ch {set(x[2] for x in m)}")
print("est hours (mean*57873):",round(np.mean(du)*57873/3600,1))
for r in rows[:3]: print({k:v for k,v in r.items() if k!="audio"})
for p in ph:
    cnt=collections.Counter("".join(r[p] for r in rows)); print(f"[{p}] top60:",cnt.most_common(60))
    print("targets:",{c:cnt[c] for c in "ʈɖɽʰʱɣxqʃ"},"stress:",{c:cnt[c] for c in "ˈˌ"})
    print("ٹڈڑھ rows:",sum(any(c in r[ur] for c in "ٹڈڑھ") for r in rows))
