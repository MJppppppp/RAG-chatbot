"""
1단계: 인덱스 생성. 데이터 수정할 때마다 재실행.
  pip install -r requirements.txt
  set OPENAI_API_KEY=sk-...
  python build_index.py
결과: into1_vectors.npy + into1_meta.json
※ 임베딩 모델을 바꾸면 반드시 인덱스를 다시 만들어야 합니다(차원이 다름).
"""
import json, os
import numpy as np
from openai import OpenAI

import re as _re
def _clean_key(v: str) -> str:
    """붙여넣기 과정에서 섞여 들어간 공백/제어문자 제거 (인증 오류 방지)"""
    return _re.sub(r"[^\x21-\x7e]", "", v or "")

client = OpenAI(api_key=_clean_key(os.environ.get("OPENAI_API_KEY","")))
try:
    import config as _cfg
    _def = getattr(_cfg, "EMBED_MODEL", "text-embedding-3-large")
except Exception:
    _def = "text-embedding-3-large"
MODEL = os.environ.get("EMBED_MODEL", _def)

DATASET = os.environ.get("DATASET", "sample_dataset.jsonl")   # 실제 운영: into1_rag_dataset.jsonl
rows = [json.loads(l) for l in open(DATASET, encoding="utf-8") if l.strip()]
# 상담이력 외 직접 관리하는 정보(판매/링크/영업시간 등)를 추가로 병합
if os.path.exists("extra_knowledge.jsonl"):
    extra = [json.loads(l) for l in open("extra_knowledge.jsonl", encoding="utf-8") if l.strip()]
    rows += extra
    print(f"  + extra_knowledge.jsonl {len(extra)}건 병합")
# 2026-09-17 6차 상담이력 추출분 (카카오 채널 7/23~9/17)
if os.path.exists("into1_rag_round6.jsonl"):
    r6 = [json.loads(l) for l in open("into1_rag_round6.jsonl", encoding="utf-8") if l.strip()]
    rows += r6
    print(f"  + into1_rag_round6.jsonl {len(r6)}건 병합")
print(f"{len(rows)}건 임베딩 시작 (model={MODEL})...")

vecs = []
for i in range(0, len(rows), 100):
    batch = rows[i:i+100]
    resp = client.embeddings.create(model=MODEL, input=[r["embed_text"] for r in batch])
    vecs.extend(d.embedding for d in resp.data)
    print(f"  {i+len(batch)}/{len(rows)}")

arr = np.array(vecs, dtype=np.float32)
np.save("into1_vectors.npy", arr)
meta = [{k: r[k] for k in ("id","카테고리","모델","문제","증상표현","해결책","확인여부","처리유형")} for r in rows]
json.dump({"embed_model": MODEL, "rows": meta}, open("into1_meta.json","w",encoding="utf-8"), ensure_ascii=False)
print(f"완료 → into1_vectors.npy {arr.shape}, into1_meta.json")
