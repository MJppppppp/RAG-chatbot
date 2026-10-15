# into1 A/S Chatbot — RAG customer-support bot for KakaoTalk

A production retrieval-augmented chatbot that handles first-line A/S (after-sales) support for an e-reader service centre over KakaoTalk. Built and deployed solo as a contract project; in production since August 2026.

> **What's in this repo:** the full application code, tests, and a synthetic sample dataset.
> **What's not:** the production dataset (real customer conversations), the embedding index, runtime logs, and API keys. See [Data & privacy](#data--privacy).

---

## What it does

A customer messages the KakaoTalk channel. The bot:

1. **Masks PII** in the incoming message (13 regex families, fail-closed) before anything is logged
2. **Runs a guard chain** — ~30 deterministic rules for cases where a wrong answer costs money or safety: repair prices, cancellations, refunds, battery-swelling safety, out-of-scope models. These return canned answers and never call an LLM.
3. **Falls through to retrieval** for everything else — embeds the question, cosine-searches ~900 past support cases, re-ranks with a small model, and has GPT-4o write an answer grounded only in those cases
4. **Filters the output** — strips any price not in an approved set, any warranty claim the customer didn't ask about, any "I looked at your photo" hallucination, and any mention of a model other than the customer's

Measured on 3,600 production queries: ~57% resolved by guards (no API cost), ~43% by retrieval.

## Architecture

```
KakaoTalk ──▶ Open Builder ──▶ /skill (FastAPI, Render)
                                   │
                                   ├─ PII mask
                                   ├─ guard chain  ──▶ canned answer (no LLM)
                                   │
                                   └─ retrieve ──▶ rerank ──▶ GPT-4o ──▶ output filters ──▶ reply
                                        │
                                   into1_vectors.npy  (text-embedding-3-large, 3072-d)
```

Kakao's 5-second skill timeout is handled with an immediate ack + callback; see `main.py`.

A full walkthrough — request lifecycle, retrieval maths, the guard layer, privacy engineering, cost, and the things that went wrong — is in [`ARCHITECTURE_EXPLAINER.md`](ARCHITECTURE_EXPLAINER.md).

## Running it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

# Build the index from the synthetic sample (12 rows, ~$0.001)
python build_index.py

# Run the guard-chain test suite — no API key needed, <1 s
python tests/test_guards.py

# Start the skill server
uvicorn main:app --host 0.0.0.0 --port 8000
```

`build_index.py` reads `sample_dataset.jsonl` by default. Point it at your own data with `DATASET=your_file.jsonl`.

## Testing

`tests/test_guards.py` runs 100+ multi-turn conversations through the guard chain and asserts on the reply — which price was quoted, which guard fired, what must *not* appear. It needs no API key because the guard path is pure Python. This is the suite that caught the bugs described in the explainer (a battery question quoting the screen price; "케이스 교체 비용" routed to a 200,000원 panel repair; newlines being stripped from every answer by a sentence-split regex).

## Data & privacy

The production dataset is ~900 structured rows extracted from real customer support conversations. It is **not** in this repository and will not be published: the customers never consented to that, and it isn't mine to release. `sample_dataset.jsonl` is 12 fabricated rows in the same schema so the code is runnable.

Runtime logs (`query_log.jsonl`) are excluded for the same reason — they contain customer questions, PII-masked but best-effort.

The embedding index (`into1_vectors.npy`, `into1_meta.json`) is excluded because `into1_meta.json` contains the row text verbatim.

API keys are read from environment variables only. Nothing is hardcoded.

## Stack

Python 3.10 · FastAPI · OpenAI (`text-embedding-3-large`, `gpt-4o`, `gpt-4o-mini`) · NumPy · Render · 카카오 i 오픈빌더

## Licence

Code is published with the client's permission for portfolio purposes. Not licensed for reuse in a competing service.
