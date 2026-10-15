# into1 A/S Chatbot — How It Actually Works

A complete walkthrough of the system, written so you can explain any part of it
out loud without notes. Every number here was read out of the code, not remembered.

---

## 0. The 30-second version

A Korean e-reader repair centre gets the same customer questions every day on
KakaoTalk. They have no engineers. I built a chatbot that answers those questions
by searching 873 of their own past support conversations and having GPT write a
reply grounded in whatever it found — and, critically, that refuses to answer
when it isn't confident.

Stack: Python, FastAPI, OpenAI embeddings + GPT-4o, deployed on Render, connected
to KakaoTalk through Kakao i Open Builder.

The interesting parts are not the LLM call. They are: how it decides *not* to
answer, how it survives a 5-second platform timeout, and how it strips personal
data before anything leaves the building.

---

## 1. System map

```
    CUSTOMER (KakaoTalk)
           │  "화면이 안 켜져요"
           ▼
    Kakao i Open Builder          ← conversation menus, buttons, fallback block
           │  POST /skill  (5-second limit!)
           ▼
    ┌──────────────────────────────────────────┐
    │  main.py   (FastAPI on Render)           │
    │   • auth check (X-Skill-Token)           │
    │   • PII masking  ← fail-closed           │
    │   • rate limiting                        │
    │   • return "one moment" in <1s           │
    │   • spawn background worker  ────────┐   │
    └──────────────────────────────────────┼───┘
                                           ▼
    ┌──────────────────────────────────────────┐
    │  rag_core.py                             │
    │   1. guard rules (no LLM, instant)       │
    │   2. embed the question                  │
    │   3. cosine search over 873 vectors      │
    │   4. threshold filter                    │
    │   5. LLM re-rank                         │
    │   6. build prompt, call GPT              │
    │   7. output filters (strip claims)       │
    └──────────────────────────────────────────┘
                     │ answer
                     ▼
        POST to Kakao's callback URL → customer
```

---

## 2. Part One — Building the knowledge base (offline)

This runs on my laptop, not on the server. `build_index.py`.

**Input:** `into1_rag_dataset.jsonl` (826 rows scraped and structured from real
KakaoTalk support history) + `extra_knowledge.jsonl` (47 rows of facts that live
in staff heads, not chat logs — opening hours, prices, policies). Total **873**.

Each row has this shape:

```
id, 카테고리(category), 모델(model), 문제(problem),
증상표현(how customers phrase it), 해결책(the solution),
확인여부(confirmed or not), 처리유형(handling type),
embed_text  ← the field that actually gets embedded
```

**Why `embed_text` is separate from the rest.** The text you search against is not
the same as the text you show the model. `embed_text` concatenates category +
model + problem + customer phrasings, because that is what a customer's question
will resemble. The 해결책 (solution) is *not* in the search text — customers
describe symptoms, not solutions. Searching solution text would match the wrong
things.

**Process:** batches of 100 rows → `text-embedding-3-large` → a 3072-dimensional
vector per row → saved as `into1_vectors.npy` (873 × 3072 float32, ~10.7 MB) plus
`into1_meta.json` holding the row data and, importantly, **which embedding model
was used**.

**Why record the model name in the metadata.** If someone later changes
`EMBED_MODEL`, the dimensions change and every stored vector becomes meaningless.
Recording it means the server can compare and fail loudly instead of silently
returning garbage. This is the kind of bug that produces no error, just bad
answers.

---

## 3. Part Two — What happens when a customer sends a message

### 3.1 Kakao calls `POST /skill`

The payload contains the utterance, a user ID (used as a session key), and a
`callbackUrl`.

### 3.2 Auth (`main.py`)

```python
if SKILL_TOKEN and req.headers.get("X-Skill-Token") != SKILL_TOKEN:
    return Response(status_code=401)
```

A shared secret set in both Render and Open Builder. Without it, `/skill` is a
public URL that anyone can hammer to burn the company's OpenAI credit. The check
is skipped if `SKILL_TOKEN` is unset — a deliberate choice so a misconfiguration
degrades service rather than taking the bot offline, with a warning printed at boot.

### 3.3 PII masking — **fail-closed**

```python
try:
    utter = get_core().mask_pii(utter)
except Exception:
    return handoff_to_human()      # do NOT continue with raw text
```

This is the single most important design decision in the file. If masking throws,
the original message — possibly containing a name, phone number, address — would
otherwise flow onward to OpenAI and into logs. Instead the request is abandoned
and the customer is offered a human.

**Fail-open vs fail-closed** is worth being able to explain: a fail-open system
keeps working when a safety component breaks. A fail-closed system stops. For
privacy, stopping is correct.

### 3.4 Rate limiting — *before* any OpenAI call

```python
RATE_PER_MIN  = 10      # per user
RATE_PER_HOUR = 60      # per user
MAX_CONCURRENT = 8      # whole server
```

Per-user limits use a `deque` of timestamps as a sliding window; entries older
than an hour are popped off the left. The server-wide limit is a
`threading.Semaphore`. The ordering matters: a blocked request costs nothing
because it never reaches the API.

Two separate failure modes, two separate mechanisms:
- One person spamming → burns money → per-user limit
- Everyone at once → threads pile up, 0.5 CPU server dies → concurrency cap

### 3.5 Return immediately, work in the background

This is where the 5-second problem gets solved. See Part Three.

---

## 4. Part Three — The 5-second problem

**The constraint:** Kakao gives a skill server 5 seconds to respond. Miss it and
the customer's message is lost entirely — no error, no retry, nothing.

**The reality:** embedding (~0.3s) + retrieval (~0.05s) + LLM re-ranking (~1–2s)
+ GPT-4o generation (~3–8s) = regularly over 10 seconds.

**The resolution:** Kakao supports a callback pattern. You reply within 5 seconds
with a placeholder and a promise, then POST the real answer to a callback URL when
it's ready.

```python
if cb:
    threading.Thread(target=worker, args=(utter, cb, sid), daemon=True).start()
    return {"version": "2.0", "useCallback": True,
            "data": {"text": "확인하고 있어요. 잠시만 기다려 주세요!"}}
```

Three details that are easy to get wrong:

**`CB_MIN_DELAY = 1.5`** — if a guard rule answers in 0.0 seconds, the callback
can arrive *before* Kakao has finished setting up to receive it, and the message
vanishes. So instant answers are deliberately delayed to 1.5 seconds. Making the
system slower on purpose to make it reliable.

**`WORKER_DEADLINE = 25`** — the worker thread joins with a timeout. If generation
hangs, we send the handoff message rather than letting the callback window expire
silently. Something is always sent.

**Retry once** — the callback POST is attempted twice with a 1-second gap, because
a transient network failure would otherwise mean the customer waits forever.

*Why this is a good interview story:* the platform's limit was not negotiable, the
work genuinely takes longer than the limit, and the fix required understanding the
platform's own escape hatch rather than optimising the code.

---

## 5. Part Four — Retrieval, and the numbers behind it

### 5.1 The mechanic

```python
q    = embed(question)          # 3072-dim, normalised to length 1
sims = VEC @ q                  # dot product of unit vectors = cosine similarity
idx  = argsort(sims)[::-1][:CAND_K]
out  = [(META[i], sims[i]) for i in idx if sims[i] >= THRESHOLD]
```

All stored vectors are pre-normalised at load, so a single matrix multiply gives
cosine similarity against all 873 cases at once. At this size it's instant — no
vector database needed, and adding one would be complexity without benefit.

### 5.2 The confidence funnel

| Threshold | Value | What it decides |
|---|---|---|
| `SIM_THRESHOLD` | **0.38** | Below this, a case isn't even a candidate |
| `CLARIFY_SIM` | 0.45 | Below this, ask a clarifying question instead of answering |
| `SKIP_RERANK_SIM` | 0.55 | Above this, the match is obvious — skip the LLM re-rank, save time and money |
| pinned floor | 0.60 | Score given to keyword-forced official answers |

Read top to bottom it's rising confidence: *worth considering* → *confident enough
to answer* → *so obvious we can skip a step* → *official answer, don't argue*.

### 5.3 Why 0.38 — the honest answer

It was tuned by hand, not derived. But the corpus geometry explains why the
ballpark is right. Measured across all 873 vectors:

- Two **unrelated** cases average **0.465** similarity
- A case and its **true nearest match** average **0.834**

Everything sits in a narrow band because the whole corpus is one domain (e-reader
repair) — even unrelated cases share vocabulary. Real customer questions are short
and colloquial while stored cases are rich and structured, so query-to-case scores
land *lower* than case-to-case scores. That's why a floor that looks low in
absolute terms is meaningful for live queries.

**Say this in an interview:** "0.38 is empirical. What I can defend is the
reasoning — I measured the distribution, I know unrelated pairs sit near 0.46
case-to-case, and I know queries score lower than that. The threshold is a floor
below which a match is indistinguishable from domain noise."

### 5.4 Re-ranking

Vector similarity finds *topically* similar cases. It cannot tell that "battery
swells" and "battery drains fast" are different problems. So a second pass sends
the candidate list to `gpt-4o-mini` with instructions to return only cases that
*directly* answer the question, as JSON.

If re-ranking returns fewer than `MIN_KEEP` (1) cases → hand off to a human. The
bot would rather say nothing than answer from a loosely related case.

**Why a cheap model here:** re-ranking is a filtering task, not a writing task.
`gpt-4o-mini` is good enough and roughly 20× cheaper. Model selection per task,
not one model for everything.

---

## 6. Part Five — The guard layer (why rules run before the AI)

Before any embedding happens, `answer()` runs a chain of hardcoded checks. Each
one that fires returns a fixed string and never touches the LLM.

| Guard | Triggers on | Why it's not left to the AI |
|---|---|---|
| `is_gibberish` | `ㅁㄴㅇㄹ` | Nothing to search |
| `is_opener` | "안녕하세요" with no symptom | Ask what they need first |
| `is_vague_symptom` | "고장났어요" | Guessing from a vague symptom produces confident nonsense |
| `is_model_only` | "페블이요" alone | A model name is not a problem |
| `AGENT_REQUEST` | "상담원 연결" | Instant, free |
| **`SAFETY`** | swollen battery, burning smell, smoke | **Must never depend on retrieval succeeding** |
| `is_visit` | "방문하고 싶어요" | Fixed company policy |
| `is_appearance` | scratches, stains | Goes to a photo board, not the A/S form |
| `is_price_question` | "얼마예요" | Must not quote a price without knowing the model |
| `PANEL-BREAK` | confirmed panel damage + model | Fixed policy: replacement, not repair |
| `detect_unsupported` | 크레마 업, other brands | Prevent promising service they don't offer |

**The safety one is the point.** If a customer says their battery is swelling,
the correct response is "stop using it immediately" — and that must not depend on
a vector search returning a good match. Only 7 cases in the dataset mention
swelling. Retrieval could plausibly miss. A keyword rule cannot.

**General principle worth stating out loud:** anything where being wrong is
expensive should not be left to a probabilistic system. Use the LLM for the long
tail; hardcode the parts that matter.

---

## 7. Part Six — Privacy engineering

### 7.1 Where masking runs

Two points, both before data leaves the process:
1. Before the text is sent to OpenAI
2. Before it is written to any log

`_log()` calls `mask_pii(q)` again independently, so a log write can't leak even
if the caller forgot.

### 7.2 What it catches — 13 pattern families

Phone numbers (mobile and landline), email, 주민등록번호, credit card, bank
account, order/waybill numbers, any 10+ digit run, passport numbers,
외국인등록번호, four address forms (road-name, lot-number, unit "101동 502호",
apartment-name), and three name rules.

### 7.3 Names are the hard part

Korean names are 2–3 characters and look like ordinary words. Three strategies,
applied in order:

1. **Keyword context** — "성함은 김철수입니다" → the word before gives it away
2. **Adjacency to an already-masked phone number** — this is the clever one:
   ```
   "김철순 010-1234-5678"
     → phone rule runs first  →  "김철순 [연락처]"
     → name rule matches a surname immediately before [연락처]  →  "[이름] [연락처]"
   ```
   **Regex order in the `_PII` list is load-bearing.** Later rules consume the
   output of earlier ones.
3. **Honorifics** — "홍길동님" → name + 님

**False positives were the real problem.** "고객님" is surname 고 + 객 + 님 — it
matches the honorific rule perfectly and is not a name. Solution: a `_NAME_STOP`
set of ~50 common words (고객, 담당, 화면, 충전, 제품 …) checked before
substitution.

That is a genuine precision/recall tradeoff and worth naming as one: loosen the
name rules and you mask "고객님" into nonsense; tighten them and you miss real
names. The stop-list buys precision without giving up much recall.

---

## 8. Part Seven — Output filtering (after generation)

The LLM's answer is not trusted as-is. Seven filters run over it:

| Filter | Removes |
|---|---|
| `_strip_unapproved_prices` | Any figure not in `APPROVED_PRICES` — repair {90,000 / 120,000 / 150,000 / 170,000 / 190,000 / 200,000 / 220,000}, retail {219,000 / 459,000 / 469,000}, trade-in {279,600 / 289,600 for 7.8 2nd, 255,600 / 265,600 for 2nd Plus}. Updated 2026-10-01 when 인투원 raised PCB prices; the old 140,000 is deliberately no longer approved so stale transcript rows can't surface it. |
| `_strip_all_prices` | All prices, if the customer never asked about cost |
| `_strip_other_models` | Sentences about models other than the confirmed one |
| `_strip_warranty` | Warranty claims when warranty wasn't asked about |
| `_strip_speculative_diagnosis` | Cause-of-damage assertions made without seeing the device |
| `_strip_seen_attachment` | **"보내주신 사진 확인 결과…"** |
| `_strip_false_claims` | Invented procedures like advance shipping |

**The photo one is my favourite example of RAG going subtly wrong.** 26 of the
training cases were written by a human agent who *had* been sent photos. The model
learned that voice and started saying "based on the photo you sent" to customers
who had sent nothing. Nothing in the pipeline was broken — retrieval was correct,
generation was faithful to the source. The training data simply contained a
context the live system doesn't have.

That's a good thing to be able to articulate: **RAG inherits the assumptions of
its source material, not just its facts.**

---

## 9. Part Eight — Sessions, cache, cost

**Sessions** — last 6 turns, 30-minute TTL, keyed by Kakao user ID. Used to keep
context ("페블이요" after "화면이 깨졌어요" must continue the same topic) and to
detect that a previously suggested fix already failed, so the bot escalates
instead of repeating itself.

**Cache** — SHA-1 of the normalised question → answer, LRU capped at 2000, 24-hour
TTL. Only used when the session has no prior turns, because a cached answer would
ignore conversational context. Common questions cost nothing on repeat.

**Cost** — about **₩14 (~£0.008) per conversation** measured. Many conversations
cost zero because a guard rule answered without any API call. The re-rank skip
above 0.55 removes another call on obvious matches.

**Boot behaviour** — `rag_core` is imported lazily in a background thread at
startup, not at module import. If it were imported at the top, the very first
customer message would wait for a 10 MB index load plus TLS handshake and blow
the 5-second window. Warmup also fires one throwaway embedding to pay the
connection cost before a real customer does.

---

## 10. Part Nine — The legal layer

Not code, but part of the system, and unusual for an engineer to have done:

- **Cross-border transfer disclosure** (PIPA Art. 28-2/8) — customer messages go
  to OpenAI and Render, both US. Korean law requires disclosing seven specific
  items per recipient plus a refusal route.
- **AI disclosure** — Korea's AI Framework Act requires telling users they're
  talking to generative AI.
- **Processing consignment** (Art. 26) — OpenAI, Render, Kakao listed as processors.
- **Retention** — logs live on Render's ephemeral disk and vanish on redeploy, so
  the policy states a period that matches reality. Writing "3 months" when the
  real answer is "until the next deploy" is itself a violation.
- **Pseudonymisation basis** (Art. 28-2) — the dataset was extracted from real
  support history originally collected for a different purpose.

**The point to make:** the system could not legally operate without this, and
nobody else at the company could write it. Compliance was part of shipping, not
an afterthought.

---

## 11. Decisions worth defending

**Flat numpy array instead of a vector database.** 873 × 3072 is 10 MB. A single
matrix multiply searches everything in microseconds. Pinecone or FAISS would add
an operational dependency for zero measurable gain. Choose complexity when the
data demands it, not in advance.

**Two models, not one.** `gpt-4o` writes, `gpt-4o-mini` re-ranks. Filtering is
easier than composing.

**Config in a plain file with Korean comments.** After I leave, nobody at the
company codes. `config.py` is editable in Notepad and every value is annotated in
Korean with what happens if you raise or lower it. Threshold tuning shouldn't
require a developer.

**A `BUILD` string in the health endpoint.** `/` returns build ID, case count,
uptime, and load error. It answers "did my upload actually deploy?" in one glance
— the question a non-technical operator will ask most often.

**Menus in Kakao's console, not in code.** Buttons are configured in Open Builder
so staff can rewrite them without a deploy. The tradeoff is that button text
becomes a search query, so labels must be full symptom sentences — "충전이 안
돼요", never "배터리".

---

## 12. Things that went wrong (the best interview material)

**The 크레마 false positives.** I wrote a general rule to catch unsupported
크레마 models. Tested against the dataset: **51 false positives** — "크레마
리모콘", "크레마 기종이", "크레마 모델중" all matched. I removed the general rule
entirely and replaced it with an explicit list of nine model names. Hit rate
dropped to 35, all genuine.

The reasoning: telling a real PEBBLE owner their device isn't supported is worse
than the miss we started with. **Asymmetric error cost decides the design.**

**The stale-index accusation.** I flagged the vector index as out of date because
its file timestamp was older than the dataset's. Then I actually compared all 872
embedding inputs and they were identical — the dataset had been touched without
its content changing. mtime is not evidence. I retracted it.

**The threshold that wasn't where I thought.** `rag_core.py` defaults
`SIM_THRESHOLD` to 0.30, but `config.py` sets 0.38 and config wins. Two sources of
truth for one number is a bug waiting to happen; the health endpoint should
probably report the effective value.

---

## 13. Explaining it to a non-technical person

> The bot doesn't know anything about e-readers. What it has is 873 conversations
> your team already had with customers. When a new question comes in, it finds the
> few past conversations that are most similar in meaning — not matching words,
> meaning — and asks an AI to write a reply based only on those.
>
> The important part is what happens when nothing matches well. Instead of
> guessing, it says it can't help and offers a person. I'd rather it stay quiet
> than confidently tell a customer the wrong repair price.

That second paragraph is the one that matters to a customer, and it's the same
idea as the 0.67 number: knowing the line below which you don't trust yourself.

---

## 14. Numbers to have ready

| | |
|---|---|
| Cases in the index | 873 (826 chat history + 47 curated facts) |
| Embedding | `text-embedding-3-large`, 3072-dim |
| Index size | 10.7 MB (873 × 3072 float32) |
| Similarity floor | 0.38 |
| Clarify below | 0.45 |
| Skip re-rank above | 0.55 |
| Candidates → final | 12 → 4 |
| Unrelated pair similarity | ~0.465 average |
| True-match similarity | ~0.834 average |
| Kakao response limit | 5 seconds |
| Worker deadline | 25 seconds |
| Minimum callback delay | 1.5 seconds |
| Rate limits | 10/min, 60/hour per user; 8 concurrent |
| PII pattern families | 13 |
| Name stop-words | ~50 |
| Approved price whitelist | 6 values |
| Session memory | 6 turns, 30 min |
| Cache | 2000 entries, 24h |
| Cost per conversation | ~₩14 (~£0.008) |

---

## 15. If you only remember three things

1. **The system's best feature is that it refuses to answer.** 0.38, 0.45,
   MIN_KEEP — three separate places where it hands off instead of guessing.
2. **Guards before AI.** Safety, prices, and unsupported models are hardcoded
   because being wrong there costs more than being silent.
3. **The 5-second limit was solved by understanding the platform, not the code.**
   Callbacks existed; the work was finding them and handling the three edge cases
   that come with async delivery.
