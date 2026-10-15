"""카카오톡 스킬 서버 (인투원 A/S 봇)"""
import os, json, time, threading, httpx
from collections import deque
from fastapi import FastAPI, Request, Response
# 배포가 실제로 반영됐는지 눈으로 확인하는 표식.
# 업로드할 때마다 이 값을 바꾸면 헬스체크에서 바로 보인다.
BUILD = "2026-10-02-01"

app = FastAPI()

# rag_core 는 임포트 시 색인(.npy)과 OpenAI 클라이언트를 만든다.
# 최상단에서 임포트하면 서버가 완전히 준비될 때까지 첫 요청이 대기하게 되어,
# 카카오 5초 제한을 넘겨 첫 메시지가 통째로 유실된다. → 지연 로딩 + 사전 예열.
_CORE = None
_CORE_ERR = None
_core_lock = threading.Lock()
START_TS = time.time()
HANDOFF = "죄송합니다, 제가 정확히 안내드리기 어려운 내용이에요.\n상담원 연결해 드리겠습니다."

def get_core():
    global _CORE, _CORE_ERR
    if _CORE is None:
        with _core_lock:
            if _CORE is None:
                try:
                    import rag_core
                    _CORE = rag_core
                    print(f"[WARM] 색인 로드 완료 ({len(rag_core.META)}건, "
                          f"{time.time()-START_TS:.1f}s)")
                except Exception as e:
                    _CORE_ERR = repr(e); print("[WARM] 로드 실패:", e); raise
    return _CORE

def _warmup():
    """부팅 직후 색인 로드 + OpenAI 연결(TLS/DNS)까지 미리 끝내둔다."""
    try:
        core = get_core()
        core._embed("예열")                      # 첫 임베딩의 핸드셰이크 비용 제거
        core.start_heartbeat()                  # 이후 유휴 상태에서도 커넥션 유지
        print(f"[WARM] 예열 완료 ({time.time()-START_TS:.1f}s)")
    except Exception as e:
        print("[WARM] 예열 실패:", e)

@app.on_event("startup")
def _on_start():
    threading.Thread(target=_warmup, daemon=True).start()

AGENT_LABEL  = os.environ.get("AGENT_LABEL", "상담원 연결")
REPORT_LABEL = os.environ.get("REPORT_LABEL", "답변이 이상해요")
# 버튼 노출 여부 — Render 환경변수로 끄고 켤 수 있다 (코드 재배포 불필요)
SHOW_AGENT_QR  = os.environ.get("SHOW_AGENT_QR",  "1") != "0"   # 상담원 연결 버튼
SHOW_REPORT_QR = os.environ.get("SHOW_REPORT_QR", "1") != "0"   # 답변이 이상해요 버튼

# ── /skill 공유 비밀 헤더 ─────────────────────────────────────
# 오픈빌더 스킬 화면의 "헤더값 입력" 에 X-Skill-Token = <같은 값> 을 넣고,
# Render 환경변수 SKILL_TOKEN 에도 같은 값을 넣으면 외부 호출이 차단된다.
# 설정하지 않으면 검사를 건너뛴다 — 운영 중 서비스가 끊기지 않게 하려는 것이며,
# 설정 전까지는 누구나 /skill 을 호출해 OpenAI 크레딧을 소모할 수 있다.
SKILL_TOKEN = os.environ.get("SKILL_TOKEN")
if not SKILL_TOKEN:
    print("[경고] SKILL_TOKEN 미설정 — /skill 이 인증 없이 공개되어 있습니다.")

# 고객이 쓸 일 없는 단어여야 한다. "초기화" 는 기기 초기화 문의와 충돌하므로 제외.
#
# "처음으로" — 오픈빌더의 [처음으로] 블록과 짝을 이룬다 (2026-08-25 추가).
#   메뉴 처음으로 돌아갔는데 이전 대화가 남아 있으면, 다음에 누른 짧은 버튼
#   ("충전이 안 돼요" 등)이 직전 주제와 합쳐져 검색된다.
#   예: "화면이 깨졌어요" 뒤에 "충전이 안 돼요" → 패널 사례가 딸려 나옴.
#   따라서 처음으로 돌아갈 때 대화 기록도 함께 비운다.
RESET_WORDS  = {"/reset", "reset", "대화초기화", "대화초기화하기", "리셋하기",
                "새로시작", "처음으로"}

# 되돌림 안내 문구. 오픈빌더 [처음으로] 블록이 메뉴를 함께 띄운다면
# 이 문구가 겹쳐 보일 수 있으므로, Render 환경변수 RESET_TEXT 로 바꿀 수 있다.
RESET_TEXT = os.environ.get("RESET_TEXT", "이전 대화 내용을 지웠습니다. 새로 문의해 주세요.")

def _is_reset(u):
    """공백·대소문자·마침표에 영향받지 않도록 정규화 후 비교.

    정확히 그 단어일 때만 True 다. "처음으로 돌아가고 싶어요" 처럼
    문장 안에 들어 있으면 일반 문의로 처리된다 (요청대로 단어 1개일 때만)."""
    return (u or "").strip().lower().replace(" ", "").rstrip(".!") in RESET_WORDS
LOG_FILE     = os.environ.get("CHAT_LOG", "chat_log.jsonl")
_llock = threading.Lock()

def _save(kind, user, q, a, hits=()):
    """모든 대화와 신고를 기록 (팀 테스트용)"""
    try:
        with _llock, open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.strftime("%m-%d %H:%M"), "kind": kind,
                "user": (user or "")[-6:],          # 식별용 뒤 6자리만
                "q": q, "a": a,   # q는 이미 마스킹된 상태
                "ids": [h["id"] for h, _ in hits],
                "sim": round(hits[0][1], 2) if hits else None,
            }, ensure_ascii=False) + "\n")
    except Exception as e:
        print("LOG ERR:", e)

# ── A/S 신청 방법 안내 (접수를 권하는 답변에 두 번째 말풍선으로 자동 첨부) ──
# A/S 안내 문구는 rag_core.AS_GUIDE_TEXT 하나만 사용한다 (문구 불일치 방지)

# 기기를 보내야 하는 답변에만 붙인다.
# "점검" 같은 단어 하나로 판단하면 "서버가 점검 중" 같은 문장에도 붙어버린다.
AS_PHRASES = (
    "제품 수거", "온라인 A/S", "A/S 신청", "A/S 접수", "AS 접수", "신청서 접수",
    "서비스 대상", "유상 서비스", "무상 서비스", "수리 비용",
    # "제품 입고" / "입고 및 점검" 은 가격 안내문에도 들어 있어서
    # 모든 가격 답변 뒤에 방문접수 안내가 따라붙던 문제 → 제외 (2026-09-30)
    "점검이 필요", "점검을 진행", "점검 후 확정",
    "교환해 드리", "교환 처리",
)
# 기다리면 해결되는 상황 — 접수 안내가 오히려 방해가 된다.
NOT_AS = (
    "서버", "점검 중", "일시적", "다시 시도", "재시도", "기다려",
    "기다리시", "후에 다시", "잠시 후",
    "문의 게시판", "문의게시판",   # 게시판 접수 안내에는 A/S 신청서를 덧붙이지 않는다
)

# A/S 접수를 안내한 답변에는 항상 표준 A/S 안내를 함께 보낸다.
# (고객이 어떤 단어를 썼는지로 판단하면 '패널 손상' 처럼 목록에 없는 표현에서 빠진다)
def needs_as_guide(text, question=""):
    t = text or ""
    if "into1.co.kr" in t:            # 이미 안내·링크가 들어 있으면 중복 금지
        return False
    if any(w in t for w in NOT_AS):   # 기다리면 해결 · 게시판 접수 → 붙이지 않는다
        return False
    return any(p in t for p in AS_PHRASES)

def kakao_text(t, offer_agent=False, question=""):
    outs = [{"simpleText":{"text":t}}]
    if needs_as_guide(t, question):        # 두 번째 말풍선으로 신청 방법 안내
        outs.append({"simpleText":{"text":get_core().AS_GUIDE_TEXT}})
    tpl = {"outputs": outs}
    qr = []
    if offer_agent and SHOW_AGENT_QR:     # 답변 실패 시 상담원 연결 버튼
        qr.append({"label": AGENT_LABEL, "action": "message", "messageText": AGENT_LABEL})
    if SHOW_REPORT_QR:                    # 팀 테스트용 오답 신고 버튼
        qr.append({"label": REPORT_LABEL, "action": "message", "messageText": REPORT_LABEL})
    if qr:
        tpl["quickReplies"] = qr
    return {"version":"2.0","template":tpl}

# 콜백이 너무 빨리 도착하면 카카오가 아직 준비되지 않아 메시지가 유실됨.
# 가드(안전/환불/불만 등)는 0초에 응답하므로 최소 대기 시간을 둔다.
CB_MIN_DELAY = float(os.environ.get("CB_MIN_DELAY", "1.5"))

WORKER_DEADLINE = float(os.environ.get("WORKER_DEADLINE", "25"))

# ── 요청 제한 ────────────────────────────────────────────────
# ① 사용자별 제한 — 한 사람이 계속 보내서 OpenAI 비용을 태우는 것을 막는다
# ② 동시 처리 제한 — 스레드가 무한정 늘어나 서버가 멈추는 것을 막는다
# 셋 다 Render 환경변수로 조절 가능 (코드 재배포 불필요)
RATE_PER_MIN  = int(os.environ.get("RATE_PER_MIN",  "10"))   # 1인 1분당
RATE_PER_HOUR = int(os.environ.get("RATE_PER_HOUR", "60"))   # 1인 1시간당
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "8"))  # 서버 전체 동시 처리

BUSY_TEXT = ("지금 문의가 많아 잠시 지연되고 있습니다.\n"
             "잠시 후 다시 시도해 주시거나, 상담원 연결을 눌러 주세요.")
TOO_MANY_TEXT = ("짧은 시간에 많은 문의를 보내셨습니다.\n"
                 "잠시 후 다시 시도해 주시거나, 상담원 연결을 눌러 주세요.")

_rate, _rate_lock = {}, threading.Lock()
_worker_sem = threading.Semaphore(MAX_CONCURRENT)

def _rate_ok(sid):
    """사용자별 요청 빈도 확인. 허용이면 True 를 반환하며 기록도 남긴다."""
    if not sid:                       # 세션 ID 가 없으면(콘솔 테스트 등) 제한하지 않음
        return True
    now = time.time()
    with _rate_lock:
        dq = _rate.setdefault(sid, deque())
        while dq and now - dq[0] > 3600:        # 1시간 지난 기록 정리
            dq.popleft()
        if len(dq) >= RATE_PER_HOUR:
            return False
        if sum(1 for t in dq if now - t <= 60) >= RATE_PER_MIN:
            return False
        dq.append(now)
        if len(_rate) > 5000:                   # 메모리 보호 — 빈 항목부터 정리
            for k in [k for k, v in _rate.items() if not v][:1000]:
                _rate.pop(k, None)
        return True

def worker(q, cb, sid=None):
    """동시 처리 수를 넘으면 잠깐 기다렸다가, 그래도 자리가 없으면 안내 후 종료."""
    if not _worker_sem.acquire(timeout=3):
        print(f"[포화] 동시 처리 {MAX_CONCURRENT}건 초과 — 대기 안내 전송")
        try:
            httpx.post(cb, json=kakao_text(BUSY_TEXT, offer_agent=True), timeout=10)
        except Exception as e:
            print("CB ERR(busy):", e)
        return
    try:
        _worker(q, cb, sid)
    finally:
        _worker_sem.release()

def _worker(q, cb, sid=None):
    t0 = time.time()
    box = {}
    def _run():
        try:    box["r"] = get_core().answer(q, session_id=sid)
        except Exception as e: box["e"] = e
    th = threading.Thread(target=_run, daemon=True); th.start()
    th.join(WORKER_DEADLINE)          # 답변이 늦어도 콜백 만료 전에 반드시 무언가 보낸다

    try:
        if "r" not in box:
            reason = box.get("e") or f"{WORKER_DEADLINE}초 초과"
            print(f"[TIMEOUT] {q} — {reason}")
            _save("timeout", sid, q, str(reason))
            t, hits = HANDOFF, []
            raise StopIteration          # 아래 전송 단계로
        t, hits = box["r"]
        src = " | ".join(f"{h['id']}({h['카테고리']},{sc:.2f})" for h, sc in hits) or "근거없음→상담원"
        print(f"[Q] {q}\n[A] {t[:120]}...\n[근거] {src}")
        _save("chat", sid, q, t, hits)
    except StopIteration:
        pass
    except Exception as e:
        print("ERR:", e); t, hits = HANDOFF, []
        _save("error", sid, q, str(e))

    elapsed = time.time() - t0
    if elapsed < CB_MIN_DELAY:                 # 즉답이면 잠깐 대기 후 전송
        time.sleep(CB_MIN_DELAY - elapsed)

    for attempt in (1, 2):                     # 실패 시 1회 재시도
        try:
            _ctx = q + " " + get_core().session_user_text(sid)   # 대화 전체 기준으로 판단
            r = httpx.post(cb, json=kakao_text(t, offer_agent=not hits, question=_ctx), timeout=30)
            print(f"CB {attempt} status={r.status_code} ({elapsed:.1f}s)")
            if r.status_code < 300:
                return
        except Exception as e:
            print(f"CB ERR {attempt}:", e)
        time.sleep(1.0)

@app.post("/skill")
async def skill(req: Request):
    # 공유 비밀 헤더 검사 (SKILL_TOKEN 을 설정한 경우에만 적용)
    if SKILL_TOKEN and req.headers.get("X-Skill-Token") != SKILL_TOKEN:
        print("[차단] X-Skill-Token 불일치")
        return Response(status_code=401)

    body  = await req.json()
    ur    = body.get("userRequest", {})
    utter = (ur.get("utterance") or "").strip()
    cb    = ur.get("callbackUrl")
    sid   = (ur.get("user") or {}).get("id")     # 카카오 사용자 ID = 대화 세션 키
    # 개인정보 마스킹 — 이후 모든 처리(세션·로그·OpenAI)는 마스킹본만 사용한다
    # 마스킹이 실패하면 원문이 OpenAI·로그로 흘러가므로 처리를 중단한다 (fail-closed).
    try:
        utter = get_core().mask_pii(utter)
    except Exception as _e:
        print("[ERR] mask_pii 실패 — 개인정보 보호를 위해 처리 중단:", _e)
        return {"version": "2.0", "template": {
            "outputs": [{"simpleText": {"text": HANDOFF}}],
            "quickReplies": [{"label": AGENT_LABEL, "action": "message",
                              "messageText": AGENT_LABEL}]}}
    if _is_reset(utter):                     # 테스트용 대화 초기화
        try:    get_core().reset_session(sid)
        except Exception as e: print("RESET ERR:", e)
        print(f"[RESET] '{utter}' — user…{(sid or '')[-6:]} 대화 기록 삭제")
        return kakao_text(RESET_TEXT)

    if utter == REPORT_LABEL:                    # 테스터가 오답 신고
        _save("REPORT", sid, "[직전 답변 신고]", "")
        return kakao_text("신고해 주셔서 감사합니다. 확인 후 개선하겠습니다.")

    # 사용자별 요청 빈도 제한 — OpenAI 호출 전에 차단한다 (비용 방어)
    if not _rate_ok(sid):
        print(f"[제한] 요청 빈도 초과 user…{(sid or '')[-6:]}")
        return kakao_text(TOO_MANY_TEXT, offer_agent=True)

    if cb:
        threading.Thread(target=worker, args=(utter, cb, sid), daemon=True).start()
        return {"version":"2.0","useCallback":True,
                "data":{"text":"확인하고 있어요. 잠시만 기다려 주세요!"}}
    t, hits = get_core().answer(utter, session_id=sid)
    _save("chat", sid, utter, t, hits)
    return kakao_text(t, offer_agent=not hits,
                      question=utter + " " + get_core().session_user_text(sid))

# /review · /reset(HTTP) 라우트는 개인정보 노출 위험으로 제거했습니다 (2026-08-24).
# 대화 내용 확인은 Render → Logs 의 [Q]/[A] 출력으로만 합니다.
# 채팅방에서 "/reset" 등을 입력하는 대화 초기화 기능은 그대로 동작합니다.

@app.get("/")
def health(response: Response):
    """색인 로드를 기다리지 않는다 — 이 경로가 막히면 예열 핑이 무의미해진다.
    브라우저가 옛 응답을 캐시하면 배포 확인이 무의미해지므로 캐시를 금지한다."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return {
        "ok": True,
        "build": BUILD,
        "warm": _CORE is not None,
        "cases": len(_CORE.META) if _CORE else None,
        "uptime_sec": round(time.time() - START_TS, 1),   # 콜드스타트 판별용
        "load_error": _CORE_ERR,
    }
