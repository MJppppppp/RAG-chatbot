import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""멀티턴(대화 이어가기) 테스트:  python test_multiturn.py"""
import time
from rag_core import answer, reset_session

CONVERSATIONS = [
    # 1) 대화 도중 주제 전환 (기기 문제 → 액세서리 구매)
    ["페블 쓰는데 화면이 자꾸 멈춰요",
     "재부팅 해봤는데 똑같아요",
     "아 그리고 케이스도 사고 싶은데 어디서 사나요"],

    # 2) 모델을 먼저 말함 → 모델을 다시 묻지 말아야 함
    ["sam 7.8 쓰는데 글자 크기 어떻게 키워요"],

    # 3) 불만 누적 → 환불 요구 (대화 중에도 환불 가드 작동해야 함)
    ["수리 맡긴 지 2주 됐는데 연락이 없어요",
     "너무 오래 걸리는 거 아닌가요",
     "그냥 환불해주세요"],

    # 4) 대화 도중 안전 이슈 발생 (2턴째에 안전 가드가 떠야 함)
    ["배터리가 빨리 닳는 것 같아요",
     "그리고 어제부터 뒷면이 좀 볼록해진 것 같아요"],

    # 5) 긴 대화 + 앞에서 말한 내용 되짚기
    ["액정에 금이 갔어요",
     "페블이요",
     "얼마 정도 나올까요",
     "아까 말씀하신 주소로 직접 방문해도 되나요"],

    # 6) 대화 도중 범위 밖 이슈로 드러남
    ["책이 안 열려요",
     "리디북스에서 산 책이에요"],

    # 7) 짧은 예/아니오 응답 이해
    ["충전이 안 됩니다",
     "네 정품 충전기예요",
     "아니요 충전 램프도 안 들어와요"],
]

for ci, convo in enumerate(CONVERSATIONS, 1):
    sid = f"test-{ci}"
    reset_session(sid)
    print("\n" + "=" * 72)
    print(f"[대화 {ci}]")
    for turn, q in enumerate(convo, 1):
        t0 = time.time()
        reply, hits = answer(q, session_id=sid)
        src = ", ".join(f"{h['id']}({s:.2f})" for h, s in hits) or "근거 없음"
        print(f"\n  고객 {turn}> {q}")
        print(f"  봇   > {reply}")
        print(f"  ({time.time()-t0:.1f}초 · {src})")
print("\n완료")
