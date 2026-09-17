"""급등봉 시그널 (15m봉) — 크립토 전용.

**장대양봉 하나를 잡는다.** 다른 시그널이 '자리'(눌림·터치·배열)를 보는
것과 달리, 이건 봉 하나의 모양만 본다 — 조용하던 종목에 갑자기 매수가
쏟아져 15분 만에 크게 오른 순간이다.

  ① 몸통      (종가 − 시가) ÷ 시가 ≥ body
  ② 거래량    직전 vol_ma봉(기본 96봉 = 24시간) 평균의 vol_mult배 이상
  ③ 종가위치  (종가 − 저가) ÷ (고가 − 저가) ≥ close_pos
  ④ 거래대금  24h 거래대금이 하한 이상 (호출 측에서 티커로 건다)

**②가 진짜 판별자다.** 몸통 +8%짜리 15분봉은 잡코인에서 흔하지만, 거래량이
평소의 열 배로 터지는 건 드물다. 실측(368종·5.2일·15m 14만 봉, 거래대금
$1M 이상): 몸통 8%만 보면 하루 27건인데 ②·③을 얹으면 하루 3.6건이 된다.

**③이 없으면 이미 되밀린 봉이 섞인다.** 고가에서 한참 밀려 마감한 봉은
그 시점에 이미 끝난 사건이다. 2026-09-17 AVA(몸통 24.3% · 거래량 54배)는
종가위치 0.88이었다.

**추적하지 않는다.** 급등봉은 상태가 아니라 순간이라, 그 봉에서 한 번
알리고 끝낸다. 같은 종목이 다시 터지면 새 봉이니 다시 알린다.
"""

from dataclasses import dataclass

import pandas as pd

from . import SignalEvent

NAME = "spike_bar"
LABEL = "급등봉"
CRYPTO_ONLY = True          # ETF·주식 스캔에서는 돌리지 않는다
INTERVAL = "15m"
KLINE_LIMIT = 200           # vol_ma(96) + grace + 여유


@dataclass(frozen=True)
class Params:
    body: float = 0.08          # ① 몸통 상승률 하한
    vol_mult: float = 10.0      # ② 거래량 / 직전 평균
    vol_ma: int = 96            # ② 평균을 낼 봉 수 (96봉 = 24시간)
    close_pos: float = 0.7      # ③ 종가위치 — 1이면 고가 마감
    min_turnover_usd: float = 1_000_000   # ④ 24h 거래대금 하한
    # 스캔이 한 시간 간격이라 그 사이 지나간 봉도 소급해 잡는다(15m × 4 = 1시간).
    # **이 시그널은 소급이 특히 중요하다** — 봉이 15분짜리라 소급이 없으면
    # 스캔 직전 봉 하나만 보게 되어 네 봉 중 셋을 놓친다.
    grace_bars: int = 4


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """마감된 15m OHLCV(오름차순, UTC 인덱스)에서 급등봉을 찾는다.

    마지막 grace_bars+1개 봉을 각각 독립 후보로 본다. 중복 발송 방지는
    호출 측(state)이 dedup_key로 처리한다.
    """
    n = len(df)
    if n < params.vol_ma + 2:
        return []
    o, h, l, c, v = (df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])
    # 평균은 **판정 봉을 빼고** 낸다 — 급등봉 자신이 평균을 밀어 올리면
    # 배수가 작아져서, 클수록 잡히기 어려워지는 거꾸로 된 기준이 된다.
    volma = v.shift(1).rolling(params.vol_ma).mean()

    out = []
    for t in range(max(params.vol_ma, n - 1 - params.grace_bars), n):
        op, hi, lo, cl = (float(o.iloc[t]), float(h.iloc[t]),
                          float(l.iloc[t]), float(c.iloc[t]))
        base = float(volma.iloc[t]) if pd.notna(volma.iloc[t]) else 0.0
        if op <= 0 or hi <= lo or base <= 0:
            continue
        body = cl / op - 1
        mult = float(v.iloc[t]) / base
        pos = (cl - lo) / (hi - lo)
        if body < params.body or mult < params.vol_mult or pos < params.close_pos:
            continue
        out.append(SignalEvent(
            symbol=symbol, signal=NAME, bar_time=df.index[t], price=cl,
            detail={"label": LABEL, "body": body, "vol_mult": mult,
                    "close_pos": pos, "interval": INTERVAL,
                    "bar_high": hi, "bar_low": lo},
        ))
    return out


def still_active(df: pd.DataFrame, event: SignalEvent, params: Params = Params()) -> bool:
    """추적하지 않는다 — 급등봉은 상태가 아니라 순간이다."""
    return False
