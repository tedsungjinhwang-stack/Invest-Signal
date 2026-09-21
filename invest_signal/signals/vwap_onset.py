"""상승초입 — 분기 상단밴드 위 · 월 상단밴드 아래 (15m 역배열).

**두 시간 눈금이 엇갈리는 자리를 잡는다.**

  ① 15m **역배열** — MA240 < MA480 < MA960 (2.5일 < 5일 < 10일).
     짧은 눈금에서는 아직 하락 구조다.
  ② 종가가 **분기 앵커드 VWAP 상단밴드(+1σ) 위**.
     분기 기준으로는 이미 평균 위로 한참 올라왔다.
  ③ 종가가 **월 앵커드 VWAP 상단밴드(+1σ) 아래**.
     이달 기준으로는 아직 위가 남았다.

즉 **캔들이 분기 상단과 월 상단 사이**에 있는 구간이다. ②가 '바닥은
지났다'를, ③이 '아직 안 갔다'를 말하고, ①이 '짧은 눈금은 아직 안 돌았다'를
말한다 — 셋이 같이 서면 오래 눌렸다가 막 올라오기 시작한 자리가 된다.

VWAP 밴드는 TradingView의 Anchored VWAP(Standard Deviation 모드)과 같은
식이다: 소스 (고+저+종)/3, 거래량 가중 표준편차, 배수 1.

**프레임이 둘이다.** ①은 15m, ②③은 4h로 잰다 — 분기 앵커드 VWAP을 15m으로
계산하려면 한 분기치 ≈ 8,800봉이라 종목당 요청이 아홉 번이 된다. 4h 프레임은
스캔이 이미 받아 두고, 실측으로 밴드 값 차이가 0.04~0.4%였다(BTC·NEAR·AVA를
15m 10,559봉과 맞대 봤다). 밴드는 넓은 레벨이라 이 정도는 판정을 안 바꾼다.

**구간에 처음 들어온 봉에서 알리고, 머무는 동안 추적한다.** 상태 조건이라
매 스캔 다시 알리면 같은 말을 반복하게 된다.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..indicators import alignment, anchored_vwap_bands
from . import SignalEvent

NAME = "vwap_onset"
LABEL = "상승초입"
CRYPTO_ONLY = True
INTERVAL = "15m"
# MA960 + 판정 여유. 바이낸스 요청당 상한이 1000이라 이게 최대다 —
# 마지막 41봉에서만 MA960이 서고, 신규 판정(grace 4봉)에는 넉넉하다.
KLINE_LIMIT = 1000


@dataclass(frozen=True)
class Params:
    ma_align: tuple = (240, 480, 960)   # ① 15m 역배열 판정선
    band_mult: float = 1.0              # ②③ 표준편차 배수
    above_anchor: str = "Q"             # ② 이 밴드 **위**에 있어야 한다
    below_anchor: str = "M"             # ③ 이 밴드 **아래**에 있어야 한다
    grace_bars: int = 4                 # 15m × 4 = 1시간(스캔 주기)
    min_turnover_usd: float = 1_000_000
    track_bars: int = 96                # 추적 상한 — 96봉 = 하루


def _band_on_15m(df15: pd.DataFrame, df4h: pd.DataFrame, anchor: str,
                 mult: float) -> pd.Series | None:
    """4h로 낸 상단밴드를 15m 인덱스에 얹는다(계단식 전방 채움).

    4h 한 봉 안의 15m 열여섯 봉은 같은 밴드 값을 본다 — 밴드가 4시간에
    한 번 갱신된다는 뜻이고, VWAP이 느리게 움직이니 문제가 되지 않는다.
    """
    got = anchored_vwap_bands(df4h, anchor, mult)
    if got is None:
        return None
    upper = got[2].dropna()
    if upper.empty:
        return None
    return upper.reindex(df15.index, method="ffill")


def _state(df15: pd.DataFrame, df4h: pd.DataFrame,
           params: Params) -> pd.DataFrame | None:
    """15m 봉마다 ①②③ 판정 — 못 재면 None."""
    need = max(params.ma_align)
    if len(df15) < need + 2 or df4h is None or len(df4h) < 10:
        return None
    up_q = _band_on_15m(df15, df4h, params.above_anchor, params.band_mult)
    up_m = _band_on_15m(df15, df4h, params.below_anchor, params.band_mult)
    if up_q is None or up_m is None:
        return None
    c = df15["Close"]
    mas = [c.rolling(k).mean() for k in params.ma_align]
    bear = pd.Series(True, index=df15.index)
    for a, b in zip(mas, mas[1:]):
        bear &= a < b
    bear &= ~mas[-1].isna()
    return pd.DataFrame({"close": c, "bear": bear, "up_q": up_q, "up_m": up_m,
                         "ok": bear & (c > up_q) & (c < up_m)})


def _detail(row, params: Params) -> dict:
    """줄에 실을 값 — 구간 안 어디쯤인지까지 같이 낸다."""
    span = row.up_m - row.up_q
    pos = (row.close - row.up_q) / span if span > 0 else None
    return {"label": LABEL, "interval": INTERVAL,
            "band_above": float(row.up_q), "band_below": float(row.up_m),
            # 0이면 분기 상단에 붙어 있고 1이면 월 상단에 닿았다는 뜻 —
            # 구간의 어디쯤인지가 '얼마나 남았나'를 바로 말해 준다.
            "band_pos": None if pos is None else float(pos),
            "to_upper": float(row.up_m / row.close - 1),
            "align_mas": params.ma_align}


def detect(df15: pd.DataFrame, df4h: pd.DataFrame, symbol: str,
           params: Params = Params()) -> list[SignalEvent]:
    """구간에 **처음 들어온** 봉을 찾는다(직전 봉은 밖이어야 한다).

    마지막 grace_bars+1개 봉을 후보로 본다 — 스캔이 한 시간 간격이라
    그 사이 지나간 봉도 소급해 잡는다.
    """
    st = _state(df15, df4h, params)
    if st is None:
        return []
    n = len(st)
    out = []
    for t in range(max(1, n - 1 - params.grace_bars), n):
        if not st.ok.iloc[t] or st.ok.iloc[t - 1]:
            continue
        row = st.iloc[t]
        out.append(SignalEvent(symbol=symbol, signal=NAME, bar_time=st.index[t],
                               price=float(row.close), detail=_detail(row, params)))
    return out


def tracking(df15: pd.DataFrame, df4h: pd.DataFrame, symbol: str,
             params: Params = Params()):
    """지금도 구간 안이면 추적 줄용 이벤트 — 아니면 None.

    신규로 나가는 창(grace_bars) 안에서 막 들어온 건은 돌려주지 않는다 —
    같은 봉이 두 칸에 실리면 같은 말을 두 번 한다.
    """
    st = _state(df15, df4h, params)
    if st is None or not st.ok.iloc[-1]:
        return None
    ok = st.ok.to_numpy()
    start = len(ok) - 1
    while start > 0 and ok[start - 1]:
        start -= 1
    if start >= len(ok) - 1 - params.grace_bars:
        return None                     # 방금 들어왔다 — 신규 줄이 맡는다
    row = st.iloc[-1]
    d = _detail(row, params)
    d["last_price"] = float(row.close)
    d["in_bars"] = int(len(ok) - start)      # 구간에 머문 15m 봉 수
    return SignalEvent(symbol=symbol, signal=NAME, bar_time=st.index[start],
                       price=float(st.close.iloc[start]), detail=d)
