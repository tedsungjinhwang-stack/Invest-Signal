"""상승초입 — 15m 역배열에서 종가가 MA960 근처 · 월 상단밴드가 분기 상단밴드 위.

  ① 15m **역배열** — MA240 < MA480 < MA960 (2.5일 < 5일 < 10일).
     짧은 눈금에서는 아직 하락 구조다. 그래서 MA960이 **제일 위**에 있다.
  ② **월 앵커드 VWAP 상단밴드(+1σ)가 분기 상단밴드보다 위** — 이번 달
     가격대가 분기 평균보다 높다. 월 단위로는 이미 올라서고 있다는 뜻이다.
  ③ 종가가 **15m MA960 근처**(±near_pct) — 역배열의 제일 위 선까지
     올라와 닿은 자리다.

셋이 같이 서면 '월 단위로는 오르는데 짧은 눈금은 아직 역배열이고, 그
역배열의 천장(10일선)까지 가격이 올라왔다'가 된다.

> 처음엔 ③이 '종가가 분기 상단과 월 상단 **사이**'였다. 그걸 960선 근처로
> 바꾸고 ②는 두 밴드의 **위아래 관계**만 남겼다.

VWAP 밴드는 TradingView의 Anchored VWAP(Standard Deviation 모드)과 같은
식이다: 소스 (고+저+종)/3, 거래량 가중 표준편차, 배수 1.

**프레임이 둘이다.** ①③은 15m, ②는 4h로 잰다 — 분기 앵커드 VWAP을 15m으로
계산하려면 한 분기치 ≈ 8,800봉이라 종목당 요청이 아홉 번이 된다. 4h 프레임은
스캔이 이미 받아 두고, 실측으로 밴드 값 차이가 0.04~0.4%였다.

**분기 첫 달(1·4·7·10월)에는 ②가 성립하지 않는다.** 월과 분기가 같은 날부터
누적해서 두 상단밴드가 똑같아지기 때문이다.

**조건에 처음 들어온 봉에서 알리고, 머무는 동안 추적한다.** 상태 조건이라
매 스캔 다시 알리면 같은 말을 반복하게 된다.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..indicators import alignment, anchored_vwap_bands, pct_over
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
    band_mult: float = 1.0              # ② 표준편차 배수
    band_top: str = "M"                 # ② 이 상단밴드가
    band_bottom: str = "Q"              # ② 이 상단밴드보다 위에 있어야 한다
    near_ma: int = 960                  # ③ 종가가 이 15m 이평선 근처여야 한다
    near_pct: float = 0.02              # ③ '근처' 허용폭 — ±2%
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
    need = max(max(params.ma_align), params.near_ma)
    if len(df15) < need + 2 or df4h is None or len(df4h) < 10:
        return None
    up_top = _band_on_15m(df15, df4h, params.band_top, params.band_mult)
    up_bot = _band_on_15m(df15, df4h, params.band_bottom, params.band_mult)
    if up_top is None or up_bot is None:
        return None
    c = df15["Close"]
    mas = [c.rolling(k).mean() for k in params.ma_align]
    bear = pd.Series(True, index=df15.index)
    for a, b in zip(mas, mas[1:]):
        bear &= a < b
    bear &= ~mas[-1].isna()
    ma = c.rolling(params.near_ma).mean()
    dist = c / ma - 1
    near = dist.abs() <= params.near_pct
    # 7d = 15m 672봉. 1,000봉을 받으므로 마지막 구간에서는 늘 구할 수 있다.
    d7 = c / c.shift(7 * 96) - 1
    return pd.DataFrame({"close": c, "bear": bear, "up_top": up_top,
                         "up_bot": up_bot, "ma": ma, "dist": dist, "ret_7d": d7,
                         "ok": bear & (up_top > up_bot) & near})


def _detail(row, params: Params) -> dict:
    """줄에 실을 값 — 960선에서 얼마나 떨어졌는지와 종목 수익률."""
    d = {"label": LABEL, "interval": INTERVAL,
         "near_ma": params.near_ma,
         # 종가 ÷ MA960 − 1 — 음수면 선 아래, 양수면 선 위
         "ma_dist": float(row.dist),
         "band_top": float(row.up_top), "band_bottom": float(row.up_bot),
         "align_mas": params.ma_align}
    if pd.notna(row.ret_7d):
        d["ret_7d"] = float(row.ret_7d)
    return d


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
    d["in_bars"] = int(len(ok) - start)      # 조건에 머문 15m 봉 수
    # 머문 시간이 **판정이 시작되는 봉**까지 거슬러 올라갔으면 그보다 오래
    # 머물렀을 수 있다. 1,000봉 중 MA960이 서는 건 마지막 41봉(≈10시간)뿐이라,
    # 그 앞은 역배열을 잴 수가 없어 무조건 '밖'으로 나온다. 표시를 10h+로 한다.
    first = int(np.argmax(~np.isnan(st.ma.to_numpy(float))))
    d["in_capped"] = bool(start <= first)
    return SignalEvent(symbol=symbol, signal=NAME, bar_time=st.index[start],
                       price=float(st.close.iloc[start]), detail=d)
