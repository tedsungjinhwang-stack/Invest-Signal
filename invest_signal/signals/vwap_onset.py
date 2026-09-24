"""상승초입 — 15m 역배열 또는 정배열 · 월 상단밴드가 분기 상단밴드 위 · 종가가 그 밴드 근처.

  ① 15m 배열 — 둘 중 **하나**만 서면 된다.
       역배열 MA240 < MA480 < MA960 (2.5일 < 5일 < 10일) — 짧은 눈금은 아직
       하락 구조인데 가격이 밴드까지 올라왔다.
       정배열 MA120 > MA240 > MA480 (1.25일 > 2.5일 > 5일) — 짧은 눈금이
       이미 돌아서 올라가는 중에 밴드에 닿았다.
  ② **월 앵커드 VWAP 상단밴드(+1σ)가 분기 상단밴드보다 위** — 이번 달
     가격대가 분기 평균보다 높다. 월 단위로는 이미 올라서고 있다는 뜻이다.
  ③ 종가가 **월 상단밴드 또는 분기 상단밴드 근처**(±near_pct) — 둘 중
     가까운 쪽 하나에만 붙어 있으면 된다.

> 처음엔 ③이 '종가가 분기 상단과 월 상단 **사이**'였고, 그다음 '15m 960선
> 근처'였다가 지금의 '두 상단밴드 근처'가 됐다. ①도 역배열만 보던 것에
> 정배열을 **또는**으로 더했다.

VWAP 밴드는 TradingView의 Anchored VWAP(Standard Deviation 모드)과 같은
식이다: 소스 (고+저+종)/3, 거래량 가중 표준편차, 배수 1.

**프레임이 둘이다.** ①은 15m, ②③의 밴드는 4h로 잰다 — 분기 앵커드 VWAP을
15m으로 계산하려면 한 분기치 ≈ 8,800봉이라 종목당 요청이 아홉 번이 된다.
4h 프레임은 스캔이 이미 받아 두고, 실측으로 밴드 값 차이가 0.04~0.4%였다.

**분기 첫 달(1·4·7·10월)에는 ②가 성립하지 않는다.** 월과 분기가 같은 날부터
누적해서 두 상단밴드가 똑같아지기 때문이다.

**조건에 처음 들어온 봉에서 알리고, 머무는 동안 추적한다.** 상태 조건이라
매 스캔 다시 알리면 같은 말을 반복하게 된다.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..indicators import anchored_vwap_bands
from . import SignalEvent

NAME = "vwap_onset"
LABEL = "상승초입"
CRYPTO_ONLY = True
INTERVAL = "15m"
# 역배열의 MA960 + 판정 여유. 바이낸스 요청당 상한이 1000이라 이게 최대다 —
# 마지막 41봉에서만 MA960이 서고, 신규 판정(grace 4봉)에는 넉넉하다.
# 정배열(MA480까지)은 마지막 521봉을 잴 수 있다.
KLINE_LIMIT = 1000


@dataclass(frozen=True)
class Params:
    bear_align: tuple = (240, 480, 960)  # ① 15m 역배열 판정선 (짧은 것부터)
    bull_align: tuple = (120, 240, 480)  # ① 15m 정배열 판정선 — 둘 중 하나면 된다
    band_mult: float = 1.0              # ②③ 표준편차 배수
    band_top: str = "M"                 # ② 이 상단밴드가
    band_bottom: str = "Q"              # ② 이 상단밴드보다 위에 있어야 한다
    near_pct: float = 0.02              # ③ 종가가 상단밴드의 ±2% 안
    near_both: bool = False             # ③ True면 두 밴드 **모두** 근처여야 한다
    rearm_bars: int = 96                # 나갔다 이만큼 안에 다시 들어오면 새로 안 알린다
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


def _stacked(c: pd.Series, periods: tuple, rising: bool) -> tuple:
    """(배열이 섰는가, 잴 수 있는가) — 짧은 선부터 차례로 비교한다.

    rising=True면 정배열(짧은 선이 위), False면 역배열(짧은 선이 아래).
    제일 긴 선이 아직 안 섰으면 '안 섰다'로 본다(판정 불가).
    """
    mas = [c.rolling(k).mean() for k in periods]
    ok = pd.Series(True, index=c.index)
    for a, b in zip(mas, mas[1:]):
        ok &= (a > b) if rising else (a < b)
    known = ~mas[-1].isna()
    return ok & known, known


def _state(df15: pd.DataFrame, df4h: pd.DataFrame,
           params: Params) -> pd.DataFrame | None:
    """15m 봉마다 ①②③ 판정 — 못 재면 None.

    정배열은 MA480까지만 보므로 역배열(MA960)보다 일찍 잴 수 있다. 어느
    한쪽이라도 잴 수 있으면 판정한다.
    """
    need = min(max(params.bear_align), max(params.bull_align))
    if len(df15) < need + 2 or df4h is None or len(df4h) < 10:
        return None
    up_top = _band_on_15m(df15, df4h, params.band_top, params.band_mult)
    up_bot = _band_on_15m(df15, df4h, params.band_bottom, params.band_mult)
    if up_top is None or up_bot is None:
        return None
    c = df15["Close"]
    bear, bear_known = _stacked(c, params.bear_align, rising=False)
    bull, _ = _stacked(c, params.bull_align, rising=True)
    d_top = c / up_top - 1
    d_bot = c / up_bot - 1
    n_top = d_top.abs() <= params.near_pct
    n_bot = d_bot.abs() <= params.near_pct
    near = (n_top & n_bot) if params.near_both else (n_top | n_bot)
    rest = (up_top > up_bot) & near
    # 7d = 15m 672봉. 1,000봉을 받으므로 마지막 구간에서는 늘 구할 수 있다.
    d7 = c / c.shift(7 * 96) - 1
    return pd.DataFrame({"close": c, "bear": bear, "bull": bull,
                         "bear_known": bear_known, "up_top": up_top,
                         "up_bot": up_bot, "d_top": d_top, "d_bot": d_bot,
                         "ret_7d": d7, "rest": rest,
                         "ok": (bear | bull) & rest})


def _detail(row, params: Params) -> dict:
    """줄에 실을 값 — 어느 배열로 섰는지, 어느 밴드에 얼마나 붙었는지, 수익률."""
    top_closer = abs(row.d_top) <= abs(row.d_bot)
    d = {"label": LABEL, "interval": INTERVAL,
         "trend": "역배열" if row.bear else "정배열",
         # 두 상단밴드 중 가까운 쪽과 그 거리(종가 ÷ 밴드 − 1)
         "near_band": params.band_top if top_closer else params.band_bottom,
         "band_dist": float(row.d_top if top_closer else row.d_bot),
         "band_top": float(row.up_top), "band_bottom": float(row.up_bot)}
    if pd.notna(row.ret_7d):
        d["ret_7d"] = float(row.ret_7d)
    return d


def _entries(ok: np.ndarray, rearm: int) -> np.ndarray:
    """조건에 **새로 들어온** 봉 — 직전 rearm개 봉에 한 번도 안 서 있었어야 한다.

    ±2% 경계를 드나드는 종목이 많다. 직전 봉만 보면 경계에서 깜빡일 때마다
    새 알림이 나간다 — 실측(현물 221종, 24h)에서 진입이 295회였고, 한 번 나간
    뒤 24h 안의 재진입을 묶으니 65회가 됐다. 묶인 재진입은 추적 줄로 이어진다.
    """
    run = pd.Series(ok).rolling(max(1, rearm), min_periods=1).max().shift(1)
    return ok & ~run.fillna(0).astype(bool).to_numpy()


def detect(df15: pd.DataFrame, df4h: pd.DataFrame, symbol: str,
           params: Params = Params()) -> list[SignalEvent]:
    """조건에 **새로 들어온** 봉을 찾는다(_entries 참고).

    마지막 grace_bars+1개 봉을 후보로 본다 — 스캔이 한 시간 간격이라
    그 사이 지나간 봉도 소급해 잡는다.
    """
    st = _state(df15, df4h, params)
    if st is None:
        return []
    n = len(st)
    new = _entries(st.ok.to_numpy(bool), params.rearm_bars)
    out = []
    for t in range(max(1, n - 1 - params.grace_bars), n):
        if not new[t]:
            continue
        row = st.iloc[t]
        out.append(SignalEvent(symbol=symbol, signal=NAME, bar_time=st.index[t],
                               price=float(row.close), detail=_detail(row, params)))
    return out


def tracking(df15: pd.DataFrame, df4h: pd.DataFrame, symbol: str,
             params: Params = Params()):
    """지금도 조건 안이면 추적 줄용 이벤트 — 아니면 None.

    머문 시간은 **마지막으로 새로 들어온 봉**(_entries)부터 잰다 — rearm 안에
    잠깐 나갔다 들어온 건 같은 한 번으로 본다. 신규로 나가는 창(grace_bars)
    안에서 막 들어온 건은 돌려주지 않는다 — 같은 봉이 두 칸에 실리면 같은
    말을 두 번 한다.
    """
    st = _state(df15, df4h, params)
    if st is None or not st.ok.iloc[-1]:
        return None
    n = len(st)
    new = _entries(st.ok.to_numpy(bool), params.rearm_bars)
    start = int(np.flatnonzero(new).max())
    if start >= n - 1 - params.grace_bars:
        return None                     # 방금 들어왔다 — 신규 줄이 맡는다
    row = st.iloc[-1]
    d = _detail(row, params)
    d["last_price"] = float(row.close)
    d["in_bars"] = int(n - start)       # 새로 들어온 뒤 지난 15m 봉 수
    # 들어오기 전 rearm 창에 **판정할 수 없는 봉**이 있었으면 그보다 오래
    # 머물렀을 수 있다. 1,000봉 중 MA960이 서는 건 마지막 41봉(≈10시간)뿐이라,
    # 역배열 종목은 그 앞을 잴 수가 없어 무조건 '밖'으로 나온다. 밴드 조건은
    # 맞았는데 역배열을 못 잰 봉이 창 안에 있으면 표시를 `+`로 한다.
    lo = max(0, start - max(1, params.rearm_bars))
    unknown = (st.rest & ~st.bear_known).iloc[lo:start]
    d["in_capped"] = bool(start == 0 or unknown.any())
    return SignalEvent(symbol=symbol, signal=NAME, bar_time=st.index[start],
                       price=float(st.close.iloc[start]), detail=d)
