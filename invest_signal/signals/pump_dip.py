"""펌핑 눌림 시그널 (4h봉) — 크립토 전용.

수직 펌핑은 60/120선 눌림을 주지 않아 기존 상승초입·눌림목이 못 잡는다.
펌핑 코인의 첫 눌림 진입 자리를 월간 VWAP로 잡는다:
  ① 최근 peak_lookback_bars(기본 42봉 = 7일) 안의 고점이, 고점 직전
     pump_window_bars(기본 6봉 = 하루) 내 저점 대비 min_gain(기본 +30%)
     이상 급등한 고점이고 (= 하루 만에 확 오르는 펌핑)
  ② 그 고점 이후 캔들(저가)이 월간 앵커드 VWAP(MVWAP) 라인에
     "처음" 닿는 봉 → 🚀 알림.

고점 이후 첫 터치 봉에서만 알리고, 새 고점을 만든 뒤 다시 닿으면 새
셋업으로 다시 알린다. Volume이 없으면(MVWAP 계산 불가) 발화하지 않는다.
"""

from dataclasses import dataclass

import pandas as pd

from ..indicators import monthly_vwap
from . import SignalEvent

NAME = "pump_dip"
LABEL = "펌핑"
CRYPTO_ONLY = True          # ETF·주식 스캔에서는 이 시그널을 돌리지 않는다


@dataclass(frozen=True)
class Params:
    peak_lookback_bars: int = 42  # 터치 시점 기준 고점 유효기간 — 42봉 = 7일
    pump_window_bars: int = 6     # 급등 속도 판정 — 고점 직전 N봉(6봉 = 하루)
    min_gain: float = 0.3         # 하루 내 저점→고점 상승률 하한 (0.3 = +30%)
    grace_bars: int = 1           # 직전 실행을 놓쳤을 때 허용할 지각 봉 수


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """마감된 4h OHLC(오름차순, UTC 인덱스)에서 '신선한' 펌핑 눌림을 찾는다.

    마지막 grace_bars+1개 봉을 각각 독립 후보로 판정한다.
    중복 발송 방지는 호출 측(state)이 dedup_key로 처리한다.
    """
    n = len(df)
    if n < params.peak_lookback_bars + 2:
        return []
    wv = monthly_vwap(df)
    if wv is None:
        return []

    close, high, low = df["Close"], df["High"], df["Low"]

    def touches(i: int) -> bool:
        return not pd.isna(wv.iloc[i]) and low.iloc[i] <= wv.iloc[i]

    events = []
    for t in range(max(params.peak_lookback_bars, n - 1 - params.grace_bars), n):
        if not touches(t):
            continue
        # 터치 봉 직전 구간에서 고점을 찾고, 고점 직전 하루의 저점과 비교
        w0 = t - params.peak_lookback_bars
        hi = high.iloc[w0:t]
        peak_off = int(hi.values.argmax())
        peak_idx = w0 + peak_off
        peak = float(hi.iloc[peak_off])
        p0 = max(0, peak_idx - params.pump_window_bars)
        trough = float(low.iloc[p0:peak_idx + 1].min())
        if trough <= 0:
            continue
        gain = peak / trough - 1
        if gain < params.min_gain:
            continue            # ① 하루 내 급등이 아님
        if any(touches(j) for j in range(peak_idx + 1, t)):
            continue            # ② 고점 이후 "첫" 터치 봉만
        events.append(SignalEvent(
            symbol=symbol,
            signal=NAME,
            bar_time=df.index[t],
            price=float(close.iloc[t]),
            detail={
                "label": LABEL,
                "pump_gain": gain,
                "peak": peak,
                "trough": trough,
                "peak_time": df.index[peak_idx].isoformat(),
                "mvwap": float(wv.iloc[t]),
            },
        ))
    return events


def still_active(df: pd.DataFrame, event: SignalEvent, params: Params = Params()) -> bool:
    """터치 후 종가가 월간 VWAP 위에서 버티는 동안 '유지 중' — 아래로
    무너지면 셋업 실패로 리스트에서 뺀다."""
    if event.bar_time not in df.index:
        return False
    wv = monthly_vwap(df)
    if wv is None or pd.isna(wv.iloc[-1]):
        return False
    return bool(df["Close"].iloc[-1] > wv.iloc[-1])
