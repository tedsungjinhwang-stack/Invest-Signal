"""펌핑초기 시그널 (4h봉) — 크립토 전용.

기존 펌핑(pump_dip)이 이미 크게 오른 뒤 월간 VWAP까지 눌린 자리를
잡는다면, 이건 그 앞 단계 — 아직 크게 가지 않은 종목에 막 불이 붙는
봉을 잡는다:
  ① 최근 rise_bars(기본 1봉 = 4시간) 동안 시가 대비 종가가
     min_gain(기본 +5%) 이상 오르고
  ② 최근 lookback_bars(기본 42봉 = 7일) 안에 **하루(pump_window_bars,
     기본 6봉) 내 저점 → 고가가 max_pump_gain(기본 +30%) 이상 급등한
     봉이 없어야** 한다 — pump_dip의 펌핑 판정과 같은 정의라,
     이미 펌핑한 종목은 pump_dip이 눌림 자리에서 담당하고 여기선 빠진다
  ③ 같은 구간 저점 대비 종가 상승률도 max_gain(기본 +30%) 미만 —
     하루 급등은 없었어도 이미 누적으로 크게 간 종목을 거른다

VWAP 조건은 없다 — 급등 시작 포착이라 라인 위/아래를 따지지 않는다.
연속으로 급등하는 구간에서는 첫 봉에서만 알리고, 발생 후에는 종가가
급등 봉의 저가를 지키는 동안 '유지 중'으로 추적한다.
"""

from dataclasses import dataclass

import pandas as pd

from . import SignalEvent

NAME = "pump_early"
LABEL = "펌핑초기"
CRYPTO_ONLY = True          # ETF·주식 스캔에서는 이 시그널을 돌리지 않는다
INTRABAR_OK = True          # 초기 포착이 목적 — 봉 마감을 기다리면 늦는다


@dataclass(frozen=True)
class Params:
    rise_bars: int = 1          # 급등 판정 구간 — 1봉 = 4시간 (2면 8시간)
    min_gain: float = 0.05      # 구간 시가 대비 종가 상승률 하한 (0.05 = +5%)
    lookback_bars: int = 42     # '아직 초기' 판정 구간 — 42봉 = 7일
    pump_window_bars: int = 6   # 이미 펌핑했는지 볼 창 — 6봉 = 하루 (pump_dip과 동일)
    max_pump_gain: float = 0.30  # 하루 내 저점→고가 상승률 상한 — 이상이면 제외
    max_gain: float = 0.30      # 구간 저점 대비 종가 상승률 상한 (누적 기준)
    grace_bars: int = 1         # 직전 실행을 놓쳤을 때 허용할 지각 봉 수


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """4h OHLC(오름차순, UTC 인덱스)에서 '신선한' 펌핑 초기를 찾는다.

    마지막 grace_bars+1개 봉을 각각 독립 후보로 판정한다.
    중복 발송 방지는 호출 측(state)이 dedup_key로 처리한다.
    """
    n = len(df)
    need = max(params.lookback_bars, params.rise_bars)
    if n < need + 2:
        return []

    op, high, close, low = df["Open"], df["High"], df["Close"], df["Low"]

    # 각 봉의 '하루(pump_window_bars봉) 내 저점 → 고가' 상승률 —
    # pump_dip의 펌핑 판정과 같은 창(저점 구간은 판정 봉 포함)
    roll_low = low.rolling(params.pump_window_bars + 1, min_periods=1).min()
    day_gain = high / roll_low.where(roll_low > 0) - 1

    def rise_of(t: int) -> float | None:
        """급등 구간(rise_bars봉) 시가 대비 종가 상승률."""
        r0 = t - params.rise_bars + 1
        if r0 < 0:
            return None
        base = float(op.iloc[r0])
        if base <= 0:
            return None
        return float(close.iloc[t]) / base - 1

    def trough_of(t: int) -> float | None:
        w0 = t - params.lookback_bars
        if w0 < 0:
            return None
        trough = float(low.iloc[w0:t + 1].min())
        return trough if trough > 0 else None

    def had_pump(t: int) -> bool:
        """② 조회 구간 안에 하루 만에 max_pump_gain 이상 급등한 봉이 있었나
        — 있으면 이미 펌핑한 종목이라 '초기'가 아니다(pump_dip 담당)."""
        w0 = max(0, t - params.lookback_bars)
        return bool((day_gain.iloc[w0:t + 1] >= params.max_pump_gain).any())

    def is_trigger(t: int) -> bool:
        """① 급등 + ② 하루 급등 이력 없음 + ③ 누적으로도 아직 안 감."""
        rise = rise_of(t)
        if rise is None or rise < params.min_gain:
            return False
        trough = trough_of(t)
        if trough is None:
            return False
        if had_pump(t):
            return False
        return float(close.iloc[t]) / trough - 1 < params.max_gain

    events = []
    for t in range(max(need, n - 1 - params.grace_bars), n):
        if not is_trigger(t):
            continue
        if is_trigger(t - 1):
            continue            # 연속 급등 구간에서는 첫 봉만
        trough = trough_of(t)
        events.append(SignalEvent(
            symbol=symbol,
            signal=NAME,
            bar_time=df.index[t],
            price=float(close.iloc[t]),
            detail={
                "label": LABEL,
                "rise": rise_of(t),
                "rise_bars": params.rise_bars,
                "total_gain": float(close.iloc[t]) / trough - 1,
                "trough": trough,
                # 급등 봉의 저가 — 이 아래로 마감하면 셋업 실패(still_active)
                "trigger_low": float(low.iloc[t - params.rise_bars + 1:t + 1].min()),
            },
        ))
    return events


def still_active(df: pd.DataFrame, event: SignalEvent, params: Params = Params()) -> bool:
    """급등 봉의 저가를 종가가 지키는 동안 '유지 중' — 아래로 마감하면
    급등이 되돌려진 것이므로 리스트에서 뺀다."""
    if event.bar_time not in df.index:
        return False
    floor_price = event.detail.get("trigger_low")
    if floor_price is None:
        return False
    return bool(float(df["Close"].iloc[-1]) >= float(floor_price))
