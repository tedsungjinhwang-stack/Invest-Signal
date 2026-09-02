"""파동 시그널 — 크립토 전용. 수퍼트렌드 두 개(단기·장기)로 잡는다.

두 변형이 한 칸에 들어간다. 공통 전제는 **장기 수퍼트렌드가 아직 하락**인
것이고, 단기 수퍼트렌드를 어떻게 건드리는지가 갈린다.

  ⓐ ABC (4h봉, 돌파·터치 세 자리)
     **장기(30×6)는 하락인 채로 단기(22×3)가 최근 flip_window_bars(12봉
     = 이틀) 안에 상승 전환**했고 지금도 상승인 상태에서, 그 반등이 어느
     선을 어떻게 건드리는지로 갈린다. 하락 구조 안의 반등(ABC 조정)이
     시작되고 저항을 만나는 과정을 세 지점으로 나눠 잡는다:

       장기선 터치 — 반등이 아직 하락인 장기선(상단 밴드)까지 되돌린 자리
       단기선 돌파 — 단기가 상승으로 뒤집힌 봉 자체
       단기선 터치 — 방금 넘긴 선을 다시 눌러 보는 자리(하단 밴드)

     한 봉이 여럿에 걸리면 위 순서로 하나만 남는다 — 저항 도달이 전환
     보다, 전환이 지지 재확인보다 큰 사건이다. 연속으로 닿는 봉은 첫
     봉만. 셋 각각 abc_touch_slow·abc_flip·abc_touch_fast로 끌 수 있다.

     **트리거 봉은 마지막 grace_bars+1개**만 본다(기본 2봉 = 8시간).
     flip_window_bars는 "최근에 전환했나"라는 **전제**를 보는 창이지
     트리거 봉의 범위가 아니다 — 둘을 헷갈리기 쉽다.

  ⓒ 되돌림 (4h봉)
     ⓐ가 3일이면 사라지는데, 그 뒤에 실제로 움직이기 시작하는 종목을
     놓친다. **장기(30×6)가 상승으로 뒤집힌 뒤(=돌파)**, 그 파동의
     저점~고점에서 retrace_level(기본 0.5)까지 처음 밀린 봉을 다시 알린다.

     좌표는 **사건 순서**를 따라 잡는다 — 저점 → 단기 상승 전환 → 장기
     돌파 → 고점:

       고점 = 돌파봉 이후 ~ 판정 봉 직전의 최고가 (러닝 맥스, 창 없음)
       저점 = 그 돌파로 이어진 **단기 상승 전환 봉 앞 W봉**의 최저가
              (W = retrace_window_days × 6)

     저점 창을 고점에 붙이면 안 된다 — 고점이 W봉보다 늦게 서면 창이
     통째로 돌파 뒤로 밀려 랠리 중간 눌림을 저점으로 집는다(실측 판정
     봉의 33.6%). 고점에는 반대로 창을 두면 안 된다 — 진짜 고점이 창
     밖으로 밀리면 📐의 분모가 거짓이 되어 화면 값이 뒷걸음질친다.

     알림은 **한 돌파당 한 번**이고, 그 뒤로는 추적 줄의 📐가 매 스캔
     현재가로 다시 계산된다(단조 상승은 아니다 — 가격이 오르면 📐은
     내려간다). 1.0을 넘으면 저점을 깬 것이라 줄이 빠진다.

     **이건 승률 장치가 아니라 화면 장치다.** 실측에서 되돌림을 기다릴수록
     승자를 더 많이 버린다 — 0.5까지 안 밀려 아예 안 울린 돌파 133건이
     경로 승률 92%·종료 수익 중앙 +9.0%인데, 발화군 725건은 64%·−3.7%다.

  ⓑ 임펄스 (일봉, 터치)
     일봉 단기(22×3)·장기(30×6)가 **둘 다 하락**인 채로, 캔들이 단기
     추세선을 **터치**한 날 (저가 ≤ 선 ≤ 고가). 터치만이라 방향은 그대로
     하락이다 — 종가가 선을 넘었다면 그건 돌파(ⓐ)지 터치가 아니다.

두 변형 모두 **인트라바 스캔에서도 돈다**(매시간). 진행 중인 봉을 잠정
종가로 보고 판정하므로 알림에 ⏳진행봉이 붙고, 봉이 마감될 때 조건이
풀리면 무효일 수 있다. ⓐ의 단기선 돌파("돌파 후 마감")는 그래서 마감 때
되돌려질 수 있는 판정이지만, 그걸 기다리면 최대 4시간이 늦는다.

일봉은 따로 받지 않고 스캔이 이미 가진 4h봉을 리샘플해서 쓴다 — 4h봉은
UTC 00시에 맞춰 떨어지므로 6개가 하루가 된다. 아직 6개가 안 찬 오늘은
기본적으로 버리지만, include_live_day를 켜면(인트라바) 진행 중인 오늘을
포함해 ⓑ의 터치를 그날 안에 잡는다 — 안 그러면 오늘의 터치를 내일 UTC
00시가 지나서야 알게 되어 하루까지 묵는다.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..indicators import anchored_vwap, atr, pct_since, supertrend_full
from . import SignalEvent

NAME = "wave_setup"
LABEL = "파동"
CRYPTO_ONLY = True          # ETF·주식 스캔에서는 돌리지 않는다
INTRABAR_OK = True          # 진행봉을 잠정 종가로 판정 — 알림에 ⏳진행봉 표시
BARS_PER_DAY = 6            # 4h × 6 = 하루

ABC = "ABC"                 # 4h — 최근 상승 전환 후 추세선 터치
SLOW_LINE = "장기선"          # ⓐ가 건드린 선 (detail["touched"])
FAST_LINE = "단기선"
TOUCH = "터치"                # 어떻게 건드렸는지 (detail["kind"])
BREAK = "돌파"
IMPULSE = "임펄스"           # 일봉 터치
RETRACE = "되돌림"           # 4h — 장기 돌파 뒤 저점~고점 되돌림 진입
# 📐이 이 값을 넘으면 저점을 깬 것이라 되돌림이 아니라 이탈이다 — 추적 종료.
# notify._fib_tag이 '저점이탈'로 바꿔 다는 눈금과 같은 값이다.
RETRACE_INVALID = 1.0

# 알림 줄에 붙일 수익률 구간 — 이 시그널에만 있다. 봉 개수가 아니라 시각으로
# 재고(pct_since), 변형과 무관하게 **4h 프레임**에서 뽑는다: 7일 상승률은
# 종목의 성질이지 그 변형이 보는 차트 주기의 성질이 아니다. 일봉으로 재면
# 마지막 완성 봉이 이미 어제라 임펄스만 하루 묵은 값이 된다.
RETURN_SPANS = (("ret_4h", pd.Timedelta(hours=4)),
                ("ret_24h", pd.Timedelta(hours=24)),
                ("ret_7d", pd.Timedelta(days=7)))


def _returns(df4h: pd.DataFrame) -> dict:
    """마지막 4h봉 기준 구간별 수익률. 못 구한 구간은 None(알림에서 빠진다)."""
    close = df4h["Close"]
    return {k: pct_since(close, d) for k, d in RETURN_SPANS}


@dataclass(frozen=True)
class Params:
    fast_period: int = 22       # 단기 수퍼트렌드 ATR 기간
    fast_mult: float = 3.0      # 단기 수퍼트렌드 ATR 배수
    slow_period: int = 30       # 장기 수퍼트렌드 ATR 기간
    slow_mult: float = 6.0      # 장기 수퍼트렌드 ATR 배수
    abc_enabled: bool = True        # ⓐ 4h 장기선 터치
    impulse_enabled: bool = True    # ⓑ 일봉 터치
    # ⓐ 단기가 상승 전환한 뒤 이 봉 수 안에 장기선을 터치해야 한다.
    # 12봉 = 이틀. 핵심은 '최근에 뒤집혔다'이지 '며칠 지났다'가 아니라서
    # 하한은 두지 않는다 — 전환 봉에서 바로 닿아도 인정한다.
    flip_window_bars: int = 12
    # ⓐ 어느 선에 닿는 걸 볼지. 장기선은 위쪽 저항(반등이 저항까지 되돌린
    # 자리), 단기선은 아래쪽 지지(전환한 선을 다시 눌러 보는 자리)다.
    abc_touch_slow: bool = True
    abc_touch_fast: bool = True
    # 전환 봉 자체도 알린다 — 단기선을 뚫고 마감한 순간. 되돌림을 기다리지
    # 않고 그 자리에서 잡고 싶을 때가 있어서 세 번째 트리거로 둔다.
    abc_flip: bool = True
    grace_bars: int = 1         # 4h 지각 허용 봉 수 (스캔을 놓쳤을 때 소급)
    daily_grace_bars: int = 1   # 일봉 지각 허용 봉 수 — 하루 1봉이라 따로 둔다
    include_live_day: bool = False   # 진행 중인 오늘을 일봉에 포함 (인트라바 스캔)
    # 월간 앵커드 VWAP 조건 — 상승초입과 같은 규칙(above/touch/any).
    # 두 변형 모두 **4h 프레임**의 MVWAP으로 판정한다: 같은 선을 보는
    # 조건이고, 일봉으로 다시 계산하면 하루치 대표가격 하나로 뭉쳐져
    # 터치 판정이 무뎌진다. 임펄스는 그날의 마지막 4h봉을 기준점으로 쓴다.
    vwap_condition: bool = True
    vwap_period: str = "M"       # M(월간)/Q(분기)/W(주간)
    vwap_mode: str = "any"       # above / touch / any
    vwap_touch_bars: int = 5     # 터치 소급 창 — 4h 5봉 = 20시간
    # ABC만 추적 보유 기간을 따로 짧게 가져간다(일 단위, 0이면 제한 없음).
    # 스캐너가 넘겨 주는 공용 창(TRACK_DAYS)보다 짧을 때만 효력이 있다 —
    # 돌파는 4h짜리 사건이라 며칠 지나면 '방금 돌파한 자리'가 아니게 된다.
    # 임펄스는 일봉 터치라 성격이 달라 공용 창을 그대로 쓴다.
    abc_track_days: int = 3
    # ⓒ 되돌림 — 장기 돌파 뒤 저점~고점의 되돌림 레벨까지 처음 밀린 자리.
    # 실측(현물 미러 370종·124일, 4h 장기 상승 전환 883건)은 config.yaml
    # 주석에 있다. 요약: 이건 승률 장치가 아니라 **화면 장치**다 —
    # abc_track_days를 3으로 줄이면서 사라지는 줄을 되살리는 쪽이지,
    # 되돌림을 기다려서 더 좋은 자리를 잡는 게 아니다.
    retrace_enabled: bool = True
    retrace_level: float = 0.5      # (고점−현재가)÷(고점−저점). leader_break.retrace와 같은 규약
    retrace_window_days: int = 3    # 저점을 훑는 창(일). 3일 = 18봉. 고점에는 창이 없다
    retrace_hold_bars: int = 2      # 고점이 이 봉 수 이상 갱신되지 않아야 무장
    retrace_expire_bars: int = 42   # 돌파 후 이 봉 수까지만 감시 (42봉 = 7일)
    retrace_track_days: int = 2     # 발화 후 추적 보유 일수 (0이면 공용 창)
    # 🍃조용 태그 기준 — 거르지 않고 표시만 한다(quiet() 참고).
    # ⚡와 같은 축이지만 **기준값은 따로**다: 파동은 하락 추세 종목을 잡아서
    # 변동성 중앙값이 2.9%인데(⚡는 7.3%) ⚡ 기준을 그대로 쓰면 전부 붙는다.
    quiet_turnover_usd: float = 5_000_000   # 24h 거래대금(4h 6봉 종가×거래량)
    quiet_atr_min: float = 0.019            # 너무 조용해도 안 간다 — 아래 하한
    quiet_atr_max: float = 0.044            # 이 위는 위아래로 다 크게 움직인다
    quiet_atr_period: int = 14


def _daily(df: pd.DataFrame, keep_partial: bool = False) -> pd.DataFrame:
    """4h봉 → 일봉. 아직 6봉이 안 찬 마지막 날은 기본적으로 버린다.

    keep_partial=True면 진행 중인 오늘도 남긴다 — 인트라바 스캔에서 오늘의
    터치를 그날 안에 잡기 위한 것이고, 그 봉의 고·저는 아직 자라는 중이라
    판정이 뒤집힐 수 있다(알림에 ⏳진행봉이 붙는다).

    중간에 결측이 있는 날은 그대로 둔다 — 거기서 하루를 통째로 빼면
    일봉 인덱스에 구멍이 생겨 수퍼트렌드가 엉뚱한 봉을 이웃으로 본다.
    """
    if not len(df):
        return df
    agg = df.resample("1D").agg({"Open": "first", "High": "max", "Low": "min",
                                 "Close": "last", "Volume": "sum"}).dropna()
    if not len(agg):
        return agg
    counts = df["Close"].resample("1D").count().reindex(agg.index)
    if not keep_partial and counts.iloc[-1] < BARS_PER_DAY:
        agg = agg.iloc[:-1]
    return agg


def _vwap_gate(df: pd.DataFrame, params: Params):
    """4h 프레임 기준 VWAP 조건 판정기 — `ok(i)` 하나를 돌려준다.

    상승초입과 같은 정의다:
      above — 그 봉 종가가 VWAP 위
      touch — 최근 vwap_touch_bars봉 안에 VWAP이 캔들 범위 안(저가 ≤ 선 ≤ 고가)
      any   — 둘 중 하나
    조건을 껐거나 VWAP을 못 구하는 구간은 통과시킨다.
    """
    if not params.vwap_condition:
        return lambda i: True
    vw = anchored_vwap(df, params.vwap_period)
    if vw is None:
        return lambda i: True
    high, low, close = df["High"], df["Low"], df["Close"]

    def touched(i: int) -> bool:
        for j in range(max(0, i - params.vwap_touch_bars + 1), i + 1):
            if not pd.isna(vw.iloc[j]) and low.iloc[j] <= vw.iloc[j] <= high.iloc[j]:
                return True
        return False

    def ok(i: int) -> bool:
        if i < 0 or i >= len(df) or pd.isna(vw.iloc[i]):
            return True
        above = bool(close.iloc[i] > vw.iloc[i])
        if params.vwap_mode == "above":
            return above
        if params.vwap_mode == "touch":
            return touched(i)
        return above or touched(i)          # any

    return ok


def quiet(df: pd.DataFrame, params: Params = Params(), at: int = -1) -> bool | None:
    """'조용한' 자리인지 — 거래대금이 작고 변동성이 적당한 구간.

    **거르는 조건이 아니라 표시용이다.** 30일 리플레이(퍼프 281건,
    `scripts/debug_probe.py`의 PROBE_MODE=leaders + PROBE_REPLAY=1)에서
    이 시그널도 ⚡와 같은 모양이었다 — 이미 크게 오르고 변동성이 터진
    종목이 대박(7일 내 고점 +20%↑)과 쪽박(저점 −15%↓)을 다 만든다.
    24h 상승률 상위 25%는 대박 42%에 쪽박 51%, 변동성 상위 25%는 대박
    44%에 쪽박 54%였다.

    갈라지는 축도 같다. 24h 거래대금 하위 25%($5.2M 미만)에서 대박률은
    24%→27%로 **오히려 오르고** 쪽박률은 21%→7%로 떨어졌다(경로 승률 64%).
    ⚡에선 대박률이 조금 깎였는데 여기선 안 깎인다.

    변동성은 상한만 두지 않고 **밴드**로 본다. 하위 25%(1.9% 미만)는 대박
    6%·쪽박 1%로 안 깨지지만 안 가기도 했다 — 2.9~4.4% 구간이 대박 37%·
    쪽박 21%로 가장 좋았다.

    거래대금은 티커가 아니라 4h 캔들에서 구한다(종가×거래량 6봉 합) —
    기준값을 그 방식으로 쟀고, 과거 봉에도 같은 값을 매길 수 있어야 한다.
    at은 판정할 봉 위치다(기본 마지막 봉).

    판단할 값이 없으면 None — 조용하다고도, 아니라고도 하지 않는다.
    """
    n = len(df)
    if not n:
        return None
    i = at if at >= 0 else n + at
    if i < params.quiet_atr_period or i >= n:
        return None
    seg = df.iloc[max(0, i - BARS_PER_DAY + 1):i + 1]
    turnover = float((seg["Close"] * seg["Volume"]).sum())
    a = atr(df.iloc[:i + 1], params.quiet_atr_period).iloc[-1]
    close = float(df["Close"].iloc[i])
    if pd.isna(a) or close <= 0:
        return None
    vol = float(a) / close
    return bool(turnover < params.quiet_turnover_usd
                and params.quiet_atr_min <= vol < params.quiet_atr_max)


def _trends(df: pd.DataFrame, params: Params):
    fast = supertrend_full(df, params.fast_period, params.fast_mult)
    slow = supertrend_full(df, params.slow_period, params.slow_mult)
    return fast, slow


def _down(s: pd.DataFrame, i: int) -> bool:
    v = s["dir"].iloc[i]
    return not pd.isna(v) and v < 0


def _up(s: pd.DataFrame, i: int) -> bool:
    v = s["dir"].iloc[i]
    return not pd.isna(v) and v > 0


def _event(df, symbol, i, kind, fast, slow, params, rets) -> SignalEvent:
    close = df["Close"]
    return SignalEvent(
        symbol=symbol,
        signal=NAME,
        bar_time=df.index[i],
        price=float(close.iloc[i]),
        detail={
            "label": LABEL,
            # stage에 넣으면 dedup 키가 갈린다 — 같은 종목의 4h 터치와
            # 일봉 터치가 우연히 같은 시각에 떨어져도 서로를 막지 않는다
            "stage": kind,
            "interval": "4h" if kind in (ABC, RETRACE) else "1d",
            "fast_line": float(fast["line"].iloc[i]),
            "slow_line": float(slow["line"].iloc[i]),
            "fast_st": f"{params.fast_period}×{params.fast_mult:g}",
            "slow_st": f"{params.slow_period}×{params.slow_mult:g}",
            **rets,
        },
    )


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """마감된 4h OHLC(오름차순, UTC 인덱스)에서 두 변형을 모두 찾는다."""
    events = []
    if not len(df):
        return events
    rets = _returns(df)
    vwap_ok = _vwap_gate(df, params)
    # 4h 수퍼트렌드는 ⓐ·ⓒ가 같은 걸 본다 — 370종×매 스캔이라 한 번만 돌린다
    need = max(params.fast_period, params.slow_period) + 2
    fast = slow = None
    if len(df) >= need + 1 and (params.abc_enabled or params.retrace_enabled):
        fast, slow = _trends(df, params)
    if params.abc_enabled and fast is not None:
        events += _detect_abc(df, symbol, params, rets, vwap_ok, fast, slow)
    if params.retrace_enabled and fast is not None:
        events += _detect_retrace(df, symbol, params, rets, vwap_ok, fast, slow)
    if params.impulse_enabled:
        daily = _daily(df, params.include_live_day)
        # 일봉 트리거를 4h 프레임 위치로 옮긴다 — 그날의 마지막 4h봉이 기준점
        def day_ok(day) -> bool:
            return vwap_ok(_at_4h(df, day))
        events += _detect_impulse(daily, symbol, params, rets, day_ok)
    # 🍃조용 판정은 변형과 무관하게 **4h 프레임**에서 한다 — 일봉으로 재면
    # 거래대금 창도 변동성 눈금도 달라져 두 변형을 같은 잣대로 못 본다.
    for ev in events:
        i = (_at_4h(df, ev.bar_time) if ev.detail.get("stage") == IMPULSE
             else int(df.index.searchsorted(ev.bar_time, "left")))
        q = quiet(df, params, i)
        if q is not None:
            ev.detail["quiet"] = q
    return events


def _at_4h(df: pd.DataFrame, day) -> int:
    """그날의 마지막 4h봉 위치 — 일봉 사건을 4h 프레임에 얹을 때 쓴다."""
    return int(df.index.searchsorted(day + pd.Timedelta(days=1), "left")) - 1


def _detect_abc(df: pd.DataFrame, symbol: str, params: Params, rets: dict,
                vwap_ok, fast: pd.DataFrame, slow: pd.DataFrame) -> list[SignalEvent]:
    """ⓐ 4h — 단기가 **최근에** 상승 전환했고, 캔들이 추세선을 터치한 봉.

    단기 수퍼트렌드가 하락에서 상승으로 뒤집히는 순간과, 그 뒤 반등이 어느
    선을 건드리는지를 본다. 세 자리 다 알린다:

      **단기선 돌파**(abc_flip) — 단기가 상승으로 뒤집힌 봉 자체. 종가가
        단기 추세선을 넘겨 마감한 자리이고, 되돌림을 기다리지 않는다.
      **장기선 터치**(위쪽 저항) — 반등이 아직 하락인 장기선까지 되돌린 자리.
      **단기선 터치**(아래쪽 지지) — 방금 넘긴 선을 다시 눌러 보는 자리.

    장기가 하락이면 장기선은 상단 밴드라 저항이고, 종가가 그 선을 넘겼다면
    장기도 뒤집혀 여기 안 걸린다 — '터치'와 '돌파'는 자동으로 갈린다
    (임펄스와 같은 정의). 단기선은 반대로 방향이 상승이라 하단 밴드다.

    '최근'은 flip_window_bars(기본 12봉 = 이틀)다. 하한은 없다 — 전환 봉에서
    바로 장기선까지 닿았어도 조건은 성립한다. 연속으로 닿는 봉은 첫 봉만.

    예전 정의는 '전환 봉 자체'에서 알렸다. 그 자리는 아직 저항을 만나기
    전이라, 한 단계 뒤인 이 자리로 옮겼다.
    """
    n = len(df)
    high, low = df["High"], df["Low"]

    def flipped_recently(i: int) -> bool:
        """i봉 기준 최근 flip_window_bars 안에 단기가 상승 전환했는지."""
        if not _up(fast, i):
            return False
        for j in range(i, max(0, i - params.flip_window_bars) - 1, -1):
            if j >= 1 and _up(fast, j) and _down(fast, j - 1):
                return True     # 전환 봉을 창 안에서 찾았다
            if j < i and not _up(fast, j):
                return False    # 창 안에서 이미 하락으로 끊겼다
        return False

    def touching(i: int, st) -> bool:
        if i < 0 or not (_down(slow, i) and flipped_recently(i)):
            return False
        line = st["line"].iloc[i]
        return not pd.isna(line) and low.iloc[i] <= line <= high.iloc[i]

    def flipping(t: int) -> bool:
        """이 봉에서 단기가 상승으로 뒤집혔는지 — 장기는 아직 하락."""
        return _up(fast, t) and _down(fast, t - 1) and _down(slow, t)

    # 한 봉이 여러 조건에 걸리면 앞의 것 하나만 남는다 — dedup 키가
    # (종목·봉·stage)라 어차피 하나뿐이고, 순서가 곧 우선순위다.
    # 장기선 터치(저항 도달) > 단기선 돌파(전환) > 단기선 터치(지지 확인).
    checks = []
    if params.abc_touch_slow:
        checks.append((SLOW_LINE, TOUCH,
                       lambda t: touching(t, slow) and not touching(t - 1, slow)))
    if params.abc_flip:
        checks.append((FAST_LINE, BREAK, flipping))
    if params.abc_touch_fast:
        checks.append((FAST_LINE, TOUCH,
                       lambda t: touching(t, fast) and not touching(t - 1, fast)))

    look = params.grace_bars
    if params.abc_track_days > 0:
        look = min(look, params.abc_track_days * BARS_PER_DAY)
    out = []
    for t in range(max(1, n - 1 - look), n):
        if not vwap_ok(t):
            continue
        for name, kind, hit in checks:
            # 연속 터치는 첫 봉만 — 선마다 따로 센다. 장기선을 며칠 훑고 있는
            # 중에 단기선을 새로 누르면 그건 별개의 사건이다.
            if hit(t):
                e = _event(df, symbol, t, ABC, fast, slow, params, rets)
                e.detail["touched"] = name
                e.detail["kind"] = kind
                out.append(e)
                break
    return out


@dataclass(frozen=True)
class Wave:
    """장기 돌파 한 건의 되돌림 좌표. 봉 위치는 df 인덱스 기준 정수."""
    break_i: int    # 장기가 상승으로 뒤집힌 봉 (= 돌파)
    low: float      # 저점 — 단기 상승 전환 봉 앞 W봉의 최저가 (돌파보다 앞)
    low_i: int
    high: float     # 고점 — 돌파봉 이후 ~ 판정 봉 직전의 최고가 (러닝 맥스, 창 없음)
    high_i: int
    level: float    # high - retrace_level × (high - low)


def _break_index(slow: pd.DataFrame) -> np.ndarray:
    """봉마다 '지금 살아 있는 장기 상승 구간이 시작된 봉'. 없으면 -1.

    상승 전환은 **하락에서** 뒤집힌 것만 센다(`dir[i-1] < 0`). ATR 워밍업
    구간의 dir은 NaN이라 'not up'으로 세면 워밍업이 끝나는 봉마다 종목당
    한 건씩 유령 돌파가 생긴다.
    """
    d = slow["dir"].to_numpy(float)
    out = np.full(len(d), -1, dtype=int)
    cur = -1
    for i in range(len(d)):
        if d[i] > 0:
            if i >= 1 and d[i - 1] < 0:
                cur = i
            out[i] = cur
        else:
            cur = -1
    return out


def _flip_index(fast: pd.DataFrame) -> np.ndarray:
    """봉마다 '그 봉 이하에서 가장 최근의 단기 상승 전환 봉'. 없으면 -1.

    _break_index와 같은 이유로 하락→상승만 센다(`dir[i-1] < 0`) — 워밍업
    NaN을 하락으로 세면 워밍업이 끝나는 봉이 전환으로 잡힌다.
    """
    d = fast["dir"].to_numpy(float)
    out = np.full(len(d), -1, dtype=int)
    last = -1
    for i in range(len(d)):
        if i >= 1 and d[i] > 0 and d[i - 1] < 0:
            last = i
        out[i] = last
    return out


def wave_at(df: pd.DataFrame, i: int, params: Params = Params(),
            fast: pd.DataFrame | None = None, slow: pd.DataFrame | None = None,
            idx: tuple[np.ndarray, np.ndarray] | None = None) -> Wave | None:
    """i봉 시점에서 본 '살아 있는 장기 돌파'의 되돌림 좌표. 없으면 None.

    살아 있다 = i봉에서 장기가 상승이고, 그 상승이 시작된 돌파봉이
    retrace_expire_bars 안에 있다. 돌파봉 자체(i == break_i)는 아직
    되돌릴 게 없으므로 제외한다.

    좌표는 **사건 순서**를 따라 잡는다: 저점 → 단기 상승 전환 → 장기
    돌파 → 고점.

      고점 = 돌파봉 이후 ~ 판정 봉 직전의 최고가 (러닝 맥스, **창 없음**)
      저점 = 돌파로 이어진 단기 상승 전환 봉 f0 앞 W봉의 최저가
             (W = retrace_window_days × 6)

    **저점 창은 f0에 붙인다.** 고점에 붙이면 고점이 늦게 설 때 창이 통째로
    돌파 뒤로 밀려 랠리 중간 눌림을 저점으로 집는다 — 실측에서 판정 봉의
    33.6%, 발화의 11.6%가 그렇게 순서를 어겼다. f0에 붙이면 정의상 100%다.

    **고점에는 창을 두지 않는다.** 롤링 창을 씌우면 진짜 고점이 창 밖으로
    밀리는 순간 고점이 낮아져(판정 봉의 28.9%, 중앙 −5.4%) 📐의 분모가
    거짓이 되고 화면 값이 뒷걸음질친다. 오래된 고점의 만료는
    retrace_expire_bars가 이미 처리한다.

    **판정 봉 i의 고가는 고점에서 뺀다.** 넣으면 봉이 자라는 동안 고점이
    같이 올라가 레벨이 내려오고, 그 봉의 저가가 자기 레벨에 걸린다.
    """
    n = len(df)
    if i < 1 or i >= n:
        return None
    if idx is None:
        if slow is None:
            slow = supertrend_full(df, params.slow_period, params.slow_mult)
        if fast is None:
            fast = supertrend_full(df, params.fast_period, params.fast_mult)
        idx = (_break_index(slow), _flip_index(fast))
    brk, flip = idx
    b = int(brk[i])
    if b < 0 or not 1 <= i - b <= params.retrace_expire_bars:
        return None
    highs = df["High"].to_numpy(float)[b:i]
    k = int(highs.argmax())
    high_i, high = b + k, float(highs[k])
    f0 = int(flip[b])
    if f0 < 0:                      # 프레임 안에 단기 전환이 없다 — 돌파봉으로 폴백
        f0 = b
    w = max(1, params.retrace_window_days * BARS_PER_DAY)
    lo_start = max(0, f0 - w + 1)
    lows = df["Low"].to_numpy(float)[lo_start:f0 + 1]
    m = int(lows.argmin())
    low_i, low = lo_start + m, float(lows[m])
    if not high > low:
        return None
    return Wave(b, low, low_i, high, high_i,
                high - params.retrace_level * (high - low))


def _detect_retrace(df: pd.DataFrame, symbol: str, params: Params, rets: dict,
                    vwap_ok, fast: pd.DataFrame, slow: pd.DataFrame) -> list[SignalEvent]:
    """ⓒ 4h — 장기 돌파 뒤 저점~고점의 retrace_level까지 **처음** 밀린 봉.

    한 돌파당 한 번만 알린다. 존을 나갔다 다시 들어와도, 새 고점이 서서
    레벨이 올라가도 재발화하지 않는다 — 실측에서 재발화를 허용하면 신규
    알림이 6.0건/일에서 29건/일로 튄다. 되돌림이 더 깊어지는 건 추적 줄의
    📐가 매 스캔 갱신하며 보여 준다.

    고점 확정(retrace_hold_bars)이 이 시그널의 유일한 실질 레버다: 경로
    승률이 어느 저점 정의에서도 +5~8%p 오르는데 발화율은 1~3%p만 깎인다.
    레벨은 볼륨 다이얼에 가깝다.
    """
    n = len(df)
    idx = (_break_index(slow), _flip_index(fast))
    lows = df["Low"].to_numpy(float)
    memo: dict[int, Wave | None] = {}

    def in_zone(t: int) -> Wave | None:
        """t봉이 그 시점 좌표의 되돌림 존 안인지 — 맞으면 그 좌표."""
        if t not in memo:
            w = wave_at(df, t, params, idx=idx)
            if w is not None and (t - w.high_i < params.retrace_hold_bars
                                  or lows[t] > w.level):
                w = None
            memo[t] = w
        return memo[t]

    look = params.grace_bars
    if params.retrace_track_days > 0:
        look = min(look, params.retrace_track_days * BARS_PER_DAY)
    out = []
    for t in range(max(1, n - 1 - look), n):
        w = in_zone(t)
        if w is None or not vwap_ok(t):
            continue
        prev = (in_zone(j) for j in range(w.break_i + 1, t))
        if any(z is not None and z.break_i == w.break_i for z in prev):
            continue
        e = _event(df, symbol, t, RETRACE, fast, slow, params, rets)
        e.detail["wave_low"] = w.low
        e.detail["wave_high"] = w.high
        e.detail["wave_level"] = w.level        # 트리거 시점 레벨 — 박제한다
        e.detail["break_time"] = df.index[w.break_i].isoformat()
        e.detail["fib"] = _fib(w.low, w.high, float(df["Close"].iloc[t]))
        out.append(e)
    return out


def _fib(low: float, high: float, price: float) -> float:
    """(고점−현재가)÷(고점−저점) — leader_break.retrace()와 같은 규약."""
    return (high - price) / (high - low)


def _detect_impulse(df: pd.DataFrame, symbol: str, params: Params,
                    rets: dict, vwap_ok) -> list[SignalEvent]:
    """ⓑ 일봉 — 둘 다 하락인 채로 캔들이 단기선을 터치한 날.

    방향이 하락이면 단기선은 상단 밴드다. 종가가 그 선을 넘었다면 방향이
    뒤집혀 여기 걸리지 않으므로, '터치'와 '돌파'는 자동으로 갈린다.
    연속으로 닿는 날은 첫날만 알린다.
    """
    need = max(params.fast_period, params.slow_period) + 2
    n = len(df)
    if n < need + 1:
        return []
    fast, slow = _trends(df, params)
    high, low = df["High"], df["Low"]

    def touching(i: int) -> bool:
        if i < 0 or not (_down(fast, i) and _down(slow, i)):
            return False
        line = fast["line"].iloc[i]
        return not pd.isna(line) and low.iloc[i] <= line <= high.iloc[i]

    # 스캐너는 '유지 중'을 찾을 때 grace_bars를 4h봉 기준으로 넓혀서 부른다.
    # 그 폭을 일수로 환산해 여기에도 반영해야 임펄스 추적이 다른 시그널과
    # 같은 기간을 산다 — 안 그러면 daily_grace_bars(1)에 묶여 이틀만 남는다.
    #
    # 환산에서 1을 빼는 이유: 아래 범위가 look+1개 봉을 보는데, 일봉은 완성
    # 시점이 이미 하루 지난 봉이라 그대로 두면 추적 목록에 TRACK_DAYS보다
    # 하루 많은 '6d' 줄이 남는다. 기본값(grace_bars=1)에서는 음수가 되므로
    # daily_grace_bars가 이겨서 지각 1봉 허용이 그대로 유지된다.
    look = max(params.daily_grace_bars, params.grace_bars // BARS_PER_DAY - 1)
    out = []
    for t in range(max(1, n - 1 - look), n):
        if not (touching(t) and not touching(t - 1)):    # 연속 터치는 첫날만
            continue
        if vwap_ok(df.index[t]):
            out.append(_event(df, symbol, t, IMPULSE, fast, slow, params, rets))
    return out


def refresh_detail(df: pd.DataFrame, event: SignalEvent,
                   params: Params = Params()) -> None:
    """추적 줄에 실릴 수익률을 **지금 값으로** 갱신한다.

    detect가 붙인 값은 트리거 봉 기준이라 며칠 지난 셋업이면 며칠 전
    수익률이 현재가 옆에 박제된다. 알림에 실리는 4h·24h·7d는 매 스캔
    다시 계산한다.

    추세선(fast_line·slow_line)은 갱신하지 않는다 — 알림에 싣지 않으므로
    종목마다 수퍼트렌드를 두 번 더 돌릴 이유가 없다. detail에 남아 있는
    값은 트리거 시점 기록이다.
    """
    if not len(df):
        return
    event.detail.update(_returns(df))
    q = quiet(df, params)              # 거래대금·변동성은 계속 변한다
    if q is not None:
        event.detail["quiet"] = q
    if event.detail.get("stage") == RETRACE:
        # ⓒ의 📐는 매 스캔 현재가로 다시 잰다 — 되돌림이 얼마나 깊어졌는지가
        # 이 줄의 유일한 라이브 정보다. 저장 좌표(wave_low/high/level)는
        # 트리거 시점 기록이라 갱신하지 않는다.
        lo, hi = event.detail.get("wave_low"), event.detail.get("wave_high")
        if lo is not None and hi is not None:
            # 고점은 러닝 맥스라 트리거 뒤에도 자란다. 박제한 값만 쓰면
            # 신고점을 낸 종목에서 분자가 음수가 되어 `📐-0.31`이 그대로
            # 화면에 찍힌다(실측: 추적 스텝의 3.0%, 최대 10봉 연속).
            # **마지막 봉을 포함**해야 한다 — 종가를 그 봉에서 가져오므로
            # hi ≥ High[last] ≥ Close[last]가 되어 📐 ≥ 0이 정의상 보장된다.
            t = int(df.index.searchsorted(event.bar_time, "left"))
            if t < len(df):
                hi = max(hi, float(df["High"].to_numpy(float)[t:].max()))
            if hi > lo:
                event.detail["fib"] = _fib(lo, hi, float(df["Close"].iloc[-1]))


def still_active(df: pd.DataFrame, event: SignalEvent, params: Params = Params()) -> bool:
    """트리거 조건이 그대로 살아 있는 동안만 추적한다.

    ⓐ·ⓑ는 **장기 수퍼트렌드가 상승으로 뒤집히면 제거**된다 — 그게 둘의
    공통 전제였다. 여기에 각자의 조건이 더 붙는다:
      ABC   — 단기가 다시 하락으로 밀리면 반등이 죽은 것이라 제거
      임펄스 — 단기가 상승으로 뒤집히면 그건 더 이상 터치 자리가 아니다
                (그 사건은 ABC 쪽에서 따로 잡힌다)

    ⓒ 되돌림은 장기 방향 요구가 **정반대**라 분기를 먼저 태운다. 그래서
    한 종목이 같은 봉에 ⓐ와 ⓒ를 동시에 갖는 일은 구조적으로 없다 —
    스캐너가 심볼당 최신 1건만 남기므로 돌파 순간 ⓐ 줄이 ⓒ 줄로 교대된다.
    """
    stage = event.detail.get("stage")
    frame = (df if stage in (ABC, RETRACE)
             else _daily(df, params.include_live_day))
    if not len(frame):
        return False
    fast, slow = _trends(frame, params)
    last = len(frame) - 1
    if stage == RETRACE:
        # 장기가 다시 하락으로 꺾이면 돌파가 실패한 것이라 좌표도 같이
        # 죽는다. 저점을 깨면(📐 1.0 초과) 되돌림이 아니라 이탈이라 거기서도
        # 끝낸다.
        #
        # **📐 ≥ 1.0 ⟺ 종가 ≤ 저점이다** (high−price ≥ high−low ⟺ price ≤ low).
        # 고점이 판정에 안 들어가고, 이 정의의 저점은 트리거 시점에 이미 전부
        # 과거 봉이라 안 변한다 — 그래서 좌표를 박제하든 매 스캔 다시 재든
        # 이탈 판정이 봉 단위로 완전히 같다(실측 차이 0건).
        if not _up(slow, last):
            return False
        lo, hi = event.detail.get("wave_low"), event.detail.get("wave_high")
        if lo is None or hi is None or not hi > lo:
            return True
        return _fib(lo, hi, float(frame["Close"].iloc[last])) < RETRACE_INVALID
    if not _down(slow, last):
        return False
    return _up(fast, last) if stage == ABC else _down(fast, last)
