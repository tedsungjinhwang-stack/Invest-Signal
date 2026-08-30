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
            "interval": "4h" if kind == ABC else "1d",
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
    if params.abc_enabled:
        events += _detect_abc(df, symbol, params, rets, vwap_ok)
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


def _detect_abc(df: pd.DataFrame, symbol: str, params: Params,
                rets: dict, vwap_ok) -> list[SignalEvent]:
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
    need = max(params.fast_period, params.slow_period) + 2
    n = len(df)
    if n < need + 1:
        return []
    fast, slow = _trends(df, params)
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
    if len(df):
        event.detail.update(_returns(df))
        q = quiet(df, params)          # 거래대금·변동성은 계속 변한다
        if q is not None:
            event.detail["quiet"] = q


def still_active(df: pd.DataFrame, event: SignalEvent, params: Params = Params()) -> bool:
    """트리거 조건이 그대로 살아 있는 동안만 추적한다.

    두 변형 모두 **장기 수퍼트렌드가 상승으로 뒤집히면 제거**된다 — 그게
    공통 전제였다. 여기에 각자의 조건이 더 붙는다:
      ABC   — 단기가 다시 하락으로 밀리면 반등이 죽은 것이라 제거
      임펄스 — 단기가 상승으로 뒤집히면 그건 더 이상 터치 자리가 아니다
                (그 사건은 ABC 쪽에서 따로 잡힌다)
    """
    frame = (df if event.detail.get("stage") == ABC
             else _daily(df, params.include_live_day))
    if not len(frame):
        return False
    fast, slow = _trends(frame, params)
    last = len(frame) - 1
    if not _down(slow, last):
        return False
    return _up(fast, last) if event.detail.get("stage") == ABC else _down(fast, last)
