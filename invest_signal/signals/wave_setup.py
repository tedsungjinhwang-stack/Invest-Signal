"""파동 시그널 — 크립토 전용. 수퍼트렌드 두 개(단기·장기)로 잡는다.

두 변형이 한 칸에 들어간다. 공통 전제는 **장기 수퍼트렌드가 아직 하락**인
것이고, 단기 수퍼트렌드를 어떻게 건드리는지가 갈린다.

  ⓐ ABC (4h봉, 돌파)
     단기(22×3)·장기(30×6)가 **둘 다 하락**이던 상태에서, 종가가 단기
     추세선을 **돌파해 마감**한 봉 — 그 마감이 곧 단기 수퍼트렌드를 상승
     으로 뒤집는다. 장기는 여전히 하락. 하락 구조 안의 반등(ABC 조정)이
     시작되는 자리를 잡는다.

  ⓑ 임펄스 (일봉, 터치)
     일봉 단기(22×3)·장기(30×6)가 **둘 다 하락**인 채로, 캔들이 단기
     추세선을 **터치**한 날 (저가 ≤ 선 ≤ 고가). 터치만이라 방향은 그대로
     하락이다 — 종가가 선을 넘었다면 그건 돌파(ⓐ)지 터치가 아니다.

두 변형 모두 **마감 판정 전용**이다(INTRABAR_OK 없음). ⓐ는 "돌파 후 마감"이
조건 자체라 진행봉으로 보면 뒤집힐 수 있고, ⓑ는 일봉이라 더 그렇다.

일봉은 따로 받지 않고 스캔이 이미 가진 4h봉을 리샘플해서 쓴다 — 4h봉은
UTC 00시에 맞춰 떨어지므로 6개가 하루가 된다. 아직 6개가 안 찬 마지막
날은 미완성이라 버린다.
"""

from dataclasses import dataclass

import pandas as pd

from ..indicators import pct_over, supertrend_full
from . import SignalEvent

NAME = "wave_setup"
LABEL = "파동"
CRYPTO_ONLY = True          # ETF·주식 스캔에서는 돌리지 않는다
BARS_PER_DAY = 6            # 4h × 6 = 하루

ABC = "ABC"                 # 4h 돌파
IMPULSE = "임펄스"           # 일봉 터치


@dataclass(frozen=True)
class Params:
    fast_period: int = 22       # 단기 수퍼트렌드 ATR 기간
    fast_mult: float = 3.0      # 단기 수퍼트렌드 ATR 배수
    slow_period: int = 30       # 장기 수퍼트렌드 ATR 기간
    slow_mult: float = 6.0      # 장기 수퍼트렌드 ATR 배수
    abc_enabled: bool = True        # ⓐ 4h 돌파
    impulse_enabled: bool = True    # ⓑ 일봉 터치
    grace_bars: int = 1         # 4h 지각 허용 봉 수 (스캔을 놓쳤을 때 소급)
    daily_grace_bars: int = 1   # 일봉 지각 허용 봉 수 — 하루 1봉이라 따로 둔다


def _daily(df: pd.DataFrame) -> pd.DataFrame:
    """4h봉 → 일봉. 아직 6봉이 안 찬 마지막 날은 미완성이라 버린다.

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
    if counts.iloc[-1] < BARS_PER_DAY:
        agg = agg.iloc[:-1]
    return agg


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


def _event(df, symbol, i, kind, fast, slow, params) -> SignalEvent:
    close = df["Close"]
    return SignalEvent(
        symbol=symbol,
        signal=NAME,
        bar_time=df.index[i],
        price=float(close.iloc[i]),
        detail={
            "label": LABEL,
            # stage에 넣으면 dedup 키가 갈린다 — 같은 종목의 4h 돌파와
            # 일봉 터치가 우연히 같은 시각에 떨어져도 서로를 막지 않는다
            "stage": kind,
            "interval": "4h" if kind == ABC else "1d",
            "fast_line": float(fast["line"].iloc[i]),
            "slow_line": float(slow["line"].iloc[i]),
            "fast_st": f"{params.fast_period}×{params.fast_mult:g}",
            "slow_st": f"{params.slow_period}×{params.slow_mult:g}",
            "ret_4h": pct_over(close, 1, i) if kind == ABC else None,
            "ret_24h": pct_over(close, BARS_PER_DAY if kind == ABC else 1, i),
        },
    )


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """마감된 4h OHLC(오름차순, UTC 인덱스)에서 두 변형을 모두 찾는다."""
    events = []
    if params.abc_enabled:
        events += _detect_abc(df, symbol, params)
    if params.impulse_enabled:
        events += _detect_impulse(_daily(df), symbol, params)
    return events


def _detect_abc(df: pd.DataFrame, symbol: str, params: Params) -> list[SignalEvent]:
    """ⓐ 4h — 둘 다 하락이던 상태에서 종가가 단기선을 돌파해 마감한 봉.

    돌파의 정의를 '종가 > 단기선'으로 따로 쓰지 않고 **단기 수퍼트렌드가
    이 봉에서 상승으로 뒤집혔는지**로 본다. 수퍼트렌드는 종가가 상단 밴드
    (=하락 추세선)를 넘길 때 뒤집히므로 둘은 같은 사건이고, 트레일링까지
    반영된 쪽이 선의 정의와 어긋나지 않는다.
    """
    need = max(params.fast_period, params.slow_period) + 2
    n = len(df)
    if n < need + 1:
        return []
    fast, slow = _trends(df, params)
    out = []
    for t in range(max(1, n - 1 - params.grace_bars), n):
        if not (_down(fast, t - 1) and _down(slow, t - 1)):
            continue            # 직전 봉에 둘 다 하락이어야 '둘 다 하락인 시점'
        if not (_up(fast, t) and _down(slow, t)):
            continue            # 이 봉에서 단기만 뒤집히고 장기는 아직 하락
        out.append(_event(df, symbol, t, ABC, fast, slow, params))
    return out


def _detect_impulse(df: pd.DataFrame, symbol: str, params: Params) -> list[SignalEvent]:
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
        if touching(t) and not touching(t - 1):     # 연속 터치는 첫날만
            out.append(_event(df, symbol, t, IMPULSE, fast, slow, params))
    return out


def refresh_detail(df: pd.DataFrame, event: SignalEvent,
                   params: Params = Params()) -> None:
    """추적 줄에 실릴 추세선 값을 **지금 값으로** 갱신한다.

    detect가 붙인 fast_line은 트리거 봉의 값이라, 며칠 지난 셋업에서는
    현재가와 나란히 놓였을 때 말이 안 되는 조합이 된다 — 6일 전 터치한
    종목이 그 사이 크게 빠지면 '현재가 0.033 · 22×3선 0.241' 같은 줄이
    나온다. 추세선은 지금 어디가 무효화 지점인지를 말해 줄 때 쓸모가
    있으므로 매 스캔 다시 계산한다.
    """
    frame = df if event.detail.get("stage") == ABC else _daily(df)
    if not len(frame):
        return
    fast, slow = _trends(frame, params)
    for key, s in (("fast_line", fast), ("slow_line", slow)):
        v = s["line"].iloc[-1]
        if not pd.isna(v):
            event.detail[key] = float(v)


def still_active(df: pd.DataFrame, event: SignalEvent, params: Params = Params()) -> bool:
    """트리거 조건이 그대로 살아 있는 동안만 추적한다.

    두 변형 모두 **장기 수퍼트렌드가 상승으로 뒤집히면 제거**된다 — 그게
    공통 전제였다. 여기에 각자의 조건이 더 붙는다:
      ABC   — 단기가 다시 하락으로 밀리면 돌파가 실패한 것이라 제거
      임펄스 — 단기가 상승으로 뒤집히면 그건 더 이상 터치 자리가 아니다
                (그 사건은 ABC 쪽에서 따로 잡힌다)
    """
    frame = df if event.detail.get("stage") == ABC else _daily(df)
    if not len(frame):
        return False
    fast, slow = _trends(frame, params)
    last = len(frame) - 1
    if not _down(slow, last):
        return False
    return _up(fast, last) if event.detail.get("stage") == ABC else _down(fast, last)
