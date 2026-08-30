"""크립토 모멘텀 눌림목/이탈 시그널 (15m봉) — 크립토 선물 전용.

24시간 상승률 **상위 top_n종** 중 **1h가 정배열인 종목**만 대상으로,
15분봉 종가가 ma(기본 20)선을 **하향 이탈하는 첫 봉**에서 알린다.
하루 사이 가장 많이 오른 주도주가 단기 추세선을 깨는 자리를 잡는다.
제외 규칙은 blocked() 참고.

다른 시그널과 두 가지가 다르다:
  · 4h봉이 아니라 **15m봉**으로 판정한다 (20봉 = 5시간).
  · 종목 선정이 횡단면 순위라 스캐너가 대상 top_n종을 먼저 고른 뒤
    이 모듈의 detect를 종목별로 부른다.

'첫 봉'만 알리므로 선 아래에 머무는 동안 반복 알림은 없다. 선 위로
복귀했다가 다시 깨면 새 이탈로 다시 알린다.
"""

from dataclasses import dataclass

import pandas as pd

from ..indicators import alignment, atr, pct_over, sma, supertrend_full
from . import SignalEvent

NAME = "leader_break"
LABEL = "크립토 모멘텀 눌림목/이탈"
CRYPTO_ONLY = True      # ETF·주식 스캔에서는 돌리지 않는다
INTERVAL = "15m"
KLINE_LIMIT = 200       # ma + grace + 여유 (ma를 키워도 넉넉하도록)
ALIGN_INTERVAL = "1h"   # blocked() 구조 판정용 — 종목마다 따로 받는다
ALIGN_LIMIT = 600       # MA480 성립(480봉 = 20일) + 여유
TREND_INTERVAL = "4h"   # quiet()·turn_up() 판정용 — 스캔이 받아둔 프레임을 쓴다
TREND_LIMIT = 600


@dataclass(frozen=True)
class Params:
    top_n: int = 5              # 24h 상승률 상위 몇 종을 볼지
    board_top: int = 5          # 알림 맨 위에 조건 없이 적어줄 상위 종목 수
    ma: int = 20                # 15m봉 SMA 기간 (20봉 = 5시간)
    grace_bars: int = 4         # 소급 판정 봉 수 — 15m×4 = 1시간(스캔 주기)
    min_turnover_usd: float = 1_000_000   # 24h 거래대금 하한(유동성)
    watch_days: int = 5         # 상위권에서 밀려난 뒤에도 계속 볼 기간
    max_watch: int = 60         # 한 스캔에서 15m 캔들을 받을 최대 종목 수
    track_break_only: bool = True   # 추적도 '20선 아래'인 동안만 (복귀하면 제외)
    exhausted_filter: bool = True        # 아래 blocked() 조건에 걸리면 대상에서 제외
    require_aligned: bool = True         # 1h 배열을 보는지 (끄면 배열 무관 전부 통과)
    allow_bearish: bool = False          # 켜면 역배열도 통과 (혼조는 어느 쪽이든 제외)
    allow_short_history: bool = False    # 이력이 짧아 배열 판정 불가면 통과시킬지
    exhausted_mas: tuple = (120, 240, 480)   # 1h봉 정배열 판정선
    # ② 4h 장기 수퍼트렌드가 상승이어야 한다 — 선은 turn_slow_* 를 같이 쓴다
    # '조용한' 종목 표시 기준 — 거르지 않고 태그만 붙인다(quiet() 참고)
    quiet_turnover_usd: float = 6_000_000    # 24h 거래대금이 이 아래이고
    quiet_atr_pct: float = 0.073             # 4h ATR÷종가가 이 아래면 조용
    quiet_atr_period: int = 14
    # 📐 되돌림 표시 — 1h 파동의 저점·고점 대비 지금 어디까지 밀렸는지.
    # 프레임은 구조 판정과 같은 것을 쓴다(ALIGN_INTERVAL/ALIGN_LIMIT).
    fib_enabled: bool = True
    fib_fast_period: int = 22       # 단기 수퍼트렌드 — 파동의 시작·끝
    fib_fast_mult: float = 3.0
    fib_slow_period: int = 30       # 장기 수퍼트렌드 — 저점 구간의 시작
    fib_slow_mult: float = 6.0
    fib_min: float = 0.5            # 이 아래면 표시하지 않는다
    # 🔼 단기전환 — 4h 장기가 상승인 채로 단기가 하락→상승으로 뒤집힌 자리.
    # 눌림이 끝나고 다시 붙는 지점이다(되돌림이 '얼마나'라면 이건 '언제').
    turn_enabled: bool = True
    turn_fast_period: int = 22
    turn_fast_mult: float = 3.0
    turn_slow_period: int = 30
    turn_slow_mult: float = 6.0
    turn_bars: int = 3              # 전환·터치 후 이 봉 수까지 표시 (3 × 4h = 12시간)
    turn_touch: bool = True         # 단기선 터치도 같이 표시할지
    # ↗️ 1h 저항 테스트 — 위 🔼와 정반대 국면. 1h 단기선이 **위에** 있고
    # 캔들이 아래에서 올라와 그 선을 건드리거나 뚫는 자리다(resist_test()).
    # 프레임은 구조 판정과 같은 것을 쓴다(ALIGN_INTERVAL/ALIGN_LIMIT).
    turn1h_enabled: bool = True
    turn1h_bars: int = 24           # 터치·돌파 후 이 봉 수까지 표시 (24시간)
    turn1h_require_bearish: bool = False  # 1h 장기도 하락이어야 하는지


def leaders(ticker: dict[str, dict], symbols: set[str],
            params: Params = Params()) -> list[tuple[str, dict]]:
    """24h 상승률 상위 top_n종 — (심볼, 통계) 목록을 상승률 내림차순으로.

    symbols(스캔 유니버스)에 있고 거래대금 하한을 넘는 종목만 후보다.
    하한이 없으면 상위권이 거래가 거의 없는 잡코인으로 채워진다.
    """
    cand = [(s, t) for s, t in ticker.items()
            if s in symbols and t["quote_volume"] >= params.min_turnover_usd]
    cand.sort(key=lambda kv: kv[1]["change_pct"], reverse=True)
    return cand[:max(0, params.top_n)]


def watch_list(top: list[tuple[str, dict]], recent: dict, symbols: set[str],
               params: Params = Params(),
               ticker: dict[str, dict] | None = None) -> list[tuple[str, dict | None]]:
    """이번 스캔에서 15m을 확인할 종목 — 현재 상위권 + 최근 상위권 잔류분.

    상위권은 하루에도 여러 번 바뀌므로, 한 번 뽑힌 종목이 순위에서 밀렸다고
    바로 눈을 떼면 정작 꺾이는 순간을 놓친다. recent(마지막 등재 시각)에
    남아 있는 종목을 최근 등재 순으로 이어 붙이되, 매 스캔 캔들을 받아야
    하므로 max_watch로 총량을 묶는다. 반환값의 두 번째 항목은 마지막 등재
    시각(현재 상위권이면 None)이라 호출 측이 '추적 N일차'를 붙일 수 있다.

    **이월분에도 거래대금 하한을 적용한다.** 상위권에 들 때는 하한을
    넘었어도 그 뒤 거래가 말라붙는 종목이 많은데, 그대로 두면 max_watch
    슬롯을 차지해 더 최근 상위권 종목을 밀어낸다. ticker를 안 주면
    (테스트 등) 하한 검사를 건너뛴다.
    """
    out: list[tuple[str, dict | None]] = [(s, None) for s, _ in top]
    in_top = {s for s, _ in top}
    room = max(0, params.max_watch - len(out))

    def liquid(sym: str) -> bool:
        if ticker is None:
            return True
        return (ticker.get(sym) or {}).get("quote_volume", 0) >= params.min_turnover_usd

    carried = sorted((s for s in recent
                      if s not in in_top and s in symbols and liquid(s)),
                     key=lambda s: recent[s], reverse=True)
    out.extend((s, recent[s]) for s in carried[:room])
    return out


def blocked(df1h: pd.DataFrame, params: Params = Params(),
            df4h: pd.DataFrame | None = None) -> bool:
    """대상에서 뺄 자리인지 — True면 제외. **1h 프레임으로 본다.**

    ① **정배열이 아니면 제외**(require_aligned). 24h 상승률 상위라도 1h
       구조가 역배열·혼조면 그 상승은 하락 추세 안의 반등이거나 방향이
       아직 안 잡힌 것이다 — 주도주의 눌림으로 볼 자리가 아니다.

       allow_bearish를 켜면 역배열도 통과한다(혼조만 남는 컷). 한 번 켜서
       돌려 봤다가 껐다 — 통과 종목이 15종에서 48종으로 3배 늘고 알림도
       그만큼 늘었다. 켤 때는 알림 줄의 🌱상승초기 마크로 두 종류가
       구분된다.
    ② **정배열일 때만 4h 장기 수퍼트렌드가 하락이면 제외**
       (exhausted_filter). 캔들이 4h 장기선 **위**에 있어야 통과다. 큰
       추세가 아직 안 꺾였다는 확인이고, 꺾였으면 오를 만큼 오른 뒤
       돌아선 것이라 새로 잡을 눌림이 아니다. **역배열엔 적용하지 않는다** —
       역배열은 장기선 아래인 게 정상이라, 걸면 역배열이 통째로 다시 잘린다.

       고정 이동평균 대신 수퍼트렌드를 쓴다. 트레일링 스톱이라 얕은 눌림엔
       안 꺾이고 추세가 실제로 돌아설 때 꺾인다 — '오를 만큼 오른 뒤'를
       묻는 조건에 이쪽이 맞다. 선은 turn_slow_period × turn_slow_mult
       (기본 30×6)이며, turn_enabled를 꺼도 이 게이트는 계속 쓴다.

    MA480을 못 구할 만큼 이력이 짧으면 alignment가 None을 주므로 ①에서
    걸린다. 1h 프레임이라 480봉 = **20일**이다 — 4h로 보던 때의 80일보다
    문턱이 훨씬 낮아 신규 상장이 덜 잘린다. 그래도 통과시키려면
    allow_short_history를 켠다.

    df4h가 없거나 봉이 모자라면 ②를 건너뛴다 — ①과 같이 **판단 불가는
    제외하지 않는다**.
    """
    aligned = alignment(df1h, tuple(params.exhausted_mas))
    passable = {"정배열"} | ({"역배열"} if params.allow_bearish else set())
    if params.require_aligned and aligned not in passable:
        if aligned is None and params.allow_short_history:
            return False        # 판단 불가는 통과 (설정으로 켰을 때만)
        return True
    if not params.exhausted_filter or aligned != "정배열":
        return False
    if df4h is None or len(df4h) < params.turn_slow_period + 2:
        return False
    d = supertrend_full(df4h, params.turn_slow_period,
                        params.turn_slow_mult)["dir"].iloc[-1]
    return bool(not pd.isna(d) and d <= 0)


def quiet(stat: dict | None, df4h: pd.DataFrame | None,
          params: Params = Params()) -> bool | None:
    """'조용한' 자리인지 — 거래대금이 작고 4h 변동성이 낮은 종목.

    **거르는 조건이 아니라 표시용이다.** 최근 10일 퍼프 알림 1785건 실측
    (`scripts/debug_probe.py`의 PROBE_MODE=leaders)에서 이 시그널의 대박
    (7일 내 고점 +20% 이상)과 쪽박(저점 −15% 이하)은 **같은 축**에 있었다 —
    이미 크게 오르고 변동성이 터진 종목이 양쪽을 다 만들어서, 대박을 남기는
    조건이 곧 쪽박을 남기는 조건이었다. 확장도·변동성·거래대금·직전 상승폭
    어느 것으로 잘라도 최상위 구간은 대박률 36~44%에 쪽박률 49~67%였다.

    유일하게 비대칭이었던 게 거래대금 하위 구간이다. 24h 거래대금 $6.2M
    미만에서 대박률은 28%→25%로 거의 안 깎이는데 쪽박률이 37%→10%로
    떨어졌고, 3일 뒤 중앙 수익률이 플러스(+0.7%)인 유일한 구간이었다
    (경로 승률 68% — 전체 평균은 53%). 변동성 조건을 같이 걸면 쪽박률이
    8%까지 내려간다. 그래서 두 조건을 AND로 본다.

    판단에 필요한 값이 없으면 None — 조용하다고도, 아니라고도 하지 않는다.
    """
    turnover = (stat or {}).get("quote_volume")
    if turnover is None or df4h is None or len(df4h) <= params.quiet_atr_period:
        return None
    a = atr(df4h, params.quiet_atr_period).iloc[-1]
    close = float(df4h["Close"].iloc[-1])
    if pd.isna(a) or close <= 0:
        return None
    return bool(float(turnover) < params.quiet_turnover_usd
                and float(a) / close < params.quiet_atr_pct)


def last_wave(df: pd.DataFrame, params: Params = Params()):
    """마지막으로 **닫힌** 파동의 (저점, 고점). 없으면 None. **1h 프레임.**

    파동 한 사이클:
        장기 하락 전환 ─┐  이 구간 최저 저가 = 저점
        단기 상향 돌파 ─┘
                        │  이 구간 최고 고가 = 고점
        단기 하향 돌파 ──  파동 확정 (= 눌림 시작)

    저점은 **장기가 다시 깨질 때까지 고정**이다. 그동안 단기가 여러 번
    오르내리면 같은 저점에 고점만 다른 파동이 이어진다 — 되돌림을 매번
    원래 파동 바닥 기준으로 재려는 것이다.

    저점 구간을 장기 기준으로 잡는 이유: 장기는 단기보다 늦게 깨지므로
    그 창이 하락의 깊은 부분만 덮고, 하락 중의 잔 반등에 흔들리지 않는다.
    단기 기준으로 자르면 마지막 짧은 다리만 봐서 진짜 바닥을 놓친다.

    **30m에서 1h으로 옮겼다.** 30m 파동은 폭이 2~8%밖에 안 돼서 분모가
    작았고, 조금만 밀려도 되돌림이 1.0을 훌쩍 넘었다(📐2.28·📐9.28 같은
    값이 실제로 나왔다). 같은 종목을 1h으로 재면 폭이 8~79%가 되고 값도
    0.6~1.2로 돌아온다. 게이트가 이미 받아 두는 프레임이라 요청도 준다.
    """
    n = len(df)
    if n < max(params.fib_fast_period, params.fib_slow_period) + 2:
        return None
    fast = supertrend_full(df, params.fib_fast_period, params.fib_fast_mult)
    slow = supertrend_full(df, params.fib_slow_period, params.fib_slow_mult)
    fd, sd = fast["dir"], slow["dir"]
    high, low = df["High"], df["Low"]

    def flip(series, i, up):
        a, b = series.iloc[i - 1], series.iloc[i]
        if pd.isna(a) or pd.isna(b):
            return False
        return (b > 0 >= a) if up else (b < 0 <= a)

    lo = lo_from = start = None
    closed = None
    for i in range(1, n):
        if flip(sd, i, up=False):           # 장기 하락 전환 → 저점 구간 시작
            lo, lo_from, start = None, i, None
        if flip(fd, i, up=True):            # 단기 상향 돌파 → 파동 시작
            if lo is None:
                if lo_from is None:
                    continue                # 장기 전환을 아직 못 봤다
                lo = float(low.iloc[lo_from:i + 1].min())
            start = i
        elif start is not None and flip(fd, i, up=False):
            closed = (lo, float(high.iloc[start:i].max()))
            start = None
    if closed is None or closed[1] <= closed[0]:
        return None
    return closed


def retrace(df: pd.DataFrame, params: Params = Params()) -> float | None:
    """마지막 닫힌 파동 대비 **지금** 되돌림 비율. 표시 기준 미만이면 None.

    (고점 − 현재가) ÷ (고점 − 저점). 1.0을 넘으면 저점을 깬 것이라
    되돌림이 아니라 이탈인데, 값은 그대로 돌려주고 표시 쪽에서 가른다.

    매 스캔 현재가로 다시 재는 값이다 — 되돌아 올라가면 표시도 사라진다.
    파동이 닫힌 시점의 값으로 재면 안 된다: 단기 수퍼트렌드는 고점에서
    조금만 밀려도 뒤집혀서, 그 순간 되돌림은 구조적으로 0.3 언저리다
    (PTB 10일 실측 9개 파동이 전부 0.09~0.34였다).
    """
    if not params.fib_enabled or df is None or not len(df):
        return None
    wave = last_wave(df, params)
    if wave is None:
        return None
    lo, hi = wave
    back = (hi - float(df["Close"].iloc[-1])) / (hi - lo)
    return back if back >= params.fib_min else None


TURN_FLIP = "전환"
TURN_TOUCH = "터치"


def turn_up(df4h: pd.DataFrame, params: Params = Params()) -> "str | bool | None":
    """4h에서 **장기가 상승인 채로** 단기가 눌림을 끝냈는지. 두 모양이 있다.

    - ``"전환"`` — 단기가 하락에서 상승으로 막 뒤집혔다.
    - ``"터치"`` — 단기가 이미 상승인데 봉이 단기선까지 밀렸다 닿았다.

    같은 사건의 앞뒤다. 전환은 선을 되찾는 순간이고, 터치는 그 뒤 눌릴
    때마다 선이 받쳐 주는 자리다. 전환을 놓치고 들어온 종목은 터치에서만
    보인다 — 그래서 둘 다 잡되 어느 쪽인지는 줄에 적는다. 한 봉이 둘 다면
    **전환이 이긴다**(선을 되찾는 게 다시 받히는 것보다 큰 사건이다).

    장기가 **상승**이어야 한다는 게 파동 ⓐABC와 정반대다. 거기선 하락 추세
    안의 반등을 잡지만, 여기는 이미 선 추세 안의 눌림이 끝나는 자리다.

    봉 하나만 보면 표시가 4시간 만에 사라진다 — 스캔은 매시간 도는데 그 사이
    상태는 그대로다. turn_bars(기본 3봉 = 12시간) 안이면 계속 표시한다.
    해당 없으면 False, 판단할 봉이 모자라면 None.
    """
    if not params.turn_enabled or df4h is None:
        return None
    need = max(params.turn_fast_period, params.turn_slow_period) + 2
    if len(df4h) < need:
        return None
    fast = supertrend_full(df4h, params.turn_fast_period, params.turn_fast_mult)
    slow = supertrend_full(df4h, params.turn_slow_period, params.turn_slow_mult)
    last = len(df4h) - 1
    sd = slow["dir"].iloc[last]
    if pd.isna(sd) or sd <= 0:
        return False                    # 장기가 상승이 아니면 이 자리가 아니다
    fd, fl = fast["dir"], fast["line"]
    touched = False
    for i in range(last, max(0, last - params.turn_bars), -1):
        a, b = fd.iloc[i - 1], fd.iloc[i]
        if pd.isna(a) or pd.isna(b):
            continue
        if b > 0 >= a:
            return TURN_FLIP            # 전환이 터치를 이긴다
        # 단기가 상승인 채로 봉이 선까지 밀렸다 — 눌림을 선이 받은 자리
        if params.turn_touch and b > 0 and not touched:
            line = fl.iloc[i]
            if not pd.isna(line) and float(df4h["Low"].iloc[i]) <= float(line) \
                    <= float(df4h["High"].iloc[i]):
                touched = True
    return TURN_TOUCH if touched else False


RESIST_TOUCH = "터치"
RESIST_BREAK = "돌파"


def resist_test(df1h: pd.DataFrame, params: Params = Params()) -> "str | bool | None":
    """1h에서 **단기선이 위에 있는 채로** 캔들이 아래에서 올라와 닿았는지.

    - ``"터치"`` — 단기가 아직 하락인데 고가가 선까지 올라갔다(저항 확인).
    - ``"돌파"`` — 종가가 선을 넘어 단기가 상승으로 뒤집혔다.

    🔼(4h)와 정반대 국면이다. 거기는 **장기 상승** 안에서 눌림이 끝나는
    자리를 보지만, 여기는 아직 꺾여 있는 종목이 위쪽 저항을 처음 건드리는
    자리다 — 뚫으면 국면이 바뀌고 못 뚫으면 저항이 확인된다. 어느 쪽이든
    지금 무슨 일이 벌어지는지가 그 줄에서 보여야 한다.

    turn1h_require_bearish면 1h 장기까지 하락이어야 한다. **기본은 꺼져
    있다** — ⚡ 게이트가 1h 정배열을 요구하므로 통과 종목의 1h 장기는 거의
    항상 상승이고, 켜 두면 이 표시가 구조적으로 안 붙는다. 꺼진 상태에서는
    '큰 추세는 살아 있는데 1h 단기가 꺾인 눌림'에서 그 단기선을 다시
    건드리거나 뚫는 자리를 잡는다.

    한 봉이 둘 다면 **돌파가 이긴다**. turn1h_bars(기본 3봉 = 3시간) 안이면
    계속 표시하고, 해당 없으면 False, 판단할 봉이 모자라면 None.
    """
    if not params.turn1h_enabled or df1h is None:
        return None
    need = max(params.turn_fast_period, params.turn_slow_period) + 2
    if len(df1h) < need:
        return None
    fast = supertrend_full(df1h, params.turn_fast_period, params.turn_fast_mult)
    last = len(df1h) - 1
    sd = None
    if params.turn1h_require_bearish:
        sd = supertrend_full(df1h, params.turn_slow_period,
                             params.turn_slow_mult)["dir"]
    fd, fl = fast["dir"], fast["line"]
    touched = False
    for i in range(last, max(0, last - params.turn1h_bars), -1):
        a, b = fd.iloc[i - 1], fd.iloc[i]
        if pd.isna(a) or pd.isna(b):
            continue
        # 장기는 **직전 봉** 기준으로 본다 — 세게 뚫는 봉은 장기까지 같이
        # 뒤집혀서, 그 봉으로 재면 정작 잡고 싶은 돌파가 통째로 걸러진다.
        if sd is not None:
            prev = sd.iloc[i - 1]
            if pd.isna(prev) or prev >= 0:
                continue
        if b > 0 >= a:
            return RESIST_BREAK     # 돌파가 터치를 이긴다
        # 단기가 아직 하락 — 선이 위에 있으므로 고가가 닿으면 저항 테스트다
        if b <= 0 and not touched:
            line = fl.iloc[i]
            if not pd.isna(line) and float(df1h["Low"].iloc[i]) <= float(line) \
                    <= float(df1h["High"].iloc[i]):
                touched = True
    return RESIST_TOUCH if touched else False


def tracking(df: pd.DataFrame, params: Params = Params()) -> dict | None:
    """현재 상태 스냅샷 — 종가가 ma선 위인지 아래인지.

    track_break_only(기본 켬)면 **지금 ma선 아래인 종목만** 돌려준다 —
    추적 목록도 이탈 조건을 그대로 받는다는 뜻이다. 끄면 감시 창 안의
    전 종목을 선 위/아래 상관없이 상태만 보여준다(예전 동작).
    데이터가 모자라 ma선을 못 구하면 None.
    """
    if len(df) < params.ma:
        return None
    m = sma(df["Close"], params.ma)
    if pd.isna(m.iloc[-1]):
        return None
    close, ma = float(df["Close"].iloc[-1]), float(m.iloc[-1])
    if params.track_break_only and close >= ma:
        return None                  # 선 위로 복귀 — 이탈 상태가 아니다
    return {"label": LABEL, "last_price": close, "ma": ma,
            "above_ma": bool(close >= ma), "ma_period": params.ma,
            "interval": INTERVAL}


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """마감된 15m OHLC(오름차순, UTC 인덱스)에서 ma선 하향 이탈을 찾는다.

    마지막 grace_bars+1개 봉을 각각 후보로 본다 — 스캔이 1시간 간격이라
    그 사이에 지나간 봉에서 발생한 이탈도 소급해 잡는다.
    중복 발송 방지는 호출 측(state)이 dedup_key로 처리한다.
    """
    n = len(df)
    if n < params.ma + 2:
        return []
    close = df["Close"]
    m = sma(close, params.ma)

    events = []
    for t in range(max(params.ma, n - 1 - params.grace_bars), n):
        if pd.isna(m.iloc[t]) or pd.isna(m.iloc[t - 1]):
            continue
        # 하향 '이탈' — 직전 봉은 선 위(또는 선상), 이번 봉은 선 아래
        if not (close.iloc[t] < m.iloc[t] and close.iloc[t - 1] >= m.iloc[t - 1]):
            continue
        events.append(SignalEvent(
            symbol=symbol,
            signal=NAME,
            bar_time=df.index[t],
            price=float(close.iloc[t]),
            detail={
                "label": LABEL,
                "ma": float(m.iloc[t]),
                "ma_period": params.ma,
                "interval": INTERVAL,
                # 15m봉이라 1h=4봉·4h=16봉. 24h는 스캐너가 티커에서 붙인다
                # (gain_24h) — 캔들 역산보다 지연이 없다.
                "ret_1h": pct_over(close, 4, t),
                "ret_4h": pct_over(close, 16, t),
            },
        ))
    return events
