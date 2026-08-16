"""크립토 모멘텀 눌림목/이탈 시그널 (15m봉) — 크립토 선물 전용.

24시간 상승률 **상위 top_n종** 중 **4h가 정배열인 종목**만 대상으로,
15분봉 종가가 ma(기본 20)선을 **하향 이탈하는 첫 봉**에서 알린다.
하루 사이 가장 많이 오른 주도주가 단기 추세선을 깨는 자리를 잡는다.
제외 규칙은 blocked() 참고.

다른 시그널과 두 가지가 다르다:
  · 4h봉이 아니라 **15m봉**으로 판정한다 (20봉 = 5시간).
  · 종목 선정이 횡단면 순위라 스캐너가 대상 5종을 먼저 고른 뒤
    이 모듈의 detect를 종목별로 부른다.

'첫 봉'만 알리므로 선 아래에 머무는 동안 반복 알림은 없다. 선 위로
복귀했다가 다시 깨면 새 이탈로 다시 알린다.
"""

from dataclasses import dataclass

import pandas as pd

from ..indicators import alignment, pct_over, sma
from . import SignalEvent

NAME = "leader_break"
LABEL = "크립토 모멘텀 눌림목/이탈"
CRYPTO_ONLY = True      # ETF·주식 스캔에서는 돌리지 않는다
INTERVAL = "15m"
KLINE_LIMIT = 200       # ma + grace + 여유 (ma를 키워도 넉넉하도록)
TREND_INTERVAL = "4h"   # 아래 exhausted() 판정용 — 4h 프레임이 없을 때만 따로 받는다
TREND_LIMIT = 600       # MA480 성립(480봉) + 여유


@dataclass(frozen=True)
class Params:
    top_n: int = 15             # 24h 상승률 상위 몇 종을 볼지
    board_top: int = 5          # 알림 맨 위에 조건 없이 적어줄 상위 종목 수
    ma: int = 20                # 15m봉 SMA 기간 (20봉 = 5시간)
    grace_bars: int = 4         # 소급 판정 봉 수 — 15m×4 = 1시간(스캔 주기)
    min_turnover_usd: float = 1_000_000   # 24h 거래대금 하한(유동성)
    watch_days: int = 3         # 상위권에서 밀려난 뒤에도 계속 볼 기간
    max_watch: int = 60         # 한 스캔에서 15m 캔들을 받을 최대 종목 수
    track_break_only: bool = True   # 추적도 '20선 아래'인 동안만 (복귀하면 제외)
    exhausted_filter: bool = True        # 아래 blocked() 조건에 걸리면 대상에서 제외
    require_aligned: bool = True         # 4h 정배열이 아니면 제외 (역배열·혼조 컷)
    allow_short_history: bool = False    # 이력이 짧아 배열 판정 불가면 통과시킬지
    exhausted_mas: tuple = (120, 240, 480)   # 4h봉 정배열 판정 3선
    exhausted_below: int = 480               # 정배열이어도 종가가 이 선 아래면 제외


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


def blocked(df4h: pd.DataFrame, params: Params = Params()) -> bool:
    """대상에서 뺄 자리인지 — True면 제외. 조건 두 개를 함께 본다.

    ① **정배열이 아니면 제외**(require_aligned). 24h 상승률 상위라도 4h
       구조가 역배열·혼조면 그 상승은 하락 추세 안의 반등이거나 방향이
       아직 안 잡힌 것이다 — 주도주의 눌림으로 볼 자리가 아니다.
    ② **정배열이어도 종가가 exhausted_below선 아래면 제외**
       (exhausted_filter). 상승 구조가 완성된 상태에서 480선까지 밀렸으면
       오를 만큼 오른 뒤 꺾인 것이라 새로 잡을 눌림이 아니다.

    MA480을 못 구할 만큼 이력이 짧으면(상장 80일 미만) alignment가 None을
    주므로 ①에서 걸린다. 신규 상장이 24h 상승률 상위를 자주 차지하는
    만큼 이 컷은 체감이 크다 — 통과시키려면 allow_short_history를 켠다.
    """
    aligned = alignment(df4h, tuple(params.exhausted_mas))
    if params.require_aligned and aligned != "정배열":
        if aligned is None and params.allow_short_history:
            return False        # 판단 불가는 통과 (설정으로 켰을 때만)
        return True
    if not params.exhausted_filter or aligned != "정배열":
        return False
    ref = sma(df4h["Close"], params.exhausted_below)
    if pd.isna(ref.iloc[-1]):
        return False
    return bool(float(df4h["Close"].iloc[-1]) < float(ref.iloc[-1]))


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
