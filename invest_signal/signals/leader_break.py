"""주도주 이탈 시그널 (15m봉) — 크립토 선물 전용.

24시간 상승률 **상위 top_n종**만 대상으로, 15분봉 종가가 ma(기본 60)선을
**하향 이탈하는 첫 봉**에서 알린다. 하루 사이 가장 많이 오른 종목들이
단기 추세선을 깨는 자리 — 주도주의 모멘텀이 꺾이는 지점을 잡는다.

다른 시그널과 두 가지가 다르다:
  · 4h봉이 아니라 **15m봉**으로 판정한다 (60봉 = 15시간).
  · 종목 선정이 횡단면 순위라 스캐너가 대상 5종을 먼저 고른 뒤
    이 모듈의 detect를 종목별로 부른다.

'첫 봉'만 알리므로 60선 아래에 머무는 동안 반복 알림은 없다. 60선 위로
복귀했다가 다시 깨면 새 이탈로 다시 알린다.
"""

from dataclasses import dataclass

import pandas as pd

from ..indicators import sma
from . import SignalEvent

NAME = "leader_break"
LABEL = "주도주 이탈"
CRYPTO_ONLY = True      # ETF·주식 스캔에서는 돌리지 않는다
INTERVAL = "15m"
KLINE_LIMIT = 200       # ma(60) + grace + 여유


@dataclass(frozen=True)
class Params:
    top_n: int = 10             # 24h 상승률 상위 몇 종을 볼지
    ma: int = 60                # 15m봉 SMA 기간 (60봉 = 15시간)
    grace_bars: int = 4         # 소급 판정 봉 수 — 15m×4 = 1시간(스캔 주기)
    min_turnover_usd: float = 1_000_000   # 24h 거래대금 하한(유동성)


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


def detect(df: pd.DataFrame, symbol: str, params: Params = Params()) -> list[SignalEvent]:
    """마감된 15m OHLC(오름차순, UTC 인덱스)에서 60선 하향 이탈을 찾는다.

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
            },
        ))
    return events
