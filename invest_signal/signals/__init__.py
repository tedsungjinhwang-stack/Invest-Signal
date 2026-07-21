"""시그널 모듈 — 시그널 하나당 파일 하나.

새 시그널(예: 풀백)을 추가할 때는 uptrend_onset.py를 본떠서
detect(df, symbol, params) -> SignalEvent | None 함수를 만들고
scanner.SIGNAL_DETECTORS에 등록하면 된다.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class SignalEvent:
    """시그널 1건 — 어떤 종목의 어떤 봉에서 무슨 시그널이 떴는지."""

    symbol: str
    signal: str                 # 예: "uptrend_onset"
    bar_time: pd.Timestamp      # 트리거 봉의 시작 시각(UTC)
    price: float                # 트리거 봉 종가
    detail: dict = field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        """상태 파일에서 중복 알림을 막을 때 쓰는 키."""
        return f"{self.symbol}|{self.signal}|{self.bar_time.isoformat()}"
