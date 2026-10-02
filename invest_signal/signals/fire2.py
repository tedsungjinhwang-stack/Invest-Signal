"""🔥🔥 — 🔥(떡상 조짐) 줄 중 '롱이 몰려 있지 않은' 줄 (크립토 전용, 선물 소스에서만).

🔥가 붙은 줄에 선물 수급 조건 하나를 더 본다. 칸마다 다르다:

  🟢 상승초입  최근 24h(15m 96봉) 평균 현선갭(퍼프 ÷ 인덱스 − 1) ≤ onset_gap_mean24_max(0)
              — 선물이 현물보다 비싸게 거래되지 않았다
  ⚡ 모멘텀    OI(미결제약정)가 최근 4h 동안 leader_oi4_max(−0.5%) 이하로 줄었다
              — 레버리지가 정리되는 중이다
  🌊 파동      지금(마지막 마감 15m봉) 현선갭 ≤ wave_gap_now_max(−0.1%) — 선물 디스카운트

퍼프 백테스트(07-20~09-21, 🔥 줄만, '−15%보다 +30% 먼저' 7일 · 앞 기간 / 뒤 기간):
  상승초입 28% / 27% vs 나머지 🔥 16% / 14% · 같은 날 비교 +10%p(90% +5~+14) · −15% 먼저 14% vs 33%
  ⚡       57% / 46% vs 35% / 36% · 같은 날 +15%p(+3~+26) · 조건 O 83건뿐
  파동     23% / 41% vs 15% / 20% · 실제 알림 50% vs 29%(12건) · 같은 날 +10%p(−1~+22) — 경계선
🔥가 아닌 줄에 같은 조건을 걸면 효과가 거의 없다(0~+5%p) — 🔥와 겹칠 때만 뜻이 있다.
여기서 현선갭은 ±0.1% 수준의 미세한 차이다(🚨세력 이탈의 5~10%와 다른 얘기).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

GAP_INTERVAL = "15m"
DAY_BARS = 96
OI_PERIOD = "5m"


@dataclass(frozen=True)
class Params:
    enabled: bool = True
    onset_gap_mean24_max: float = 0.0
    leader_oi4_max: float = -0.005
    wave_gap_now_max: float = -0.001


def gap_stats(px15: pd.DataFrame | None, idx15: pd.DataFrame | None,
              now: pd.Timestamp) -> dict | None:
    """마감 15m봉 기준 현선갭 — 지금 값(gap_now)과 24h 평균(gap_mean24). 못 재면 None.

    24h 평균은 96봉 중 50봉 이상 맞물릴 때만 낸다.
    """
    if px15 is None or idx15 is None or not len(px15) or not len(idx15):
        return None
    step = pd.Timedelta(minutes=15)
    c = px15["Close"][px15.index + step <= now].iloc[-DAY_BARS:]
    i = idx15["Close"][idx15.index + step <= now]
    g = (c / i.reindex(c.index) - 1).dropna()
    if g.empty or g.index[-1] != c.index[-1]:
        return None
    out = {"gap_now": float(g.iloc[-1])}
    if len(g) >= 50:
        out["gap_mean24"] = float(g.mean())
    return out


def oi_change(oi: pd.Series | None, now: pd.Timestamp, hours: float = 4) -> float | None:
    """OI 지금 ÷ hours시간 전 − 1. 두 스냅샷 모두 기준 시각 30분 안이어야 한다."""
    if oi is None or not len(oi):
        return None
    o = oi[~oi.index.duplicated()].sort_index()
    o = o[o.index <= now]
    tol = pd.Timedelta(minutes=30)
    if o.empty or now - o.index[-1] > tol:
        return None
    then = o[o.index <= now - pd.Timedelta(hours=hours)]
    if then.empty or (now - pd.Timedelta(hours=hours)) - then.index[-1] > tol or then.iloc[-1] <= 0:
        return None
    return float(o.iloc[-1] / then.iloc[-1] - 1)


def mark(signal: str, gaps: dict | None, oi4: float | None,
         params: Params = Params()) -> bool:
    """이 🔥 줄에 🔥🔥를 붙일지. 값을 못 재면 False(붙이지 않는다)."""
    if not params.enabled:
        return False
    if signal == "vwap_onset":
        v = (gaps or {}).get("gap_mean24")
        return v is not None and not np.isnan(v) and v <= params.onset_gap_mean24_max
    if signal == "leader_break":
        return oi4 is not None and oi4 <= params.leader_oi4_max
    if signal == "wave_setup":
        v = (gaps or {}).get("gap_now")
        return v is not None and not np.isnan(v) and v <= params.wave_gap_now_max
    return False
