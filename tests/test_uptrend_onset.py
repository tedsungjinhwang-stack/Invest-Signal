"""상승초입 시그널 검출 로직 테스트 — 네트워크 없이 합성 데이터로."""

import numpy as np
import pandas as pd

from invest_signal.signals import uptrend_onset
from invest_signal.signals.uptrend_onset import Params

DECLINE = 600           # 역배열을 만들 하락 구간 봉 수

# 아래 픽스처는 대부분 단조 하락이라 수퍼트렌드가 계속 하락추세다.
# ①역배열·②터치·③이탈 로직만 떼어 보려고 수퍼트렌드는 꺼 두고,
# 240 터치 경로는 명시적으로 켠다(기본값은 꺼짐).
# 합성 데이터에 Volume이 없으면 VWAP 조건은 자동 통과한다.
LEGACY = dict(touch_condition=True, supertrend_condition=False)
P = Params(**LEGACY)


def QP(on: bool) -> Params:
    """분기 앵커 VWAP 파라미터 — 픽스처가 한 분기 안에 들어가 리셋이 없다."""
    return Params(vwap_condition=on, vwap_period="Q", **LEGACY)


def make_df(closes, highs, volumes=None, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="4h", tz="UTC")
    c = pd.Series([float(x) for x in closes], index=idx)
    h = pd.Series([float(x) for x in highs], index=idx)
    data = {"Open": c, "High": np.maximum(c, h), "Low": np.minimum(c, h), "Close": c}
    if volumes is not None:
        data["Volume"] = [float(v) for v in volumes]
    return pd.DataFrame(data)


def base_decline(bars=DECLINE):
    """400→100 선형 하락 — 끝에서 120<240<480 역배열, 종가는 모든 MA 아래."""
    closes = list(np.linspace(400.0, 100.0, bars))
    highs = closes.copy()
    return closes, highs


def add_touch(closes, highs, spike_to=None):
    """다음 봉에서 고가만 240선까지(또는 spike_to까지) 찍는 터치 봉을 추가."""
    t = len(closes)
    closes.append(closes[-1])
    ma240 = float(np.mean(closes[t - 239:t + 1]))
    highs.append(max(ma240 + 1.0, spike_to or 0.0))
    return t


def add_sideways(closes, highs, bars, level=None):
    """이탈선 위·240선 아래에서 옆으로 기는 구간(터치 없음)."""
    level = level if level is not None else closes[-1] + 20.0
    for i in range(bars):
        closes.append(level + 0.01 * i)
        highs.append(closes[-1])


def add_drop(closes, highs, price=None):
    price = price if price is not None else closes[-1] - 5
    closes.append(price)
    highs.append(price)


def test_fires_on_first_close_below_entry_ma_after_touch():
    closes, highs = base_decline()
    add_touch(closes, highs)
    add_drop(closes, highs)                      # 하락 지속 → 종가 < 이탈선
    df = make_df(closes, highs)
    evs = uptrend_onset.detect(df, "TEST", P)
    assert len(evs) == 1
    ev = evs[0]
    assert ev.signal == "uptrend_onset"
    assert ev.bar_time == df.index[-1]
    assert ev.detail["touch_time"] == df.index[-2].isoformat()


def test_no_touch_no_signal():
    closes, highs = base_decline()
    df = make_df(closes, highs)                  # 종가는 늘 이탈선 아래지만 터치가 없음
    assert uptrend_onset.detect(df, "TEST", P) == []


def test_touch_without_bearish_alignment_no_signal():
    """상승장(정배열)에서는 고가가 240선 위여도 시그널이 없어야 한다."""
    closes = list(np.linspace(100.0, 400.0, DECLINE))   # 상승 → 정배열
    highs = closes.copy()
    closes.append(closes[-1] * 0.8)              # 급락 → 종가 < 이탈선
    highs.append(closes[-1] * 1.3)               # 고가는 240선보다 훨씬 위
    df = make_df(closes, highs)
    entry = df["Close"].rolling(P.ma_entry).mean().iloc[-1]
    m240 = df["Close"].rolling(240).mean().iloc[-1]
    assert df["Close"].iloc[-1] < entry and df["High"].iloc[-1] >= m240   # 전제 확인
    assert uptrend_onset.detect(df, "TEST", P) == []


def test_stale_trigger_respects_grace():
    closes, highs = base_decline()
    add_touch(closes, highs)
    add_drop(closes, highs)                      # 트리거 봉
    add_drop(closes, highs)                      # 그 뒤 1봉 경과
    df = make_df(closes, highs)
    evs = uptrend_onset.detect(df, "TEST", Params(grace_bars=1, **LEGACY))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-2]   # 트리거 봉 기준
    assert uptrend_onset.detect(df, "TEST", Params(grace_bars=0, **LEGACY)) == []

    add_drop(closes, highs)                      # 2봉 경과 → grace=1로도 stale
    df = make_df(closes, highs)
    assert uptrend_onset.detect(df, "TEST", Params(grace_bars=1, **LEGACY)) == []


def test_touch_window_boundary_exact():
    """터치 후 정확히 60봉째 이탈은 발화, 61봉째는 미발화."""
    def build(sideways):
        closes, highs = base_decline()
        add_touch(closes, highs)
        add_sideways(closes, highs, bars=sideways)
        add_drop(closes, highs, 90.0)
        return make_df(closes, highs)

    p = Params(touch_window_bars=60, **LEGACY)
    df60 = build(59)                             # 터치→트리거 간격 = 60봉
    evs = uptrend_onset.detect(df60, "TEST", p)
    assert len(evs) == 1 and evs[0].bar_time == df60.index[-1]
    df61 = build(60)                             # 간격 61봉 → 윈도우 초과
    assert uptrend_onset.detect(df61, "TEST", p) == []


def test_rearm_on_new_touch_after_fire():
    """발화 → 이탈선 위 복귀 → 새 240 터치 → 다시 이탈하면 재발화."""
    closes, highs = base_decline()
    add_touch(closes, highs)
    add_drop(closes, highs)                      # 1차 발화 봉
    add_sideways(closes, highs, bars=5)          # 이탈선 위로 복귀
    add_touch(closes, highs)                     # 새 터치
    add_drop(closes, highs, 90.0)                # 2차 이탈
    df = make_df(closes, highs)
    evs = uptrend_onset.detect(df, "TEST", P)
    assert len(evs) == 1
    assert evs[0].bar_time == df.index[-1]
    assert evs[0].detail["touch_time"] == df.index[-2].isoformat()


def test_missed_bar_trigger_found_even_if_it_touches():
    """grace 소급 판정: 놓친 트리거 봉 자체가 240 터치여도 잡아야 한다."""
    closes, highs = base_decline()
    add_touch(closes, highs)                     # 원래 터치
    t = len(closes)                              # 트리거 봉인데 위꼬리가 240선도 찍음
    closes.append(closes[-1] - 5)
    ma240 = float(np.mean(closes[t - 239:t + 1]))
    highs.append(ma240 + 1.0)
    add_sideways(closes, highs, bars=1)          # 다음 봉은 이탈선 위 복귀
    df = make_df(closes, highs)
    evs = uptrend_onset.detect(df, "TEST", Params(grace_bars=1, **LEGACY))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-2]


def test_ride_above_ma240_does_not_refresh_touch():
    """캔들 전체가 240선 위인 봉은 터치가 아니다 — 터치 시점은 진짜
    윗꼬리 터치 봉으로 남아야 한다 (매봉 갱신 버그 방지)."""
    closes, highs = base_decline()
    t_idx = add_touch(closes, highs)          # 진짜 터치 봉 (윗꼬리)
    df0 = make_df(closes, highs)
    m240 = float(df0["Close"].rolling(240).mean().iloc[-1])
    for _ in range(8):                        # 240선 위 올라탄 봉들 — 터치 아님
        closes.append(m240 + 20.0)
        highs.append(m240 + 25.0)
    add_drop(closes, highs, 110.0)            # 이탈선 아래 붕괴 → 발화
    df = make_df(closes, highs)
    evs = uptrend_onset.detect(df, "TEST", P)
    assert len(evs) == 1
    assert evs[0].detail["touch_time"] == df.index[t_idx].isoformat()


def rebound_after_decline(rise_bars=120):
    """하락 뒤 반등 레그 — MA120은 MA240 위로 올라오지만 MA240 < MA480은 유지.

    120선을 역배열 판정에 넣느냐 마느냐가 갈리는 유일한 구간이라
    두 설정의 차이를 재는 데 쓴다.
    """
    closes, highs = base_decline()
    for _ in range(rise_bars):
        closes.append(closes[-1] + 1.0)
        highs.append(closes[-1])
    t = len(closes)
    ma240 = float(np.mean(closes[t - 240:t]))
    closes.append(ma240 - 5.0)              # 240선을 걸친 눌림 봉 (저가<240선<고가)
    highs.append(ma240 + 5.0)
    add_drop(closes, highs, closes[-1] - 1.0)   # 이탈선 아래 마감 → 트리거
    return make_df(closes, highs)


def test_ma_align_two_lines_admits_what_three_lines_reject():
    """(240,480)은 반등으로 120선이 뒤집힌 자리도 잡지만
    기본값 (120,240,480)은 역배열이 깨졌다고 보고 거른다."""
    df = rebound_after_decline()
    c = df["Close"]
    ma = {k: float(c.rolling(k).mean().iloc[-1]) for k in (120, 240, 480)}
    assert ma[120] > ma[240] < ma[480]          # 전제: 120선만 뒤집힌 상태
    assert len(uptrend_onset.detect(df, "TEST", Params(ma_align=(240, 480), **LEGACY))) == 1
    assert uptrend_onset.detect(df, "TEST", Params(ma_align=(120, 240, 480), **LEGACY)) == []


def test_ma_align_three_lines_still_fires_on_full_bearish():
    """기본값(3선)은 완전 역배열 구간에서 그대로 발화한다."""
    closes, highs = base_decline()
    add_touch(closes, highs)
    add_drop(closes, highs)
    df = make_df(closes, highs)
    evs = uptrend_onset.detect(df, "TEST", Params(ma_align=(120, 240, 480), **LEGACY))
    assert len(evs) == 1 and evs[0].bar_time == df.index[-1]


def test_vwap_condition_gates_signal():
    """트리거 봉 종가가 앵커드 VWAP 위여야 발화 — 아래면 대기."""
    # 522봉(≈87일). VWAP 리셋이 픽스처를 흔들지 않도록 분기 앵커로 고정한다
    dec = 495
    closes, highs = base_decline(bars=dec)
    add_sideways(closes, highs, bars=25, level=120.0)   # 이탈선(MA20)을 ~120으로
    add_touch(closes, highs)
    # 트리거 후보: 직전 저점(100)보다 위지만 20선(≈120)보다는 아래로 마감
    add_drop(closes, highs, 110.0)
    n = len(closes)
    # 평평한 120 구간에 거래량 몰빵 → VWAP(≈121) > 종가 110 → 아직 아래 → 대기
    vols_high = [1.0] * n
    for i in range(dec, n):
        vols_high[i] = 5000.0
    df_below = make_df(closes, highs, volumes=vols_high)
    assert uptrend_onset.detect(df_below, "TEST", QP(True)) == []
    # 같은 데이터라도 조건을 끄면 이탈선 하회만으로 발화
    evs = uptrend_onset.detect(df_below, "TEST", QP(False))
    assert len(evs) == 1 and evs[0].detail["above_vwap"] is False
    # 저가 구간(~100)에 거래량 몰빵 → VWAP(≈109)이 종가(110) 아래로 → 발화
    vols_low = [1.0] * n
    for i in range(dec - 30, dec):
        vols_low[i] = 5000.0
    df_above = make_df(closes, highs, volumes=vols_low)
    evs = uptrend_onset.detect(df_above, "TEST", QP(True))
    assert len(evs) == 1 and evs[0].detail["above_vwap"] is True


def test_fires_when_vwap_line_drops_below_price():
    """이탈 봉이 VWAP 아래면 대기 — VWAP 선이 내려와 종가가 그 위가
    되는 첫 봉에서 발화한다 (선이 위에 있다가 확 내려오는 케이스)."""
    closes, highs = base_decline(bars=520)
    add_touch(closes, highs)
    add_drop(closes, highs, 102.0)          # 이탈선 아래·VWAP(≈105) 아래 → 대기
    vols = [1.0] * len(closes)
    for i, c in enumerate(closes):
        if 104.0 <= c <= 106.0:             # 저가 구간 몰빵 → VWAP ≈ 105
            vols[i] = 100000.0
    df_wait = make_df(closes, highs, volumes=vols)
    assert uptrend_onset.detect(df_wait, "TEST", QP(True)) == []
    # 대량 저가 거래로 VWAP 선이 ~101로 확 내려옴 (이 봉 자체는 98 < VWAP → 여전히 대기)
    add_drop(closes, highs, 98.0)
    vols.append(500000.0)
    # 다음 봉 종가 101.5 > VWAP(≈101) — 선이 캔들 아래로 온 첫 봉 → 발화
    add_drop(closes, highs, 101.5)
    vols.append(1.0)
    df_fire = make_df(closes, highs, volumes=vols)
    evs = uptrend_onset.detect(df_fire, "TEST", QP(True))
    assert len(evs) == 1 and evs[0].bar_time == df_fire.index[-1]
    assert evs[0].detail["above_vwap"] is True


def test_insufficient_data_returns_empty():
    closes = list(np.linspace(200.0, 100.0, 100))
    df = make_df(closes, closes)
    assert uptrend_onset.detect(df, "TEST", P) == []


def test_still_active_tracks_until_240_refail():
    """상승초입은 이탈선 복귀·240선 돌파까지 유지하고, 240선 재이탈 시 제거."""
    closes, highs = base_decline()
    add_touch(closes, highs)
    add_drop(closes, highs)                            # 트리거
    df = make_df(closes, highs)
    ev = uptrend_onset.detect(df, "TEST", P)[0]
    add_sideways(closes, highs, bars=2, level=130.0)   # 이탈선 위 복귀 — 유지
    assert uptrend_onset.still_active(make_df(closes, highs), ev, P)
    add_sideways(closes, highs, bars=2, level=200.0)   # 240선(≈160) 위 마감 — 유지
    assert uptrend_onset.still_active(make_df(closes, highs), ev, P)
    add_drop(closes, highs, 140.0)                     # 240선 아래 재마감 → 실패
    assert not uptrend_onset.still_active(make_df(closes, highs), ev, P)


def rebound_with_supertrend_up(rise=25, step=4.0, flat=15, dip=0.01, dip_bars=2):
    """하락 → 반등 → 횡보 → 얕은 눌림.

    끝 봉에서 역배열(MA120<MA240<MA480)은 그대로인데 수퍼트렌드는 상승으로
    뒤집혀 있고, 종가는 MA20 아래다 — 이 시그널이 노리는 바로 그 상태다.
    눌림을 깊게 잡으면 수퍼트렌드가 도로 하락으로 뒤집혀 픽스처가 깨진다.
    """
    closes, highs = base_decline()
    for _ in range(rise):                        # 반등 — 수퍼트렌드를 상승으로
        closes.append(closes[-1] + step)
        highs.append(closes[-1])
    for _ in range(flat):                        # 횡보 — MA20이 따라붙는다
        closes.append(closes[-1] + 0.01)
        highs.append(closes[-1])
    for _ in range(dip_bars):                    # 얕은 눌림 — 종가 < MA20
        closes.append(closes[-1] * (1 - dip))
        highs.append(closes[-1])
    return make_df(closes, highs)


def test_supertrend_condition_gates_signal():
    """수퍼트렌드가 상승일 때만 발화 — 같은 가격에서 배수만 조여 하락으로
    뒤집으면 다른 조건이 다 맞아도 안 뜬다."""
    from invest_signal.indicators import supertrend

    df = rebound_with_supertrend_up()
    c = df["Close"]
    ma = {k: float(c.rolling(k).mean().iloc[-1]) for k in (20, 120, 240, 480)}
    assert ma[120] < ma[240] < ma[480]                   # 전제: 아직 역배열
    assert c.iloc[-1] < ma[20]                           # 전제: MA20 아래
    assert supertrend(df, 22, 3.0).iloc[-1] == 1         # 전제: 수퍼트렌드 상승
    evs = uptrend_onset.detect(df, "TEST", Params())
    assert len(evs) == 1 and evs[0].detail["supertrend"] == 1

    # 배수를 조이면 같은 가격인데 수퍼트렌드가 하락으로 뒤집힌다 → 차단
    assert supertrend(df, 22, 1.0).iloc[-1] == -1
    assert uptrend_onset.detect(df, "TEST", Params(st_mult=1.0)) == []
    # 조건 자체를 끄면 다시 발화 — 수퍼트렌드가 유일한 차단 사유였다
    assert len(uptrend_onset.detect(
        df, "TEST", Params(st_mult=1.0, supertrend_condition=False))) == 1


def test_without_touch_condition_fires_on_state_entry():
    """터치 조건이 꺼진 기본값은 '상태에 막 들어선 봉'만 알린다 —
    이어지는 봉은 같은 이탈이라 재알림하지 않는다."""
    df = rebound_with_supertrend_up()
    evs = uptrend_onset.detect(df, "TEST", Params(grace_bars=3))
    assert len(evs) == 1                          # 눌림 2봉이지만 첫 봉만
    assert evs[0].bar_time == df.index[-2]
    assert evs[0].detail["touch_time"] is None    # 터치 기준점이 없다
    # 240 터치를 다시 요구하면 이 픽스처엔 터치가 없어 걸러진다
    assert uptrend_onset.detect(df, "TEST",
                                Params(grace_bars=3, touch_condition=True)) == []


def test_vwap_mode_touch_vs_above():
    """종가가 VWAP 아래인 눌림 — above는 거르고, 캔들이 선을 찍었으면 touch는 통과."""
    from invest_signal.indicators import anchored_vwap

    df = rebound_with_supertrend_up()
    vol = [1.0] * len(df)
    for i in range(DECLINE + 25, DECLINE + 40):   # 횡보 구간 몰빵 → VWAP을 위로
        vol[i] = 1e7
    df = df.assign(Volume=vol)
    vw = float(anchored_vwap(df, "M").iloc[-1])
    assert df["Close"].iloc[-1] < vw                    # 전제: 종가가 VWAP 아래
    assert df["High"].iloc[-1] < vw                     # 전제: 선을 찍지도 않음
    for mode in ("above", "touch", "any"):
        assert uptrend_onset.detect(df, "TEST", Params(vwap_mode=mode)) == []

    # 윗꼬리가 선을 찍게 하면 touch·any만 통과 — above는 종가 기준이라 그대로 차단
    hit = df.copy()
    hit.iloc[-1, hit.columns.get_loc("High")] = vw + 1.0
    assert uptrend_onset.detect(hit, "TEST", Params(vwap_mode="above")) == []
    assert len(uptrend_onset.detect(hit, "TEST", Params(vwap_mode="touch"))) == 1
    assert len(uptrend_onset.detect(hit, "TEST", Params(vwap_mode="any"))) == 1
