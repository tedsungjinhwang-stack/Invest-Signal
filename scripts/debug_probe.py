"""심볼 진단 프로브 — 러너에서 실데이터로 특정 코인의 시그널 판정을 재현.

PROBE_SYMBOLS 환경변수(쉼표 구분)의 각 심볼에 대해 최근 봉들의 OHLC와
월간/분기 VWAP, 일자별 거래량·거래대금, 4h 시그널 재현 결과를 출력한다.
알림이 왜 왔는지/안 왔는지 검증할 때 .github/workflows/debug-probe.yml의
심볼을 바꿔 푸시.

PROBE_DAYS를 주면 그 날짜(UTC, 쉼표 구분)의 일봉 요약을 따로 찍는다 —
샌드박스에서 못 보는 퍼프 전용 종목의 과거 거래대금을 확인할 때 쓴다.

PROBE_MODE=perf면 **알림 성과 분석**을 돈다: state/alerts_state.json에
남은 최근 PROBE_DAYS_BACK일치 알림을 시그널별로 모아, 발화 봉 종가를
진입가로 보고 이후 24h·3d·7d 수익률과 MFE/MAE, 경로 승률(±10% 중 어느
쪽에 먼저 닿았는지)을 낸다.

PROBE_MODE=leaders면 ⚡(크립토 모멘텀) 알림만 골라 **대박/쪽박 선별
분석**을 돈다: 발화 봉 시점의 피처(그때까지 상승폭·확장도·변동성·
거래대금·재알림 횟수)와 사후 MFE/MAE를 교차해, 어떤 조건이 대박군을
남기고 쪽박군을 거르는지 룰 시뮬레이션까지 낸다.

PROBE_MODE=turnover면 심볼 진단 대신 **거래대금 하한 스윕**을 돈다:
유니버스 전체를 받아 하한별 통과 종목 수를 세고, 최근 발송된 알림
(state/sent_log.jsonl)의 종목이 각 하한에서 살아남는지 하나씩 찍는다.
샌드박스는 현물 미러로 폴백해 퍼프보다 거래대금이 훨씬 작게 찍히므로,
하한을 조정할 때는 반드시 여기(러너, 프록시 경유 퍼프)서 재야 한다.
"""

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from invest_signal import data_binance, indicators
from invest_signal.signals import pullback, pump_early, uptrend_onset

PROBES = (uptrend_onset, pullback, pump_early)   # 4h 종가 시그널 — 모듈을 늘리면 그대로 재현된다

symbols = [s.strip().upper() for s in
           os.environ.get("PROBE_SYMBOLS", "BTCUSDT").split(",") if s.strip()]

FLOORS = [float(x) * 1e6 for x in
          os.environ.get("PROBE_FLOORS", "1,2,3,5,10").split(",") if x.strip()]
RECENT_MSGS = int(os.environ.get("PROBE_RECENT_MSGS", "6"))
SECTION_KEY = {"상승초입": "uptrend_onset", "펌핑초기": "pump_early",
               "파동": "wave_setup", "크립토 모멘텀 눌림목/이탈": "leader_break"}


def _recent_alert_symbols(limit: int) -> dict:
    """최근 발송 메시지에서 칸별 등장 종목 — {시그널키: {심볼}}."""
    import json
    import re
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "state", "sent_log.jsonl")
    try:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    except OSError:
        return {}
    out, sec = {}, None
    for r in rows[-limit:]:
        for ln in r.get("text", "").splitlines():
            m = re.match(r"^([🟢🔵🌱🌊⚡🔻]) <b>(.+?)</b>", ln)
            if m:
                sec = SECTION_KEY.get(m.group(2))
                continue
            if sec is None or not ln.startswith(("• ", "↳ ")):
                continue
            plain = re.sub(r"<[^>]+>", "", ln)
            parts = plain.split()
            if len(parts) > 1:
                out.setdefault(sec, set()).add(parts[1] + "USDT")
    return out


CRYPTO_SIGNALS = ("uptrend_onset", "pump_early", "wave_setup", "leader_break")


def _performance(sess) -> None:
    """최근 알림의 사후 성과 — 발화 봉 종가를 진입가로 본 전진 수익률."""
    import json
    import statistics as st

    import pandas as pd

    from invest_signal import config as cfg_mod

    back = int(os.environ.get("PROBE_DAYS_BACK", "10"))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "state", "alerts_state.json"), encoding="utf-8") as f:
        alerts = json.load(f).get("alerts", {})

    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(days=back)
    events = []
    for key in alerts:
        parts = key.split("|")
        if len(parts) < 3 or parts[1] not in CRYPTO_SIGNALS:
            continue
        try:
            t = pd.Timestamp(parts[2])
        except ValueError:
            continue
        if t.tz is None:
            t = t.tz_localize("UTC")
        if t >= cutoff:
            events.append((parts[0], parts[1], t))
    if not events:
        print(f"최근 {back}일 알림 없음")
        return

    cfg = cfg_mod.load()
    c = cfg.get("crypto") or {}
    source, syms = data_binance.resolve_source(sess, c.get("source", "auto"),
                                               set(c.get("exclude") or []), print)
    want = sorted({sym for sym, _, _ in events} & set(syms))
    print(f"\n최근 {back}일 알림 {len(events)}건 · 종목 {len(want)}종 "
          f"(유니버스 밖 {len({s for s,_,_ in events}) - len(want)}종 제외) · {source}")
    frames = data_binance.fetch_all(sess, want, source, limit=750, log=print,
                                    workers=int(c.get("fetch_workers", 6)))

    HORIZONS = (("24h", 6), ("3d", 18), ("7d", 42))
    MAX_H = 42
    per = {}
    detail = []
    for sym, sig, t in events:
        df = frames.get(sym)
        if df is None or not len(df):
            continue
        i = int(df.index.searchsorted(t, "left"))
        if i >= len(df):
            continue                       # 아직 그 봉이 프레임에 없다
        entry = float(df["Close"].iloc[i])
        if entry <= 0:
            continue
        fwd = df.iloc[i + 1:i + 1 + MAX_H]
        rec = {"sym": sym, "sig": sig, "t": t, "bars": len(fwd)}
        for name, n in HORIZONS:
            seg = df.iloc[i + n]["Close"] if i + n < len(df) else None
            rec[name] = None if seg is None else float(seg) / entry - 1
        if len(fwd):
            rec["mfe"] = float(fwd["High"].max()) / entry - 1
            rec["mae"] = float(fwd["Low"].min()) / entry - 1
            hit_up = hit_dn = None
            for j in range(len(fwd)):
                if hit_up is None and float(fwd["High"].iloc[j]) / entry - 1 >= 0.10:
                    hit_up = j
                if hit_dn is None and float(fwd["Low"].iloc[j]) / entry - 1 <= -0.10:
                    hit_dn = j
                if hit_up is not None or hit_dn is not None:
                    break
            rec["path"] = (None if hit_up is None and hit_dn is None
                           else (hit_up is not None and
                                 (hit_dn is None or hit_up <= hit_dn)))
        per.setdefault(sig, []).append(rec)
        detail.append(rec)

    def med(vals):
        vals = [v for v in vals if v is not None]
        return None if not vals else st.median(vals)

    def pct(v):
        return "  —  " if v is None else f"{v * 100:+6.1f}%"

    print(f"\n== 시그널별 성과 (최근 {back}일, 진입 = 발화 봉 종가)\n")
    print(f"  {'시그널':<16}{'건수':>5}{'24h':>8}{'3d':>8}{'7d':>8}"
          f"{'MFE':>8}{'MAE':>8}{'경로승률':>9}{'±10%도달':>9}")
    for sig in CRYPTO_SIGNALS:
        rs = per.get(sig) or []
        if not rs:
            continue
        paths = [r["path"] for r in rs if r.get("path") is not None]
        wr = f"{sum(paths) / len(paths) * 100:.0f}%" if paths else "—"
        print(f"  {sig:<16}{len(rs):>5}"
              f"{pct(med(r.get('24h') for r in rs))}"
              f"{pct(med(r.get('3d') for r in rs))}"
              f"{pct(med(r.get('7d') for r in rs))}"
              f"{pct(med(r.get('mfe') for r in rs))}"
              f"{pct(med(r.get('mae') for r in rs))}"
              f"{wr:>9}{len(paths):>9}")

    print(f"\n== 종목별 MFE 상위 12 (같은 종목 여러 번이면 최고치)\n")
    best = {}
    for r in detail:
        k = (r["sym"], r["sig"])
        if r.get("mfe") is None:
            continue
        if k not in best or r["mfe"] > best[k]["mfe"]:
            best[k] = r
    top = sorted(best.values(), key=lambda r: -r["mfe"])[:12]
    for r in top:
        kst = r["t"].tz_convert("Asia/Seoul").strftime("%m-%d %H시")
        print(f"  {r['sym'].replace('USDT',''):<12}{r['sig']:<15}{kst:>12}"
              f"  MFE {pct(r['mfe'])}  MAE {pct(r['mae'])}"
              f"  7d {pct(r.get('7d'))}")

    print(f"\n== MAE 하위 8 (가장 크게 밀린 자리)\n")
    worst = sorted((r for r in detail if r.get("mae") is not None),
                   key=lambda r: r["mae"])[:8]
    for r in worst:
        kst = r["t"].tz_convert("Asia/Seoul").strftime("%m-%d %H시")
        print(f"  {r['sym'].replace('USDT',''):<12}{r['sig']:<15}{kst:>12}"
              f"  MAE {pct(r['mae'])}  MFE {pct(r['mfe'])}"
              f"  7d {pct(r.get('7d'))}")


def _turnover_sweep(sess) -> None:
    """유니버스 전체를 받아 하한별 통과 종목 수와 최근 알림 생존 여부를 찍는다."""
    from invest_signal import config as cfg_mod
    from invest_signal import scanner

    cfg = cfg_mod.load()
    c = cfg.get("crypto") or {}
    source, syms = data_binance.resolve_source(sess, c.get("source", "auto"),
                                               set(c.get("exclude") or []), print)
    frames = data_binance.fetch_all(sess, syms, source, limit=750, log=print,
                                    workers=int(c.get("fetch_workers", 6)))
    ticker = {}
    try:
        ticker = data_binance.ticker_24h(sess, source)
    except Exception as e:                          # noqa: BLE001
        print(f"  24h 티커 조회 실패({e}) — 크립토 모멘텀은 건너뜀")

    scfg = cfg.get("signal") or {}
    keys = ["uptrend_onset", "pump_early", "wave_setup"]
    print(f"\n== 거래대금 하한 스윕 · {source} · {len(frames)}종\n")
    header = "  하한  " + "".join(f"{k:>14}" for k in keys) + f"{'모멘텀후보':>12}"
    print(header)
    eligible = {}
    for floor in FLOORS:
        cells = []
        for k in keys:
            rcfg = dict((scfg.get(k) or {}).get("crypto_rank_filter") or {})
            rcfg["min_turnover_usd"] = floor
            e = scanner._crypto_rank_eligible(frames, rcfg)
            eligible[(k, floor)] = e
            cells.append(f"{len(e):>14}")
        lb = sum(1 for s in syms
                 if (ticker.get(s) or {}).get("quote_volume", 0) >= floor)
        eligible[("leader_break", floor)] = {
            s for s in syms if (ticker.get(s) or {}).get("quote_volume", 0) >= floor}
        print(f"  ${floor / 1e6:>4.0f}M" + "".join(cells) + f"{lb:>12}")

    recent = _recent_alert_symbols(RECENT_MSGS)
    if not recent:
        print("\n  (state/sent_log.jsonl 없음 — 최근 알림 대조 생략)")
        return
    print(f"\n== 최근 {RECENT_MSGS}개 메시지에 등장한 종목의 생존 여부\n")
    for key in keys + ["leader_break"]:
        got = sorted(recent.get(key, ()))
        if not got:
            continue
        print(f"  [{key}] 등장 {len(got)}종")
        for floor in FLOORS:
            alive = [s for s in got if s in eligible[(key, floor)]]
            cut = [s for s in got if s not in eligible[(key, floor)]]
            note = ", ".join(x.replace("USDT", "") for x in cut[:12])
            if len(cut) > 12:
                note += f" 외 {len(cut) - 12}"
            print(f"    ${floor / 1e6:>4.0f}M → {len(alive):>3}종 생존"
                  + (f" · 컷: {note}" if cut else ""))
        print()

def _leader_segment(sess) -> None:
    """⚡ 알림을 **발화 시점 피처**로 쪼개, 대박과 쪽박을 가르는 축을 찾는다.

    ⚡는 성과가 반으로 갈린다 — 최근 10일 실측에서 MFE 상위도 MAE 하위도
    전부 ⚡였다. 중앙값 하나로는 그 분포를 못 보므로, 알림 하나하나에
    발화 봉 시점의 피처(그때까지의 상승폭·확장도·변동성·거래대금·재알림
    횟수 등)를 붙이고 사후 결과(MFE/MAE/전진수익률)와 교차한다.

    출력 세 단계:
      ① 대박(MFE 상위)/쪽박(MAE 하위) 그룹의 피처 중앙값 대조
      ② 피처별 사분위 버킷 성과표
      ③ 단일·2중 조건 룰 시뮬레이션 — 잔존율 대비 대박률/쪽박률
    """
    import concurrent.futures as cf
    import json
    import statistics as st

    import pandas as pd

    from invest_signal import config as cfg_mod

    back = int(os.environ.get("PROBE_DAYS_BACK", "10"))
    big = float(os.environ.get("PROBE_BIG", "20")) / 100      # 대박 기준 MFE
    bad = float(os.environ.get("PROBE_BAD", "15")) / 100      # 쪽박 기준 MAE
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "state", "alerts_state.json"), encoding="utf-8") as f:
        alerts = json.load(f).get("alerts", {})

    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(days=back)
    events = []
    for key in alerts:
        parts = key.split("|")
        if len(parts) < 3 or parts[1] != "leader_break":
            continue
        try:
            t = pd.Timestamp(parts[2])
        except ValueError:
            continue
        if t.tz is None:
            t = t.tz_localize("UTC")
        if t >= cutoff:
            events.append((parts[0], t))
    if not events:
        print(f"최근 {back}일 ⚡ 알림 없음")
        return
    events.sort(key=lambda e: e[1])

    # 재알림 횟수 — 같은 종목의 직전 72시간 내 ⚡ 알림 수
    seen: dict[str, list] = {}
    repeats = {}
    for sym, t in events:
        prior = seen.setdefault(sym, [])
        repeats[(sym, t)] = sum(1 for p in prior if t - p <= pd.Timedelta(hours=72))
        prior.append(t)

    cfg = cfg_mod.load()
    c = cfg.get("crypto") or {}
    source, syms = data_binance.resolve_source(sess, c.get("source", "auto"),
                                               set(c.get("exclude") or []), print)
    want = sorted({s for s, _ in events} & set(syms))
    print(f"\n최근 {back}일 ⚡ 알림 {len(events)}건 · 종목 {len(want)}종 "
          f"(유니버스 밖 {len({s for s, _ in events}) - len(want)}종 제외) · {source}")

    workers = int(c.get("fetch_workers", 6))
    # 유니버스 전체를 받는다 — 알림 시점의 '24h 상승률 순위'를 복원하려면
    # 알림 종목만으로는 안 되고 그때의 횡단면이 통째로 필요하다.
    frames4 = data_binance.fetch_all(sess, syms, source, limit=750,
                                     log=print, workers=workers)
    closes = pd.DataFrame({s: f["Close"] for s, f in frames4.items() if len(f)})
    ranks = (closes / closes.shift(6) - 1).rank(axis=1, ascending=False)

    def fetch15(sym):
        s = requests.Session()
        try:
            return sym, data_binance.klines(s, sym, source, "15m", limit=1000)
        except Exception:                              # noqa: BLE001
            return sym, None
        finally:
            s.close()

    frames15 = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for sym, df in ex.map(fetch15, want):
            if df is not None and len(df):
                frames15[sym] = df
    print(f"[15m] {len(frames15)}/{len(want)}종 수집")

    btc = frames4.get("BTCUSDT")
    D = pd.Timedelta
    rows = []
    for sym, t in events:
        d15, d4 = frames15.get(sym), frames4.get(sym)
        if d15 is None or d4 is None or len(d4) < 50:
            continue
        i = int(d15.index.searchsorted(t, "left"))
        if i >= len(d15) or d15.index[i] != t or i < 96:
            continue                       # 15m 창(≈10일) 밖이거나 24h 역산 불가
        entry = float(d15["Close"].iloc[i])
        j = int(d4.index.searchsorted(t, "right")) - 1
        if entry <= 0 or j < 20:
            continue
        c15, c4 = d15["Close"], d4["Close"]

        def back_ret(series, k, idx):
            return None if idx - k < 0 else entry / float(series.iloc[idx - k]) - 1

        ma20 = indicators.sma(c15, 20).iloc[i]
        r = {
            "sym": sym, "t": t,
            "재알림": repeats[(sym, t)],
            "이탈깊이": (entry / float(ma20) - 1) if pd.notna(ma20) else None,
            "1h상승": back_ret(c15, 4, i),
            "4h상승": back_ret(c15, 16, i),
            "24h상승": back_ret(c15, 96, i),
            "3d상승": back_ret(c4, 18, j),
            "7d상승": back_ret(c4, 42, j),
            "이력일": round(j / 6, 1),
        }
        # 발화 시점 24h 상승률 순위 — 15위 안이면 '현재 주도주', 밖이면
        # watch_days로 이월된 감시분이다(선정 기준이 순위라 이 구분이 핵심).
        try:
            rk = ranks.at[d4.index[j], sym]
            r["순위"] = None if pd.isna(rk) else float(rk)
            r["이월분"] = None if pd.isna(rk) else float(rk > 15)
        except KeyError:
            r["순위"] = r["이월분"] = None
        for p, name in ((120, "120선확장"), (480, "480선확장")):
            m = indicators.sma(c4, p).iloc[j] if j >= p else float("nan")
            r[name] = (entry / float(m) - 1) if pd.notna(m) else None
        hi, lo, cl = d4["High"], d4["Low"], c4
        tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()],
                       axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[j]
        r["변동성"] = (float(atr) / entry) if pd.notna(atr) else None
        seg_t = d4.iloc[max(0, j - 5):j + 1]
        r["거래대금M"] = float((seg_t["Close"] * seg_t["Volume"]).sum()) / 1e6
        if btc is not None and len(btc):
            jb = int(btc.index.searchsorted(t, "right")) - 1
            if jb >= 6:
                r["BTC24h"] = float(btc["Close"].iloc[jb]) / float(btc["Close"].iloc[jb - 6]) - 1

        fwd = d4[(d4.index > t) & (d4.index <= t + D(days=7))]
        if not len(fwd):
            continue
        r["진행h"] = round((fwd.index[-1] - t) / D(hours=1))
        r["mfe"] = float(fwd["High"].max()) / entry - 1
        r["mae"] = float(fwd["Low"].min()) / entry - 1
        for name, dl in (("24h", D(hours=24)), ("3d", D(days=3)), ("7d", D(days=7))):
            k = int(d4.index.searchsorted(t + dl, "left"))
            r[name] = (float(c4.iloc[k]) / entry - 1) if k < len(d4) else None
        up = dn = None
        for x in range(len(fwd)):
            if up is None and float(fwd["High"].iloc[x]) / entry - 1 >= 0.10:
                up = x
            if dn is None and float(fwd["Low"].iloc[x]) / entry - 1 <= -0.10:
                dn = x
            if up is not None or dn is not None:
                break
        r["path"] = None if up is None and dn is None else (up is not None and (dn is None or up <= dn))
        r["대박"] = r["mfe"] >= big
        r["쪽박"] = r["mae"] <= -bad
        rows.append(r)

    if not rows:
        print("분석 가능한 알림 없음 (15m 창 밖이거나 프레임 부족)")
        return

    FEATS = ["순위", "이월분", "재알림", "24h상승", "3d상승", "7d상승", "4h상승",
             "1h상승", "이탈깊이", "120선확장", "480선확장", "변동성", "거래대금M",
             "이력일", "BTC24h"]

    def med(vals):
        vals = [v for v in vals if v is not None]
        return None if not vals else st.median(vals)

    def pct(v, w=7):
        return f"{'—':>{w}}" if v is None else f"{v * 100:+{w - 1}.1f}%"

    def raw(v, w=7):
        return f"{'—':>{w}}" if v is None else f"{v:>{w}.2f}"

    fmt = {"재알림": raw, "거래대금M": raw, "이력일": raw, "순위": raw, "이월분": raw}
    mature = [r for r in rows if r["진행h"] >= 24 * 6]
    print(f"\n분석 대상 {len(rows)}건 · 7일 관찰 완료 {len(mature)}건 · "
          f"대박 기준 MFE≥{big:+.0%} / 쪽박 기준 MAE≤{-bad:.0%}")
    hit = sum(r["대박"] for r in rows) / len(rows)
    miss = sum(r["쪽박"] for r in rows) / len(rows)
    print(f"전체 대박률 {hit:.0%} · 쪽박률 {miss:.0%} · 둘 다 "
          f"{sum(r['대박'] and r['쪽박'] for r in rows) / len(rows):.0%}")

    # ① 대박/쪽박 그룹 피처 대조
    k = max(10, len(rows) // 5)
    top = sorted(rows, key=lambda r: -r["mfe"])[:k]
    bot = sorted(rows, key=lambda r: r["mae"])[:k]
    print(f"\n== ① 대박 상위 {k}건 vs 쪽박 하위 {k}건 · 발화 시점 피처 중앙값\n")
    print(f"  {'피처':<12}{'대박군':>10}{'쪽박군':>10}{'전체':>10}   갈림")
    for f in FEATS:
        a, b, w = med(r.get(f) for r in top), med(r.get(f) for r in bot), med(r.get(f) for r in rows)
        g = fmt.get(f, pct)
        gap = "" if a is None or b is None or not w else (
            "◀◀" if a < b * 0.7 else "▶▶" if a > b * 1.4 else "")
        print(f"  {f:<12}{g(a, 10)}{g(b, 10)}{g(w, 10)}   {gap}")

    # ② 피처별 사분위 버킷
    def stat(rs):
        paths = [r["path"] for r in rs if r["path"] is not None]
        return (f"{len(rs):>5}"
                f"{pct(med(r['mfe'] for r in rs), 8)}{pct(med(r['mae'] for r in rs), 8)}"
                f"{pct(med(r.get('24h') for r in rs), 8)}{pct(med(r.get('3d') for r in rs), 8)}"
                f"{sum(r['대박'] for r in rs) / len(rs) * 100:>7.0f}%"
                f"{sum(r['쪽박'] for r in rs) / len(rs) * 100:>7.0f}%"
                f"{(sum(paths) / len(paths) * 100 if paths else 0):>7.0f}%")

    head = (f"  {'구간':<20}{'건수':>5}{'MFE':>8}{'MAE':>8}{'24h':>8}{'3d':>8}"
            f"{'대박':>8}{'쪽박':>8}{'경로':>8}")
    print("\n== ② 피처별 사분위 성과\n")
    for f in FEATS:
        vals = sorted(r[f] for r in rows if r.get(f) is not None)
        if len(vals) < 40:
            continue
        qs = [vals[int(len(vals) * q)] for q in (0.25, 0.5, 0.75)]
        if len(set(qs)) < 3:
            continue
        print(f"  [{f}]")
        print(head)
        g = fmt.get(f, pct)
        edges = [(None, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], None)]
        for lo, hi in edges:
            rs = [r for r in rows if r.get(f) is not None
                  and (lo is None or r[f] >= lo) and (hi is None or r[f] < hi)]
            if not rs:
                continue
            lab = f"{'  ' if lo is None else g(lo, 7)}~{'' if hi is None else g(hi, 7)}"
            print(f"  {lab:<20}" + stat(rs))
        print()

    print("  [상위권 vs 이월 감시분]")
    print(head)
    for lab, sel in (("현재 상위 15위 안", lambda r: r.get("이월분") == 0.0),
                     ("이월 감시분(15위 밖)", lambda r: r.get("이월분") == 1.0)):
        rs = [r for r in rows if sel(r)]
        if rs:
            print(f"  {lab:<20}" + stat(rs))
    print()

    # ③ 룰 시뮬레이션
    print("== ③ 단일 조건 룰 — 잔존율 대비 대박/쪽박 (전체 대비 개선폭 순)\n")
    print(f"  {'조건':<26}{'잔존':>7}{'대박':>7}{'쪽박':>7}{'MFE':>8}{'MAE':>8}{'3d':>8}{'점수':>7}")
    cands = []
    for f in FEATS:
        vals = sorted(r[f] for r in rows if r.get(f) is not None)
        if len(vals) < 40:
            continue
        for q in (0.25, 0.5, 0.75):
            cut = vals[int(len(vals) * q)]
            g = fmt.get(f, pct)
            for op, sel in ((">=", lambda r, f=f, c=cut: r.get(f) is not None and r[f] >= c),
                            ("<", lambda r, f=f, c=cut: r.get(f) is not None and r[f] < c)):
                rs = [r for r in rows if sel(r)]
                if len(rs) < len(rows) * 0.15:
                    continue
                h = sum(r["대박"] for r in rs) / len(rs)
                m = sum(r["쪽박"] for r in rs) / len(rs)
                cands.append(((h - hit) - (m - miss), f"{f} {op} {g(cut, 8).strip()}", rs, sel))
    cands.sort(key=lambda x: -x[0])
    for score, name, rs, _ in cands[:12]:
        print(f"  {name:<26}{len(rs) / len(rows) * 100:>6.0f}%"
              f"{sum(r['대박'] for r in rs) / len(rs) * 100:>6.0f}%"
              f"{sum(r['쪽박'] for r in rs) / len(rs) * 100:>6.0f}%"
              f"{pct(med(r['mfe'] for r in rs), 8)}{pct(med(r['mae'] for r in rs), 8)}"
              f"{pct(med(r.get('3d') for r in rs), 8)}{score * 100:>+6.0f}p")

    print("\n== ③-2 두 조건 조합 (상위 단일 조건끼리)\n")
    print(f"  {'조건':<44}{'잔존':>7}{'대박':>7}{'쪽박':>7}{'MFE':>8}{'MAE':>8}{'3d':>8}")
    pairs = []
    for a in range(min(6, len(cands))):
        for b in range(a + 1, min(6, len(cands))):
            _, na, _, sa = cands[a]
            _, nb, _, sb = cands[b]
            if na.split()[0] == nb.split()[0]:
                continue
            rs = [r for r in rows if sa(r) and sb(r)]
            if len(rs) < len(rows) * 0.08:
                continue
            h = sum(r["대박"] for r in rs) / len(rs)
            m = sum(r["쪽박"] for r in rs) / len(rs)
            pairs.append(((h - hit) - (m - miss), f"{na}  +  {nb}", rs))
    pairs.sort(key=lambda x: -x[0])
    for _, name, rs in pairs[:8]:
        print(f"  {name:<44}{len(rs) / len(rows) * 100:>6.0f}%"
              f"{sum(r['대박'] for r in rs) / len(rs) * 100:>6.0f}%"
              f"{sum(r['쪽박'] for r in rs) / len(rs) * 100:>6.0f}%"
              f"{pct(med(r['mfe'] for r in rs), 8)}{pct(med(r['mae'] for r in rs), 8)}"
              f"{pct(med(r.get('3d') for r in rs), 8)}")

    print(f"\n  ※ 표본 {len(rows)}건·10일치라 과최적화 위험이 있다. 룰은 "
          f"'대박률이 오르면서 잔존율이 60% 이상'인 것만 실전 후보로 본다.")

with requests.Session() as sess:
    _mode = os.environ.get("PROBE_MODE", "").lower()
    if _mode == "turnover":
        _turnover_sweep(sess)
        raise SystemExit(0)
    if _mode == "perf":
        _performance(sess)
        raise SystemExit(0)
    if _mode in ("leaders", "segment"):
        _leader_segment(sess)
        raise SystemExit(0)
    for sym in symbols:
        try:
            df = data_binance.klines_4h(sess, sym, "fapi", limit=750,
                                        include_live=True)
        except Exception as e:                      # noqa: BLE001
            print(f"== {sym}: 수집 실패 {e}")
            continue
        mv = indicators.monthly_vwap(df)
        qv = indicators.quarterly_vwap(df)
        m120 = indicators.sma(df["Close"], 120)
        m240 = indicators.sma(df["Close"], 240)
        m480 = indicators.sma(df["Close"], 480)
        print(f"== {sym} — 최근 10봉 (봉 오픈시각 KST, 마지막 줄은 진행봉)")
        for i in range(max(0, len(df) - 10), len(df)):
            t = df.index[i].tz_convert("Asia/Seoul").strftime("%m-%d %H시")
            touch = bool(df["Low"].iloc[i] <= mv.iloc[i] <= df["High"].iloc[i])
            c = df["Close"].iloc[i]
            print(f"  {t}: O {df['Open'].iloc[i]:.6g} H {df['High'].iloc[i]:.6g}"
                  f" L {df['Low'].iloc[i]:.6g} C {c:.6g}"
                  f" | MVWAP {mv.iloc[i]:.6g} 터치={touch}"
                  f" | QVWAP {qv.iloc[i]:.6g} {'위' if c > qv.iloc[i] else '아래'}"
                  f" | 120 {m120.iloc[i]:.6g} 240 {m240.iloc[i]:.6g}"
                  f" 480 {m480.iloc[i]:.6g}")
        # 일자별 거래량·거래대금 — 4h봉을 UTC 날짜로 묶는다.
        # 거래대금은 봉마다 종가×거래량을 더한 근사치다(정확한 quote volume은
        # kline의 별도 필드지만 이 프레임은 OHLCV만 들고 있다).
        daily = df.resample("1D").agg({"Open": "first", "High": "max",
                                       "Low": "min", "Close": "last",
                                       "Volume": "sum"}).dropna()
        turnover = (df["Close"] * df["Volume"]).resample("1D").sum()
        want = [d.strip() for d in os.environ.get("PROBE_DAYS", "").split(",") if d.strip()]
        rows = ([daily.loc[[d]] for d in want if d in daily.index.strftime("%Y-%m-%d")]
                if want else [daily.tail(10)])
        print(f"  -- 일봉 요약 (UTC{'  · 지정일' if want else ' · 최근 10일'})")
        for chunk in rows:
            for t, r in chunk.iterrows():
                key = t.strftime("%Y-%m-%d")
                print(f"     {key}  O {r.Open:.6g} H {r.High:.6g} L {r.Low:.6g}"
                      f" C {r.Close:.6g}  거래량 {r.Volume:,.0f}"
                      f"  거래대금 ${turnover.loc[t] / 1e6:,.2f}M")
        for d in want:
            if d not in daily.index.strftime("%Y-%m-%d"):
                print(f"     {d}: 프레임 범위 밖 (750봉 = 약 125일)")

        closed = df.iloc[:-1]                       # 시그널 재현은 마감봉 기준
        for mod in PROBES:
            evs = mod.detect(closed, sym,
                             dataclasses.replace(mod.Params(), grace_bars=12))
            if not evs:
                print(f"  {mod.NAME}: 최근 12봉 내 발화 없음")
            for e in evs:
                kst = e.bar_time.tz_convert("Asia/Seoul").strftime("%m-%d %H시")
                print(f"  {mod.NAME} 발화: {kst} KST · {e.detail}")
