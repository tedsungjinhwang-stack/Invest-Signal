"""📣 커뮤니티 언급 — 수집·집계·표시 테스트.

네트워크는 안 탄다. 실제 HTML/JSON 모양만 떼어 와 파서에 물린다.
"""

import collections

from invest_signal import data_community as dc
from invest_signal import notify


class _Resp:
    def __init__(self, text="", payload=None):
        self.text = text
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class _Session:
    """get() 호출을 순서대로 미리 준비한 응답으로 받아 주는 가짜 세션."""

    def __init__(self, *responses):
        self._queue = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        if not self._queue:
            raise AssertionError("준비 안 된 요청: " + url)
        return self._queue.pop(0)


# ── ApeWisdom ────────────────────────────────────────────────────────────
def test_apewisdom_strips_crypto_suffix_and_keeps_prev():
    """크립토 티커는 `BTC.X` 꼴로 온다 — 접미사를 떼야 유니버스와 맞는다."""
    s = _Session(_Resp(payload={"pages": 1, "results": [
        {"ticker": "BTC.X", "name": "Bitcoin", "mentions": 40,
         "mentions_24h_ago": 31, "rank": 1},
        {"ticker": "ICP.X", "name": "Internet Computer", "mentions": 6,
         "mentions_24h_ago": 2, "rank": 2},
    ]}))
    rows = dc.apewisdom(s, "all-crypto")
    assert [r["ticker"] for r in rows] == ["BTC", "ICP"]
    assert rows[0]["mentions"] == 40 and rows[0]["prev"] == 31


def test_apewisdom_stops_at_last_page():
    s = _Session(_Resp(payload={"pages": 2, "results": [
                          {"ticker": "MU", "mentions": 5, "mentions_24h_ago": 1}]}),
                 _Resp(payload={"pages": 2, "results": [
                          {"ticker": "BE", "mentions": 4, "mentions_24h_ago": 0}]}))
    rows = dc.apewisdom(s, "all-stocks", pages=5)
    assert [r["ticker"] for r in rows] == ["MU", "BE"]
    assert len(s.calls) == 2          # pages=5를 줬어도 2페이지에서 멈춘다


def test_apewisdom_failure_is_empty_not_raise():
    """이 칸은 부가 정보다 — 실패가 스캔 전체를 세우면 안 된다."""
    class _Boom:
        def get(self, *a, **k):
            raise RuntimeError("503")
    logged = []
    assert dc.apewisdom(_Boom(), "all-crypto", log=logged.append) == []
    assert logged and "실패" in logged[0]


# ── 디시 목록 파싱 ────────────────────────────────────────────────────────
_ROW_TPL = ('<tr class="ub-content us-post" data-no="{no}">'
            '<td class="gall_num">{no}</td>'
            '<td class="gall_tit ub-word">  <a  href="/mgallery/board/view/?no={no}">'
            '{title}</a> <a class="reply_numbox"><span class="reply_num">[3]</span></a>'
            '</td></tr>')
_NOTICE_TPL = ('<tr class="ub-content us-post">'
               '<td class="gall_num"><em class="icon_img icon_notice"></em></td>'
               '<td class="gall_tit ub-word"> <a href="/x">공지: 갤러리 이용 안내</a>'
               '</td></tr>')


def test_dc_titles_reads_real_posts_and_skips_notices():
    """일반 글은 `<a  href=`(공백 두 칸), 공지는 한 칸이다.

    \\s+로 안 받으면 공지만 잡혀서 목록이 통째로 비어 보인다 — 실제로 한 번
    그렇게 틀렸던 자리라 회귀 테스트로 박아 둔다.
    """
    html = "<table>" + _NOTICE_TPL + _ROW_TPL.format(no=1, title="비트 가즈아") \
        + _ROW_TPL.format(no=2, title="하닉 매수각") + "</table>"
    s = _Session(_Resp(text=html))
    assert dc.dc_titles(s, "chartanalysis", pages=1) == ["비트 가즈아", "하닉 매수각"]


def test_dc_titles_unescapes_and_strips_inner_tags():
    html = _ROW_TPL.format(no=1, title='<b>ETH</b> &gt;&gt; 전고점 &amp; 눌림')
    assert dc.dc_titles(_Session(_Resp(text=html)), "g", pages=1) == \
        ["ETH >> 전고점 & 눌림"]


def test_dc_titles_stops_on_closed_gallery():
    """폐쇄 갤러리를 계속 두드리지 않는다 — 설정을 고치라고 로그를 남긴다."""
    logged = []
    s = _Session(_Resp(text="해당 갤러리는 폐쇄되었습니다."))
    assert dc.dc_titles(s, "chart", pages=6, log=logged.append) == []
    assert len(s.calls) == 1
    assert logged and "폐쇄" in logged[0]


def test_dc_titles_minor_flag_picks_the_url_segment():
    s = _Session(_Resp(text=""), _Resp(text=""))
    dc.dc_titles(s, "chartanalysis", pages=1, minor=True)
    dc.dc_titles(s, "coin", pages=1, minor=False)
    assert "mgallery/board" in s.calls[0][0]
    assert "mgallery" not in s.calls[1][0]


def test_dc_titles_failure_keeps_earlier_pages():
    class _Half:
        def __init__(self):
            self.n = 0

        def get(self, *a, **k):
            self.n += 1
            if self.n == 1:
                return _Resp(text=_ROW_TPL.format(no=1, title="비트 가즈아"))
            raise RuntimeError("timeout")
    assert dc.dc_titles(_Half(), "g", pages=3, log=lambda _: None) == ["비트 가즈아"]


# ── 언급 집계 ────────────────────────────────────────────────────────────
UNIV = {"BTC", "ETH", "XRP", "SOL"}
ALIAS = {"비트": "BTC", "이더": "ETH", "하닉": "하이닉스", "코루": "KORU"}


def test_count_mentions_matches_universe_and_aliases():
    hits, _ = dc.count_mentions(
        ["비트 3만 간다", "ETH 눌림목", "하닉 슬슬 담는다", "코루 물렸다"], UNIV, ALIAS)
    assert hits == collections.Counter({"BTC": 1, "ETH": 1, "하이닉스": 1, "KORU": 1})


def test_count_mentions_counts_one_title_once():
    """제목은 짧다 — 반복은 강조지 별개 언급이 아니다."""
    hits, _ = dc.count_mentions(["비트 비트 BTC 비트코인"], UNIV, {"비트": "BTC"})
    assert hits["BTC"] == 1


def test_count_mentions_ignores_case():
    hits, _ = dc.count_mentions(["btc 존버", "Eth 반등"], UNIV, {})
    assert hits == collections.Counter({"BTC": 1, "ETH": 1})


def test_unmatched_drops_matched_alias_text():
    """이미 잡힌 별명은 후보에서 뺀다 — 안 그러면 상위를 자기가 먹는다."""
    _, un = dc.count_mentions(["하닉 반도체 살까"] * 3, UNIV, ALIAS)
    assert "하닉" not in un
    assert un["반도체"] == 3


def test_unmatched_strips_particles_and_fragments():
    """'하이닉스가'·'차트에서'가 통째로 오르거나 '에서'가 상위를 먹으면 안 된다."""
    _, un = dc.count_mentions(["소폰가 간다", "일봉에서 눌림"], UNIV, {})
    assert un["소폰"] == 1
    assert "소폰가" not in un and "에서" not in un and "내가" not in un


def test_unmatched_drops_standalone_particles():
    """띄어쓰기로 조사가 홀로 떨어진 것('4h 에서')은 후보가 아니다.

    두 글자라 조사를 떼지 않는(떼면 한 글자) 자리라, 따로 걸러야 한다.
    """
    _, un = dc.count_mentions(["4h 에서 소폰 부터 까지"], UNIV, {})
    assert list(un) == ["소폰"]


def test_unmatched_keeps_long_names_intact():
    """조사처럼 끝난다고 이름을 자르면 안 된다."""
    _, un = dc.count_mentions(["다큰낙타 왔다", "코스피 마감"], UNIV, {})
    assert un["다큰낙타"] == 1 and un["코스피"] == 1


def test_unmatched_skips_stopwords():
    _, un = dc.count_mentions(["오늘 매수 존버 마스"], UNIV, {})
    assert list(un) == ["마스"]


def test_count_mentions_survives_unquoted_yaml_code():
    """000660을 따옴표 없이 쓰면 YAML이 정수로 읽는다 — 터지진 않게 한다."""
    hits, _ = dc.count_mentions(["하닉 가즈아"], UNIV, {"하닉": 432})
    assert hits["432"] == 1


# ── 표시 ─────────────────────────────────────────────────────────────────
def _row(t, m, p):
    return {"ticker": t, "mentions": m, "prev": p}


def test_community_lines_marks_new_entries():
    lines = notify._community_lines({
        "overseas_crypto": [_row("TRUMP", 11, 0), _row("ICP", 6, 2)],
        "overseas_crypto_src": "all-crypto"})
    body = "\n".join(lines)
    assert "TRUMP 11회 신규" in body      # 직전 0 = 새로 뜬 것
    assert "ICP 6회 +4" in body


def test_community_lines_empty_when_nothing_collected():
    assert notify._community_lines({}) == []
    assert notify._community_lines({"korea_posts": 0}) == []


def test_community_lines_shows_korea_with_unmatched_hint():
    lines = notify._community_lines({
        "korea": [("BTC", 28), ("하이닉스", 11)], "korea_src": "chartanalysis",
        "korea_posts": 296, "korea_unmatched": [("반도체", 8), ("소폰", 4)]})
    body = "\n".join(lines)
    assert "chartanalysis 296글" in body
    assert "BTC 28회 · 하이닉스 11회" in body
    assert "❓ 반도체 8 · 소폰 4" in body


def test_format_events_appends_community_at_the_end():
    text = notify.format_events([], [], {}, community={
        "korea": [("BTC", 28)], "korea_src": "chartanalysis", "korea_posts": 296})
    assert "📣" in text
    assert text.index("📣") > 0
    assert "📣" not in notify.format_events([], [], {})
