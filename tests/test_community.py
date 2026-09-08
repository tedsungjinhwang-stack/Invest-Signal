"""📣 커뮤니티 언급 — 수집·집계·표시 테스트.

네트워크는 안 탄다. 실제 HTML/JSON 모양만 떼어 와 파서에 물린다.
"""

import collections

from invest_signal import data_community as dc
from invest_signal import notify, scanner


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
    _, un = dc.count_mentions(["하닉 동양철관 살까"] * 3, UNIV, ALIAS)
    assert "하닉" not in un
    assert un["동양철관"] == 3


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
    _, un = dc.count_mentions(["메가스터디 왔다", "동양철관 마감"], UNIV, {})
    assert un["메가스터디"] == 1 and un["동양철관"] == 1


def test_unmatched_strips_verb_endings():
    """'찐반이다'가 '찐반'과 다른 말로 세어지면 불용어를 넣어도 계속 올라온다."""
    _, un = dc.count_mentions(["동양철관이다 진짜", "동양철관 간다"], UNIV, {})
    assert un["동양철관"] == 2


def test_unmatched_skips_stopwords():
    _, un = dc.count_mentions(["오늘 매수 존버 마스"], UNIV, {})
    assert list(un) == ["마스"]


def test_count_mentions_survives_unquoted_yaml_code():
    """000660을 따옴표 없이 쓰면 YAML이 정수로 읽는다 — 터지진 않게 한다."""
    hits, _ = dc.count_mentions(["하닉 가즈아"], UNIV, {"하닉": 432})
    assert hits["432"] == 1


# ── 표시 ─────────────────────────────────────────────────────────────────
def test_community_lines_splits_by_community():
    """커뮤니티마다 머리줄 + 티커 줄 + 인용 줄. 합치지 않는다."""
    lines = notify._community_lines({"sources": [
        {"emoji": "🇺🇸", "label": "r/wallstreetbets", "note": "24h 증가순",
         "rows": ["MU 59회 +37"], "quotes": ["MU earnings play"]},
        {"emoji": "📈", "label": "차트 마이너", "note": "196글",
         "rows": ["BTC 14회"], "quotes": ["비트 개쏜다"]}]})
    body = "\n".join(lines)
    assert "r/wallstreetbets</b> · 24h 증가순" in body
    assert "차트 마이너</b> · 196글" in body
    assert body.count("💬") == 2
    assert body.index("wallstreetbets") < body.index("차트 마이너")   # 설정 순서


def test_community_lines_empty_when_nothing_collected():
    assert notify._community_lines({}) == []
    assert notify._community_lines({"sources": []}) == []


def test_community_lines_escapes_user_text():
    """사람 글을 그대로 싣는다 — `<`가 새면 텔레그램이 메시지를 통째로 버린다."""
    body = "\n".join(notify._community_lines({"sources": [
        {"label": "x", "rows": ["A&B 1회"], "quotes": ["<b>가짜</b> 태그 & 기호"]}]}))
    assert "&lt;b&gt;가짜&lt;/b&gt;" in body and "&amp;" in body
    assert "<b>가짜" not in body


def test_community_lines_leaves_quotes_alone():
    """따옴표까지 escape 하면 `&#x27;`가 화면에 그대로 찍힌다."""
    body = "\n".join(notify._community_lines({"sources": [
        {"label": "x", "rows": [], "quotes": ["debuts 'LAPTOP' memecoin"]}]}))
    assert "debuts 'LAPTOP' memecoin" in body


def test_community_lines_unmatched_is_one_shared_row():
    body = "\n".join(notify._community_lines({
        "sources": [{"label": "x", "rows": ["BTC 5회"], "quotes": []}],
        "unmatched": [("소폰", 10), ("찐반", 5)]}))
    assert body.count("❓") == 1
    assert "소폰 10 · 찐반 5" in body


def test_format_events_appends_community_at_the_end():
    text = notify.format_events([], [], {}, community={
        "sources": [{"label": "차트 마이너", "rows": ["BTC 28회"], "quotes": []}]})
    assert "📣" in text and text.index("📣") > 0
    assert "📣" not in notify.format_events([], [], {})


# ── 인용문 고르기 ─────────────────────────────────────────────────────────
def test_pick_quotes_prefers_titles_about_the_listed_tickers():
    titles = ["재미난 짤 하나", "MU earnings play", "고양이 사진"]
    assert dc.pick_quotes(titles, ["MU"], set(), {}, n=2)[0] == "MU earnings play"


def test_pick_quotes_matches_tickers_outside_our_universe():
    """숫자 줄에 오른 티커는 유니버스에 없어도 인용문 매칭에 써야 한다.

    레딧 주식 커뮤니티의 MU·SPY는 우리 퍼프 유니버스에 없다 — 안 넣으면
    정작 그 종목을 말하는 글을 못 고른다.
    """
    got = dc.pick_quotes(["아무 얘기나 하는 글", "SPY 500 puts"], ["SPY"],
                         set(), {}, n=1)
    assert got == ["SPY 500 puts"]


def test_pick_quotes_fills_from_the_rest_and_dedups():
    got = dc.pick_quotes(["똑같은 제목이다 이건", "똑같은 제목이다 이건", "다른 제목이다 이건"],
                         ["BTC"], set(), {}, n=3)
    assert got == ["똑같은 제목이다 이건", "다른 제목이다 이건"]


def test_pick_quotes_drops_price_bot_posts():
    """`[14:30] BTC $78791 …`은 매시간 올라오는 봇 글이지 의견이 아니다."""
    got = dc.pick_quotes(["[14:30] BTC $78791 ETH $2481", "비트 이더로는 못 번다"],
                         ["BTC"], {"BTC"}, {}, n=2)
    assert got == ["비트 이더로는 못 번다"]


def test_pick_quotes_keeps_one_line_per_topic():
    """같은 화제가 한꺼번에 올라와 인용 세 줄을 통째로 먹으면 안 된다."""
    got = dc.pick_quotes(["테더 2000 가는거 아니였냐", "테더 근데 일정액은 들고있지",
                          "테더 기본 몇십억은 들고있지", "완전 다른 얘기를 하는 글"],
                         ["BTC"], set(), {}, n=3)
    assert got == ["테더 2000 가는거 아니였냐", "완전 다른 얘기를 하는 글"]


def test_pick_quotes_clips_long_titles():
    got = dc.pick_quotes(["가" * 200], ["BTC"], set(), {}, n=1, width=30)
    assert len(got[0]) == 30 and got[0].endswith("…")


# ── 스톡트윗 ─────────────────────────────────────────────────────────────
def test_stocktwits_trending_strips_crypto_suffix():
    s = _Session(_Resp(payload={"symbols": [{"symbol": "ORCL"},
                                            {"symbol": "BTC.X"}]}))
    assert dc.stocktwits_trending(s) == ["ORCL", "BTC"]


def test_stocktwits_stream_keeps_author_sentiment():
    """강세·약세는 **글쓴이가 직접 단 꼬리표**다 — 우리가 추측한 게 아니다."""
    s = _Session(_Resp(payload={"messages": [
        {"body": "$ORCL raised TP", "entities": {"sentiment": {"basic": "Bullish"}}},
        {"body": "$QQQ still bullish???", "entities": {"sentiment": None}}]}))
    assert dc.stocktwits_stream(s, "ORCL") == [
        ("Bullish", "raised TP"), ("", "still bullish???")]


def test_stocktwits_quotes_skip_cross_symbol_spam():
    """한 사람이 티커를 줄줄이 달아 뿌린 글이 인용 세 줄을 다 먹으면 안 된다."""
    spam = {"body": "$INTC $ORCL open 105. Close 115",
            "entities": {"sentiment": {"basic": "Bullish"}}}
    real = {"body": "$ORCL raised TP after earnings",
            "entities": {"sentiment": {"basic": "Bullish"}}}
    s = _Session(_Resp(payload={"symbols": [{"symbol": "INTC"}, {"symbol": "ORCL"}]}),
                 _Resp(payload={"messages": [spam]}),
                 _Resp(payload={"messages": [spam, real]}))
    block, _ = scanner._community_block(
        s, {"kind": "stocktwits", "label": "스톡트윗", "streams": 2},
        set(), {}, 5, 3, 3, 70, lambda *_: None)
    assert block["rows"] == ["INTC 🟢1·🔴0", "ORCL 🟢2·🔴0"]
    assert block["quotes"] == ["INTC 🟢 open 105. Close 115",
                               "ORCL 🟢 raised TP after earnings"]


def test_clean_body_drops_cashtags_and_links():
    assert dc.clean_body("$SPY $MU  Good Lord https://x.co/a selling") == \
        "Good Lord selling"


# ── 레딧 RSS ─────────────────────────────────────────────────────────────
_ATOM = ('<feed><entry><title>{a}</title></entry>'
         '<entry><title>{b}</title></entry></feed>')


def test_reddit_titles_skips_pinned_megathreads():
    """고정 메가스레드는 매번 맨 위라 '지금 무슨 얘기 중인가'에 정보가 없다."""
    s = _Session(_Resp(text=_ATOM.format(
        a="Daily Discussion Thread - Sep 07", b="Is RDDT undervalued?")))
    assert dc.reddit_titles(s, "stocks") == ["Is RDDT undervalued?"]


def test_reddit_titles_retries_on_429():
    """429는 막힌 게 아니라 잠깐 밀린 것이다 — 실측 10~15초면 풀린다."""
    import requests as rq

    class _Throttled:
        def __init__(self):
            self.n = 0

        def get(self, *a, **k):
            self.n += 1
            if self.n == 1:
                r = _Resp(text="")
                err = rq.HTTPError("429")
                err.response = type("R", (), {"status_code": 429})()
                raise err
            return _Resp(text=_ATOM.format(a="첫 글이다", b="둘째 글이다"))

    dc.time.sleep = lambda *_: None              # 테스트에서 진짜로 쉬지 않는다
    s = _Throttled()
    assert dc.reddit_titles(s, "stocks") == ["첫 글이다", "둘째 글이다"]
    assert s.n == 2


def test_reddit_titles_gives_up_quietly():
    """인용문이 빠질 뿐, 숫자 줄은 ApeWisdom에서 따로 온다."""
    class _Dead:
        def get(self, *a, **k):
            raise RuntimeError("boom")
    logged = []
    assert dc.reddit_titles(_Dead(), "stocks", log=logged.append) == []
    assert logged and "레딧" in logged[0]
