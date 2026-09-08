"""커뮤니티 언급 집계 — 요즘 사람들이 뭘 얘기하는지.

**거르는 조건이 아니라 요약 표시다.** 시그널이 잡은 자리와 별개로, 커뮤니티가
갑자기 많이 부르기 시작한 종목을 알림 맨 아래 한 칸으로 보여 준다.

두 소스를 쓴다. 성격이 완전히 달라서 처리도 다르다.

  🌍 **해외 — ApeWisdom** (`apewisdom.io`)
     레딧(r/wallstreetbets·r/CryptoCurrency 등)의 티커 언급을 이미 집계해
     주는 무료 API다. 인증이 필요 없고 **티커로 정규화된 데이터에 24시간 전
     언급 수까지** 들어 있어, 우리가 할 일은 증가율로 줄 세우는 것뿐이다.

     > 레딧을 직접 긁는 건 포기했다 — 무인증 요청이 **403**으로 막힌다
     > (2023년 정책 변경). OAuth 앱을 등록하면 되지만 시크릿이 하나 더 늘고,
     > 티커 추출·집계를 우리가 다시 짜야 한다. ApeWisdom이 그걸 다 해 준다.

  🇰🇷 **국내 — 디시인사이드 갤러리** (기본 `chartanalysis` = 차트 마이너)
     공식 API가 없어 목록 HTML을 파싱한다. 제목만 본다 — 본문까지 열면
     글 하나에 요청이 하나씩 더 붙어 스캔이 느려진다.

     > **차트갤러리(`chart`)는 폐쇄됐다**("운영원칙 위반 — 갤러리 명칭
     > 변경"). 지금 살아 있는 건 마이너 갤러리 `chartanalysis`다.

**국내는 티커 매칭이 어렵다 — 이게 이 기능의 가장 약한 고리다.** 제목이
별명 투성이라(`아케` `마스` `코루` `소폰` `리천지` `반도치`), 6페이지 296글을
직접 매칭해 보니 **티커가 잡히는 글이 20%뿐**이고 그마저 BTC·ETH가 절반이었다.
BTC·ETH는 늘 1·2위라 신호 가치도 없다.

그래서 **매칭 안 된 상위 단어를 같이 싣는다.** 그걸 보고 `aliases` 사전을
채우면 다음 스캔부터 잡힌다 — 사전이 자라야 쓸모가 생기는 구조이고, 그
과정을 알림 자체가 도와주게 만든 것이다.
"""

import collections
import html
import re
import time

import requests

APEWISDOM = "https://apewisdom.io/api/v1.0/filter/{name}/page/{page}"
REDDIT_RSS = "https://www.reddit.com/r/{sub}/{sort}/.rss"
STOCKTWITS = "https://api.stocktwits.com/api/2/{path}.json"
DC_LIST = "https://gall.dcinside.com/{seg}/lists/"
# 디시는 봇 UA를 막는다 — 평범한 브라우저로 보여야 목록이 온다.
DC_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept-Language": "ko-KR,ko;q=0.9",
}

# 레딧도 기본 python UA를 막는다 — 디시와 같은 헤더를 쓴다.
REDDIT_HEADERS = {"User-Agent": DC_HEADERS["User-Agent"]}

_ENTRY = re.compile(r"<entry>(.*?)</entry>", re.S)
_ENTRY_TITLE = re.compile(r"<title>(.*?)</title>", re.S)
# 고정된 메가스레드는 매번 맨 위라 '지금 무슨 얘기 중인가'에 아무 정보가 없다.
_MEGA = re.compile(r"daily (discussion|thread)|rate my portfolio|weekend discussion"
                   r"|what are your moves|megathread|weekly (thread|discussion)"
                   r"|monthly (thread|discussion)|discussion thread", re.I)

_ROW = re.compile(r'<tr class="ub-content us-post"(.*?)</tr>', re.S)
# `<a  href=` — 일반 글은 공백이 **두 칸**이고 공지는 한 칸이다. \s+로 안 받으면
# 공지만 잡혀서 목록이 통째로 비어 보인다(실제로 그렇게 한 번 틀렸다).
_TITLE = re.compile(r'class="gall_tit ub-word">\s*<a\s+href="[^"]*"[^>]*>(.*?)</a>', re.S)
_TAG = re.compile(r"<[^>]+>")
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9]{1,9}")
_HANGUL = re.compile(r"[가-힣]+")
# 조사를 떼고 본다. 안 떼면 '하이닉스가' '차트에서'가 통째로 후보에 오르거나
# 조각('에서' '내가')이 상위를 먹어서, 여덟 자리뿐인 ❓ 목록이 낭비된다.
# 긴 것부터 봐야 '에서'가 '서'로 잘리지 않는다.
_PARTICLES = ("으로부터", "에서는", "에서", "부터", "까지", "이랑", "으로",
              "에게", "한테", "보다", "처럼", "만큼",
              "은", "는", "이", "가", "을", "를", "의", "도", "만", "로",
              "과", "와", "랑", "에")
# 서술어 꼬리도 뗀다 — '찐반이다'가 '찐반'과 다른 말로 세어져서 불용어를
# 넣어도 계속 새로 올라온다. 여러 글자짜리만 뗀다: '다' '네' 같은 한 글자를
# 떼기 시작하면 멀쩡한 이름이 잘린다.
_ENDINGS = ("입니다", "이라고", "이다", "이네", "이냐", "이노", "인가", "인데",
            "하다", "한다", "했다", "된다", "되냐", "이야", "임")

# 매칭 안 된 단어 목록에서 뺄 말 — 종목 이름이 될 수 없는 것들.
# 완전할 수 없고 완전할 필요도 없다: 이 목록의 쓰임은 사람이 훑어보고
# aliases에 넣을 후보를 고르는 것뿐이라, 조금 새어도 눈으로 거르면 된다.
# 두 글자 말은 조사를 안 뗀다(떼면 한 글자가 된다) — 아래 마지막 두 줄이
# 그렇게 통째로 남는 흔한 말들이다. 문자열 안이라 여기에 #주석은 못 쓴다.
STOPWORDS = frozenset("""
지금 오늘 내일 어제 진짜 그냥 이거 저거 근데 아니 이제 다시 계속 아직 벌써
사람 새끼 시발 씨발 존나 ㅋㅋㅋ 개꿀 매수 매도 손절 익절 물타기 불타기 청산
상승 하락 반등 조정 돌파 이탈 지지 저항 추세 차트 분석 전망 예측 신호 타점
갤러리 형들 님들 여러분 질문 정보 뉴스 속보 실시간 오늘의 어떻게 얼마나
가즈아 가자 간다 온다 왔다 갔다 오름 내림 떡락 떡상 존버 단타 스캘핑
포지션 레버리지 선물 현물 지갑 거래소 수수료 물량 세력 개미 고래 큰손
코인 알트 코인턴 환율 뭐냐 미장 국장 증시 시황 장중 종가 시가 나스닥 지수
결국 뭔가 이렇게 그래서 그러면 사실 요즘 방금 잠깐 얼마 이번 다음 지듣노
차갤 세상 생각 이야기 사진 영상 오늘밤 새벽 주식 상장 운지 사람들 찐반 잡코장
한국 이전 주갤 시장 풀숏 풀롱 계좌 수익 손실 오늘자 실화 근황
나만 일차 시간 만에 하루 이틀 오전 오후 정도 지난 다들 진입 손가락
불로 클래리티 다큰낙타 반도체 바낸 빗썸 달러 코스피 코스닥 재능 저점 고점
시간봉 거래량 갑자기 기준 이유 모든 왤케 게이 부럽다 만들기 공유함 숏을 돈을
무빙 계단식 뜬금없 쳐라 지지선 저항선 추세선 이평선 캔들 봉임 매물대
너무 내가 나는 저는 이건 저건 그건 뭐지 같은 정말 아주 매우 하고 하는 해서
인데 라고 되면 하면 인가 인지 있다 없다 한다 된다 언제 어디 여기 거기
""".split())


def _hangul_words(text: str) -> list[str]:
    """한글 덩어리를 낱말로 끊어 돌려준다 — 조사를 떼고 2~6자만 남긴다.

    ❓ 후보 목록에만 쓴다. 완벽한 형태소 분석이 아니라, 사람이 훑어볼 목록에서
    조각과 조사 붙은 말을 걷어내는 정도면 충분하다.
    """
    out = []
    for run in _HANGUL.findall(text):
        for tails in (_ENDINGS, _PARTICLES):
            for tail in tails:
                if len(run) > len(tail) + 1 and run.endswith(tail):
                    run = run[:-len(tail)]
                    break
        # 띄어쓰기 때문에 조사가 홀로 떨어져 나온 것('4h 에서')은 버린다.
        # 두 글자라 위에서 안 떼이고, 그냥 두면 ❓ 상위를 먹는다.
        if 2 <= len(run) <= 6 and run not in _PARTICLES:
            out.append(run)
    return out


def _get(session, url, params=None, headers=None, timeout=15):
    r = session.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r


def apewisdom(session: requests.Session, name: str, pages: int = 1,
              log=print) -> list[dict]:
    """ApeWisdom 필터 하나를 가져온다 — 언급 수 순으로 이미 정렬돼 있다.

    name은 `all-crypto` `CryptoCurrency` `all-stocks` `wallstreetbets` `stocks`
    같은 값이다. 크립토 티커는 `BTC.X` 꼴로 오므로 접미사를 떼서 돌려준다.

    실패는 빈 목록 — 이 칸은 부가 정보라 스캔 전체를 세울 이유가 없다.
    """
    out = []
    try:
        for page in range(1, max(1, pages) + 1):
            data = _get(session, APEWISDOM.format(name=name, page=page)).json()
            for x in data.get("results") or []:
                ticker = str(x.get("ticker") or "").upper()
                if ticker.endswith(".X"):        # 크립토 표기
                    ticker = ticker[:-2]
                if not ticker:
                    continue
                out.append({
                    "ticker": ticker,
                    "name": html.unescape(str(x.get("name") or "")),
                    "mentions": int(x.get("mentions") or 0),
                    "prev": int(x.get("mentions_24h_ago") or 0),
                    "rank": int(x.get("rank") or 0),
                })
            if page >= int(data.get("pages") or 1):
                break
    except Exception as e:                       # noqa: BLE001
        log(f"[community] ApeWisdom {name} 실패: {type(e).__name__} {e}")
    return out


def _rss(session, url, params, tries: int = 4, pause: float = 5.0):
    """레딧 RSS는 **429를 자주 뱉는다** — 데이터센터 IP를 짜게 준다.

    막힌 게 아니라 잠깐 밀리는 것이다. 실측하니 **10~15초면 풀린다** — 그래서
    5·10·15초로 쉬며 네 번까지 물어본다. 그래도 안 되면 예외를 그대로 올려
    호출부가 빈 목록으로 넘긴다(숫자 줄은 ApeWisdom에서 오니 인용문만 빠진다).
    """
    for i in range(tries):
        try:
            return _get(session, url, params, REDDIT_HEADERS)
        except requests.HTTPError as e:
            if i == tries - 1 or getattr(e.response, "status_code", 0) != 429:
                raise
            time.sleep(pause * (i + 1))


def reddit_titles(session: requests.Session, sub: str, sort: str = "top",
                  limit: int = 25, log=print) -> list[str]:
    """서브레딧에서 **글 제목**을 긁는다 — 무인증으로 되는 유일한 길이 RSS다.

    `/hot.json`은 403, old.reddit은 로그인으로 튕긴다. `.rss`만 200으로 열린다
    (드물게 429가 나므로 실패는 빈 목록으로 넘긴다 — 숫자 줄은 ApeWisdom에서
    따로 오니 인용문만 빠진다).

    기본 정렬은 `top`(하루치)이다 — `hot`은 고정 공지가 섞이고, `new`는 아직
    아무도 안 본 글이 온다. 하루치 상위가 '오늘 실제로 반응이 온 글'이다.

    고정 메가스레드(`Daily Discussion`, `Rate My Portfolio`…)는 뺀다. 매번
    맨 위에 있어서, '지금 무슨 얘기 중인가'를 보려는 목적에 아무 정보가 없다.
    """
    out = []
    try:
        params = {"limit": limit}
        if sort == "top":
            params["t"] = "day"          # 하루치 상위 = 오늘 실제로 반응이 온 글
        r = _rss(session, REDDIT_RSS.format(sub=sub, sort=sort), params)
        for entry in _ENTRY.findall(r.text):
            m = _ENTRY_TITLE.search(entry)
            if not m:
                continue
            title = html.unescape(_TAG.sub("", m.group(1))).strip()
            if title and not _MEGA.search(title):
                out.append(title)
    except Exception as e:                       # noqa: BLE001
        log(f"[community] 레딧 r/{sub} 실패: {type(e).__name__} {e}")
    return out


def dc_titles(session: requests.Session, gallery: str, pages: int = 6,
              minor: bool = True, log=print) -> list[str]:
    """디시 갤러리 목록에서 **글 제목만** 긁는다. 공지는 뺀다.

    본문은 안 연다 — 글 하나에 요청이 하나씩 더 붙어서, 얻는 것에 비해
    스캔이 너무 느려진다. 제목만으로도 어떤 종목이 불리는지는 보인다.
    """
    seg = "mgallery/board" if minor else "board"
    url = DC_LIST.format(seg=seg)
    out = []
    for page in range(1, max(1, pages) + 1):
        try:
            r = _get(session, url, {"id": gallery, "page": page}, DC_HEADERS)
        except Exception as e:                   # noqa: BLE001
            log(f"[community] DC {gallery} {page}p 실패: {type(e).__name__} {e}")
            break
        if "폐쇄되었습니다" in r.text:
            log(f"[community] DC {gallery} 갤러리가 폐쇄됐다 — 설정을 고쳐라")
            break
        for row in _ROW.findall(r.text):
            if "icon_notice" in row:
                continue
            m = _TITLE.search(row)
            if not m:
                continue
            title = html.unescape(_TAG.sub("", m.group(1))).strip()
            if title:
                out.append(title)
    return out


def stocktwits_trending(session: requests.Session, log=print) -> list[str]:
    """스톡트윗에서 지금 뜨는 심볼 목록. 크립토는 `BTC.X` 꼴로 온다."""
    try:
        data = _get(session, STOCKTWITS.format(path="trending/symbols"),
                    headers=REDDIT_HEADERS).json()
    except Exception as e:                       # noqa: BLE001
        log(f"[community] 스톡트윗 트렌딩 실패: {type(e).__name__} {e}")
        return []
    out = []
    for x in data.get("symbols") or []:
        t = str(x.get("symbol") or "").upper()
        if t.endswith(".X"):
            t = t[:-2]
        if t:
            out.append(t)
    return out


def stocktwits_stream(session: requests.Session, symbol: str, limit: int = 30,
                      log=print) -> list[tuple[str, str]]:
    """한 종목 글타래 — (강세/약세/빈칸, 본문) 목록.

    스톡트윗은 글쓴이가 **강세·약세를 직접 달아** 준다. 우리가 문장을
    해석해서 추측하는 게 아니라 본인이 붙인 꼬리표라, 이 칸에서 '어떤
    의견인가'를 말할 수 있는 유일하게 정직한 숫자다(안 단 글이 더 많다).
    """
    try:
        data = _get(session, STOCKTWITS.format(path=f"streams/symbol/{symbol}"),
                    {"limit": limit}, REDDIT_HEADERS).json()
    except Exception as e:                       # noqa: BLE001
        log(f"[community] 스톡트윗 {symbol} 실패: {type(e).__name__} {e}")
        return []
    out = []
    for m in data.get("messages") or []:
        sent = (((m.get("entities") or {}).get("sentiment")) or {}).get("basic") or ""
        out.append((sent, clean_body(str(m.get("body") or ""))))
    return out


_CASHTAG = re.compile(r"\$[A-Za-z][A-Za-z0-9.]{0,9}")
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")
# `[14:30] BTC $78791 …` 같은 시세 봇 글 — 매시간 올라오고 의견이 아니다.
_BOTPOST = re.compile(r"^\[\d{1,2}:\d{2}\]")


def clean_body(body: str) -> str:
    """글머리 티커 나열·링크를 걷어낸 본문 — 한 줄에 실을 수 있게."""
    body = _URL.sub("", body)
    body = _CASHTAG.sub("", body)
    return _WS.sub(" ", body).strip(" -–—:·,")


def clip(text: str, n: int = 70) -> str:
    text = _WS.sub(" ", text).strip()
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


# BTC·ETH는 어느 커뮤니티에서나 늘 1·2위다. 인용 한 줄을 줘도 새로 아는 게
# 없어서, 자리가 남을 때만 싣는다 — 정작 보고 싶은 건 알트 쪽이다.
DEMOTE = ("BTC", "ETH")


def pick_quotes(titles: list[str], tickers, universe: set[str],
                aliases: dict, n: int = 3, width: int = 70) -> list[str]:
    """'무슨 얘기 중인가'로 실을 제목 몇 개 — **티커마다 한 줄씩 골고루**.

    숫자만 있으면 왜 그 종목이 불리는지 알 수 없어서 글을 같이 싣는데,
    그냥 '티커가 걸린 글 먼저'로 뽑으면 **BTC 얘기가 세 줄을 다 먹는다**.
    제일 많이 불린 이름이 후보도 제일 많기 때문이다. 그래서 티커마다 한 줄씩
    주고, BTC·ETH는 뒤로 미룬다(DEMOTE). 자리가 남으면 같은 티커의 둘째 줄,
    그래도 남으면 티커가 안 걸린 글로 채운다.

    제목에 티커 글자가 없으면(`월드숏 좀 맛있네`) 앞에 티커를 붙여 준다 —
    어느 종목 얘기인지 알아야 줄이 쓸모가 있다.
    """
    alias_map = normalize_aliases(aliases)
    order = [str(t).upper() for t in tickers]
    universe = set(universe) | set(order)
    by_ticker, rest, seen = {}, [], set()
    for title in titles:
        t = _WS.sub(" ", title).strip()
        if len(t) < 8:                           # 한두 단어짜리는 의견이 아니다
            continue
        if _BOTPOST.match(t):                    # 시세 봇 글은 의견이 아니다
            continue
        # **첫 낱말이 같으면 한 줄만 쓴다.** 같은 화제가 한꺼번에 올라오면
        # ('테더 2000 가는거…', '테더 근데 일정액은…', '테더 기본 몇십억은…')
        # 인용 세 줄을 통째로 먹어서 다른 얘기가 안 보인다. 글은 남아도니
        # 하나만 보여 주고 나머지 자리는 다른 화제로 채우는 게 낫다.
        key = t.lower().split()[0].strip(".,!?…\"'")
        if key in seen:
            continue
        seen.add(key)
        found = _match_title(t, universe, alias_map)[0]
        hit = [x for x in order if x in found]
        if hit:
            # 여러 종목이 걸리면 **뒤쪽(덜 불린 쪽)**에 준다 — 흔한 티커는
            # 어차피 자기 줄이 따로 있다.
            who = hit[-1]
            body = clip(t, width)
            by_ticker.setdefault(who, []).append(
                body if who.lower() in t.lower() else f"{who} {body}")
        else:
            rest.append(clip(t, width))

    ranked = ([x for x in order if x not in DEMOTE and x in by_ticker]
              + [x for x in order if x in DEMOTE and x in by_ticker])
    out = [by_ticker[x][0] for x in ranked][:n]           # 티커마다 한 줄 먼저
    if len(out) < n:                                     # 남으면 둘째 줄들로
        out += [q for x in ranked for q in by_ticker[x][1:]][:n - len(out)]
    if len(out) < n:
        out += rest[:n - len(out)]
    return out


def normalize_aliases(aliases) -> dict[str, str]:
    """`별명 → 티커` 사전을 소문자 키로 정규화한다.

    str()로 감싸는 이유: YAML이 따옴표 없는 국내 종목코드(000660)를 8진수
    정수로 읽어 버린다. 값이 틀리는 건 설정에서 고쳐야 하지만, 여기서
    터져서 커뮤니티 칸이 통째로 사라지는 건 막는다.
    """
    return {str(k).lower(): str(v).upper() for k, v in (aliases or {}).items()}


def _match_title(title: str, universe: set[str], alias_map: dict[str, str]):
    """제목 하나에서 찾은 티커 집합과, 매칭에 쓰인 별명 목록."""
    found, used = set(), []
    for word in _LATIN.findall(title):
        upper = word.upper()
        # 두 글자 티커는 **대문자로 썼을 때만** 인정한다. 안 그러면 디시 글의
        # `주소창에 id=stock`이 ID(Space ID) 언급으로 잡힌다. 세 글자부터는
        # 소문자도 받는다 — 'btc 존버'처럼 실제로 그렇게 쓴다.
        if len(word) <= 2 and not word.isupper():
            continue
        if upper in universe:
            found.add(upper)
        elif word.lower() in alias_map:
            found.add(alias_map[word.lower()])
    low = title.lower()
    for alias, ticker in alias_map.items():
        if alias in low:
            found.add(ticker)
            used.append(alias)
    return found, used


def count_mentions(titles: list[str], universe: set[str],
                   aliases: dict[str, str]) -> tuple[collections.Counter,
                                                     collections.Counter]:
    """제목 목록에서 티커 언급을 센다. (언급 수, 매칭 안 된 한글 단어)

    한 제목에 같은 종목이 여러 번 나와도 **한 번으로 센다** — 제목은 짧아서
    반복이 강조지 별개 언급이 아니다.

    universe는 대문자 티커 집합(예: 퍼프 심볼에서 USDT를 뗀 것)이고,
    aliases는 `별명 → 티커` 사전이다. 별명은 대소문자·한글 모두 받는다.

    두 번째 Counter는 **어느 티커에도 안 걸린 한글 단어**다. 사전을 채우는
    데 쓰라고 같이 돌려준다 — 국내 소스는 이 사전이 자라야 쓸모가 생긴다.
    """
    hits = collections.Counter()
    unmatched = collections.Counter()
    alias_map = normalize_aliases(aliases)
    for title in titles:
        found, used = _match_title(title, universe, alias_map)
        hits.update(found)
        rest = title
        for alias in used:                       # 이미 매칭된 별명은 후보에서 뺀다
            rest = re.sub(re.escape(alias), " ", rest, flags=re.I)
        for word in _hangul_words(rest):
            if word not in STOPWORDS:
                unmatched[word] += 1
    return hits, unmatched
