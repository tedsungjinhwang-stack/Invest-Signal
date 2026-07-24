"""퀀트포트폴리오 레포에서 감시 종목을 동적으로 로드.

- reports/latest.md: 매 거래일 갱신 — 미국·한국 각각 트랙①(강한 섹터
  대장주 2×3 = 6종) + 트랙②(개별 RS Top5). 중복 코드는 한 번만.
- reports/etfmom_book.json: ETF 모멘텀 TOP10(테마포착) 보유 ETF —
  리밸런싱마다 갱신되며 로스컷 표시된 종목은 제외.
"""

import json
import re

import requests

QP_REPO = "tedsungjinhwang-stack/Quant-Portfolio"
REPORT_PATH = "reports/latest.md"
ETFMOM_PATH = "reports/etfmom_book.json"    # ETF 모멘텀 TOP10(테마포착) 보유 목록
RS_TOP_N = 5

# 종목표기: 이름(티커) — 미국은 알파벳 티커, 한국은 6자리 코드
_PICK = r"([^,:|]+?)\(([A-Z][A-Z0-9\.]{0,5}|\d{6})\)"


def fetch_file(path: str, token: str | None = None, timeout: int = 20) -> str:
    """QP 레포 파일 원문 조회.

    토큰이 있으면 API(레이트리밋 여유, 프라이빗도 가능), 없으면 퍼블릭
    raw 경로(HEAD = 기본 브랜치 자동 추적)로 읽는다.
    """
    if token:
        rsp = requests.get(
            f"https://api.github.com/repos/{QP_REPO}/contents/{path}",
            timeout=timeout,
            headers={"Accept": "application/vnd.github.raw+json",
                     "Authorization": f"Bearer {token}"})
    else:
        rsp = requests.get(f"https://github.com/{QP_REPO}/raw/HEAD/{path}",
                           timeout=timeout)
    rsp.raise_for_status()
    return rsp.text


def fetch_report(token: str | None = None, timeout: int = 20) -> str:
    return fetch_file(REPORT_PATH, token, timeout)


def parse_report(md: str) -> list[dict]:
    """리포트 마크다운 → [{code, market, name}] (트랙① 전부 + 트랙② RS Top5).

    한국 종목의 코스피/코스닥 구분은 리포트에 없으므로 일단 KR(.KS)로 두고,
    데이터가 없으면 수집 단계에서 .KQ로 재시도한다(data_etf.fetch_all).
    """
    out, seen = [], set()

    def add(name: str, code: str):
        if code in seen:
            return
        seen.add(code)
        market = "KR" if code.isdigit() else "US"
        name = re.sub(r"[*🟢🔵]|풀백", "", name).strip()
        if name.upper() == code.upper():
            name = ""       # 리포트가 이름을 못 찾으면 '코드(코드)'로 쓴다 — 이름 없음 처리
        out.append({"code": code, "market": market, "name": name})

    in_section = False
    for line in md.splitlines():
        if line.startswith("## "):                    # 시장 섹션 헤더(레벨2)만
            in_section = ("미국" in line) or ("한국" in line)
            continue
        if not in_section:
            continue
        m = re.match(r"^-\s+\*\*.+?\*\*.*?:\s*(.+)$", line)   # 트랙① 섹터 대장주
        if m:
            for name, code in re.findall(_PICK, m.group(1)):
                add(name, code)
            continue
        m = re.match(r"^\|\s*(\d+)\s*\|\s*" + _PICK + r"\s*\|", line)   # 트랙② 표
        if m and int(m.group(1)) <= RS_TOP_N:
            add(m.group(2), m.group(3))
    return out


def parse_etfmom(data: dict) -> list[dict]:
    """etfmom_book.json → [{code, market, name}] — 로스컷 종목 제외."""
    out = []
    for h in data.get("holdings") or []:
        code = str(h.get("code", "")).strip()
        if not code or h.get("losscut"):
            continue
        market = "KR" if code.isdigit() else "US"
        out.append({"code": code, "market": market,
                    "name": (h.get("name") or "").strip()})
    return out


def theme_etfs(token: str | None = None) -> list[dict]:
    """ETF 모멘텀 TOP10(테마포착) 현재 보유 ETF 목록."""
    return parse_etfmom(json.loads(fetch_file(ETFMOM_PATH, token)))
