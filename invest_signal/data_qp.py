"""퀀트포트폴리오 모멘텀 리포트에서 감시 주식을 동적으로 로드.

reports/latest.md는 매 거래일 아침 갱신되므로, 스캔 때마다 GitHub API로
읽어 파싱한다(QP_GITHUB_TOKEN 필요 — Quant-Portfolio 읽기 권한 PAT).
가져오는 것: 미국·한국 각각 트랙①(강한 섹터 대장주 2×3 = 6종) +
트랙②(개별 RS Top5). 중복 코드는 한 번만.
"""

import re

import requests

QP_REPO = "tedsungjinhwang-stack/Quant-Portfolio"
REPORT_PATH = "reports/latest.md"
RS_TOP_N = 5

# 종목표기: 이름(티커) — 미국은 알파벳 티커, 한국은 6자리 코드
_PICK = r"([^,:|]+?)\(([A-Z][A-Z0-9\.]{0,5}|\d{6})\)"


def fetch_report(token: str, timeout: int = 20) -> str:
    url = f"https://api.github.com/repos/{QP_REPO}/contents/{REPORT_PATH}"
    rsp = requests.get(url, timeout=timeout, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.raw+json",
    })
    rsp.raise_for_status()
    return rsp.text


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
        out.append({"code": code, "market": market,
                    "name": re.sub(r"[*🟢🔵]|풀백", "", name).strip()})

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
