"""퀀트포트폴리오 리포트 파싱 테스트 — 실제 리포트 형식 그대로."""

from invest_signal.data_qp import parse_report

SAMPLE = """# 2트랙 모멘텀 리포트 — 2026-07-21

## 🟢 미국

### 트랙① 강한 섹터 → 대장주
- **사이버보안(CIBR)** [CIBR RS 19.2]: Fortinet(FTNT)🟢풀백, Palo Alto Networks(PANW)
- **반도체(SMH)** [SMH RS 15.9]: Micron Technology(MU), Advanced Micro Devices(AMD)🟢풀백
- **석유 E&P(XOP)** [XOP RS 12.4]: OVV(OVV), EOG Resources(EOG)

### 트랙② 개별 RS Top10 (RS=지수 대비)
| RS# | 종목 | 종가 | 1D% |
|---|---|---|---|
| 1 | Dell Technologies(DELL) | 404.15 | +5.83 |
| 2 | Sandisk(SNDK) | 1,589.4 | +14.27 |
| 3 | Micron Technology(MU) | 970.82 | +12.17 |
| 4 | Advanced Micro Devices(AMD) | 544.43 | +8.11 |
| 5 | Datadog(DDOG) | 254.79 | -3.20 |
| 6 | Seagate Technology(STX) | 891.83 | +11.14 |

## 🔵 한국

### 트랙① 강한 섹터 → 대장주
- **반도체** [091160 RS 16.2]: SK하이닉스(000660), 삼성전자(005930)
- **보험** [140700 RS 11.4]: 삼성생명(032830), 현대해상(001450)🟢풀백

### 트랙② 개별 RS Top10 (RS=지수 대비)
| RS# | 종목 | 종가 | 1D% |
|---|---|---|---|
| 1 | 주성엔지니어링(036930) | 169,200.0 | +0.77 |
| 2 | 삼성전기(009150) | 1,310,000.0 | +2.75 |
| 4 | 010120(010120) | 220,500.0 | +15.69 |
| 6 | 기가비스(420770) | 142,600.0 | -0.28 |
"""


def test_parse_report_extracts_sector_and_rs_picks():
    picks = parse_report(SAMPLE)
    codes = [p["code"] for p in picks]
    # 미국 섹터 6 + RS(중복 MU·AMD 제외) 3 = 9
    for c in ["FTNT", "PANW", "MU", "AMD", "OVV", "EOG", "DELL", "SNDK", "DDOG"]:
        assert c in codes
    assert codes.count("MU") == 1                    # 중복 제거
    assert "STX" not in codes                        # RS 6위는 제외
    assert "420770" not in codes
    # 한국
    for c in ["000660", "005930", "032830", "001450", "036930", "009150"]:
        assert c in codes
    by = {p["code"]: p for p in picks}
    assert by["000660"]["market"] == "KR" and by["FTNT"]["market"] == "US"
    assert by["FTNT"]["name"] == "Fortinet"          # 🟢풀백 표기 제거
    assert by["000660"]["name"] == "SK하이닉스"
    # 섹터 라벨(CIBR 등 괄호 티커)이 종목으로 섞이면 안 됨
    assert "CIBR" not in codes and "SMH" not in codes and "XOP" not in codes
    # 리포트가 이름을 못 찾아 '코드(코드)'로 쓴 경우 → 이름 없음으로 정규화
    assert by["OVV"]["name"] == ""
    assert "010120" in codes and by["010120"]["name"] == ""


def test_parse_report_empty_or_garbage():
    assert parse_report("") == []
    assert parse_report("# 아무 내용 없음\n그냥 텍스트") == []


def test_parse_etfmom_holdings():
    """테마포착 TOP10 — 로스컷 제외, 숫자 코드는 KR·티커는 US."""
    from invest_signal.data_qp import parse_etfmom

    book = {"holdings": [
        {"code": "XLE", "name": "에너지 섹터", "market": "US",
         "theme": "석유·가스", "losscut": False},
        {"code": "290080", "name": "RISE 200고배당커버드콜ATM", "market": "KR",
         "theme": "배당·인컴", "losscut": False},
        {"code": "FXI", "name": "중국 대형", "market": "US",
         "theme": "중국", "losscut": True},          # 로스컷 → 제외
    ], "last_reb": "2026-07-23"}
    out = parse_etfmom(book)
    assert [t["code"] for t in out] == ["XLE", "290080"]
    by = {t["code"]: t for t in out}
    assert by["XLE"]["market"] == "US"
    assert by["290080"]["market"] == "KR" and by["290080"]["name"].startswith("RISE")
    assert parse_etfmom({}) == []
