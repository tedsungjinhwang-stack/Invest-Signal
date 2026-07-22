# Invest-Signal

4시간봉 마감마다 **바이낸스 전체 USDT 무기한 + 레버리지 ETF**를 스캔해서
시그널이 뜨면 **텔레그램으로 알림**을 보내는 스캐너.

## 시그널 (모두 4h봉 기준)

### 상승초입 (uptrend_onset)

1. **역배열** — `MA120 < MA240 < MA480`
2. **240선 터치** — 캔들 고가가 240선에 닿음 (윗꼬리 터치 인정)
3. **60선 이탈** — 터치 후 `touch_window_bars`(기본 60봉 = 10일) 이내에
   종가가 60선 아래로 마감하는 **첫 봉**
4. **분기 VWAP 필터** — 트리거 봉 종가가 분기 앵커드 VWAP(QVWAP) 위여야
   함 (`qvwap_condition: false`로 끌 수 있음)

같은 셋업으로는 한 번만 알리고, 240선을 다시 터치하면 새 셋업으로 다시
알릴 수 있다.

### 눌림목 (pullback)

1. **240선 상향돌파** — 종가가 240선을 뚫고 올라간 뒤
   `breakout_window_bars`(기본 180봉 = 30일) 이내이고, 돌파 후 종가가
   240선 위를 유지하는 동안
2. **120선 하회** — 종가가 120선 아래로 마감하는 순간 알림

연속 하회 구간은 첫 봉만 알리고, 120선 위로 복귀 후 다시 하회하면 새
눌림목으로 다시 알린다.

새 시그널은 `invest_signal/signals/`에 모듈을 추가하고
`config.py`의 `detectors()`에 등록하면 된다.

## 감시 대상

- **크립토**: 바이낸스에서 거래 중인 전체 USDT 무기한 (자동 갱신)
- **레버리지 ETF**: `config.yaml`의 18종 — 미국 섹터·지수 불/베어
  (SOXL/SOXS, TQQQ/SQQQ, FAS, LABU, ERX, NUGT, YINN, RETL, TMF, DFEN,
  BITU, ETHU) + 한국 (KODEX 레버리지/코스닥150레버리지 및 인버스2X 쌍)

## 바이낸스 지역 차단(451)과 데이터 소스

GitHub Actions 러너(미국 IP) 실측 결과:

| 엔드포인트 | 결과 |
|---|---|
| `fapi.binance.com` (선물) | ❌ HTTP 451 |
| `api.binance.com` (현물) | ❌ HTTP 451 |
| `data-api.binance.vision` (공식 현물 데이터 미러) | ✅ 200 |

`crypto.source: auto`(기본)는 **fapi 시도 → 실패하면 현물 미러로 자동
폴백**한다. 현물·무기한의 4h 가격은 MA 시그널 용도로는 사실상 동일하다.
단, 커버리지 차이가 있다(실측): 활성 USDT 퍼프 787종 중 현물로 커버되는
건 377종(48%) — 나머지는 토큰화 주식 퍼프, 퍼프 전용 밈코인, 현물 상폐
코인 등. BTC·ETH·SOL 등 주요 코인은 전부 커버된다. 진짜 무기한 데이터로
돌리려면:

- `BINANCE_FAPI_BASE` 시크릿에 비미국 프록시 주소를 넣거나
- 한국 PC/서버에서 crontab으로 실행 (`--only crypto`)

## 설정

### 1. 텔레그램 봇 (필수)

1. [@BotFather](https://t.me/BotFather)에서 봇 생성 → 토큰 복사
2. 봇에게 아무 메시지 전송 후 `https://api.telegram.org/bot<토큰>/getUpdates`에서
   `chat.id` 확인
3. 레포 Settings → Secrets and variables → Actions에 등록:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`

시크릿이 비어 있으면 스캔 워크플로가 **즉시 실패**한다(빨간 런) — 알림이
조용히 증발하는 것을 막기 위한 안전장치다. 발송에 실패한 시그널은 상태에
기록하지 않으므로 다음 4h 스캔에서 한 번 더 시도된다(`grace_bars`).

### 2. 스케줄

`.github/workflows/scan.yml`이 4h봉 마감 7분 후(UTC 00/04/08/12/16/20시 =
KST 09/13/17/21/01/05시)에 자동 실행된다.
**schedule 트리거는 기본 브랜치에서만 동작**하므로 이 브랜치를 main에
머지해야 활성화된다. 중복 알림은 `state/alerts_state.json`으로 방지하며
실행 후 자동 커밋된다.

### 3. 로컬 실행

```bash
pip install -r requirements.txt
python -m invest_signal --dry-run          # 발송 없이 결과만 출력
python -m invest_signal --only crypto      # 크립토만
TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... python -m invest_signal
```

### 4. config.yaml

```yaml
signal:
  uptrend_onset:
    enabled: true
    touch_window_bars: 60      # 터치 유효기간(봉) — 60봉 = 10일
    grace_bars: 1              # 스캔 한 번 놓쳤을 때 소급 허용 봉 수
    qvwap_condition: true      # 트리거 종가 > 분기 VWAP 요구
  pullback:
    enabled: true
    breakout_window_bars: 180  # 240선 돌파 유효기간 — 180봉 = 30일
    grace_bars: 1
crypto:
  source: auto                 # auto | fapi | spot_mirror
  exclude: []                  # 제외 심볼
etf:
  tickers: [...]               # 코드/시장/이름
```

의존성은 `requirements.txt`에 정확히 고정돼 있다(무인 크론이라 상위
버전이 갑자기 깨는 것을 방지). 올릴 때는 테스트 후 갱신할 것.

## 테스트

```bash
pip install -r requirements-dev.txt
pytest
```

푸시하면 `Test & Dry-Run` 워크플로가 pytest + 실데이터 dry-run으로 검증한다.
