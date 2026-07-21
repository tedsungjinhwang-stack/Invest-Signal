# Invest-Signal

4시간봉 마감마다 **바이낸스 전체 USDT 무기한 + 레버리지 ETF**를 스캔해서
시그널이 뜨면 **텔레그램으로 알림**을 보내는 스캐너.

## 시그널: 상승초입 (4h봉)

1. **역배열** — 120·240·480 이동평균이 `MA120 < MA240 < MA480`
2. **240선 터치** — 캔들 고가가 240선에 닿음 (윗꼬리 터치 인정)
3. **60선 이탈** — 터치 후 `touch_window_bars`(기본 60봉 = 10일) 이내에
   종가가 60선 아래로 마감하는 **첫 봉**에서 알림

같은 셋업으로는 한 번만 알리고, 240선을 다시 터치하면 새 셋업으로 다시
알릴 수 있다. 알림에는 분기 앵커드 VWAP(QVWAP) 위/아래 여부도 같이 표시되며,
`require_above_qvwap: true`로 켜면 QVWAP 위에서 마감한 경우만 알린다.

> 풀백 시그널은 조건 확정 후 추가 예정 (`invest_signal/signals/`에 모듈만
> 추가하면 되는 구조).

## 감시 대상

- **크립토**: 바이낸스에서 거래 중인 전체 USDT 무기한 (자동 갱신)
- **레버리지 ETF**: `config.yaml`의 목록 — 퀀트포트폴리오 레포의
  LEV_PAIRS(불 사이드) + LEV_SINGLES 기준 26종 (미국 + 한국)

## 바이낸스 지역 차단(451)과 데이터 소스

GitHub Actions 러너는 미국 IP라 실측 결과:

| 엔드포인트 | 결과 |
|---|---|
| `fapi.binance.com` (선물) | ❌ HTTP 451 |
| `api.binance.com` (현물) | ❌ HTTP 451 |
| `data-api.binance.vision` (공식 현물 데이터 미러) | ✅ 200 |

그래서 `crypto.source: auto`(기본)는 **fapi 시도 → 451이면 현물 미러로 자동
폴백**한다. 현물·무기한의 4h 가격은 MA 시그널 용도로는 사실상 동일하지만,
무기한 전용 심볼(BTCDOMUSDT 등)은 미러에 없고 현물 유니버스로 스캔된다는
차이가 있다. 진짜 무기한 데이터로 돌리고 싶으면:

- `BINANCE_FAPI_BASE` 시크릿에 비미국 프록시 주소를 넣거나
- 한국 PC/서버에서 crontab으로 실행 (`--only crypto`)

## 설정

### 1. 텔레그램 봇

1. [@BotFather](https://t.me/BotFather)에서 봇 생성 → 토큰 복사
2. 봇에게 아무 메시지 전송 후 `https://api.telegram.org/bot<토큰>/getUpdates`에서
   `chat.id` 확인
3. 레포 Settings → Secrets and variables → Actions에 등록:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`

### 2. 스케줄

`.github/workflows/scan.yml`이 4h봉 마감 7분 후(UTC 00/04/08/12/16/20시,
KST 09/13/17/21/01/05시)에 자동 실행된다.
**schedule 트리거는 기본 브랜치에서만 동작**하므로 이 브랜치를 main에 머지해야
활성화된다. 중복 알림은 `state/alerts_state.json`으로 방지(실행 후 자동 커밋).

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
    touch_window_bars: 60      # 터치 유효기간(봉) — 60봉 = 10일
    grace_bars: 1              # 스캔 한 번 놓쳤을 때 소급 허용 봉 수
    require_above_qvwap: false # 분기 VWAP 위 마감만 알림
crypto:
  source: auto                 # auto | fapi | spot_mirror
  exclude: []                  # 제외 심볼
etf:
  tickers: [...]               # 코드/시장/이름
```

## 테스트

```bash
pip install -r requirements-dev.txt
pytest
```
